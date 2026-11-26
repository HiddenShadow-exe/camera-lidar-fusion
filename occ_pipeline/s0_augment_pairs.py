#!/usr/bin/env python3
"""STAGE 0 - build augmented samples from the recorded pairs (odd = occluded scene, even = ground truth).

Per target box (= one connected cluster of the EVEN recording's camera object points) a sample contains, all in the
LiDAR frame:
    cluster    camera points of the target box               (the only thing that gets sensor artifacts)
    origin     LiDAR origin                                   (moves with the rigid transform!)
    scan       OCCLUDED-scene LiDAR returns whose rays pass the region around the box (these define the rays)
    gt         GT LiDAR points of the target (even recording)
    gt_shadow  GT points that were hidden in the occluded scan (nearest ray of the occluded scan is much closer)
    cam_*      camera intrinsics / pose (unchanged by the augmentation: the camera does not move)

Augmentation = rotation about the VERTICAL axis through the camera centre (+ <=5 cm shift), applied rigidly to
cluster, scan, gt AND the LiDAR origin. Why this is exact: the camera looks straight down, so its view of a rotated
cluster is what a real camera would see; the LiDAR is rotated together with the scene, so in its own frame it measures
exactly what it measured before (same ranges, same channels, same incidence angles). Only the LiDAR<->camera pose
changes, and that is a legitimate new mounting pose. The rays of the augmented sample are rebuilt from the moved origin
to the moved scan points. LiDAR points are never noised - only the camera cluster gets depth noise / bleeding / dropout.
"""
import argparse
import json
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from common import (ART_DEFAULTS, Gates, Rig, augment_cloud, banner, find_pairs, in_fov_fraction, progress,
                    ray_box_interval, read_pcd_xyz, rot_z, slugify, split_scenes, step, write_csv)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="dataset root: <category>/recording_XX/{recording_data.json,pointcloud_lidar.pcd}")
    ap.add_argument("--work", default="work")
    ap.add_argument("--n-aug", type=int, default=20, help="augmented copies per target box")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-range", type=float, default=5.0, help="drop LiDAR points beyond this range [m] (as in the live script)")
    ap.add_argument("--gt-source", choices=["object", "scene"], default="object",
                    help="object: 3d_data_lidar.object_points of the even recording; scene: its whole pointcloud_lidar.pcd above the floor")
    ap.add_argument("--floor-z", type=float, default=-0.40, help="floor height in the LiDAR frame (= -LIDAR_HEIGHT)")
    ap.add_argument("--ground-margin", type=float, default=0.08)
    ap.add_argument("--crop-margin", type=float, default=0.40, help="keep rays/GT within this xy margin around the cluster [m] (must exceed CV error + shift range)")
    ap.add_argument("--crop-margin-z", type=float, default=0.20, help="same in z (must cover the bottom/top shift range)")
    ap.add_argument("--shadow-ang-deg", type=float, default=0.5, help="angular tolerance to match a GT point to a ray of the occluded scan")
    ap.add_argument("--shadow-margin", type=float, default=0.05, help="GT point is shadowed if the matched occluded ray is closer by more than this [m]")
    ap.add_argument("--min-shadow-pts", type=int, default=10, help="drop boxes with fewer shadowed GT points")
    ap.add_argument("--cluster-eps", type=float, default=0.05)
    ap.add_argument("--cluster-min-pts", type=int, default=150)
    ap.add_argument("--cluster-voxel", type=float, default=0.01)
    ap.add_argument("--max-rot-deg", type=float, default=180.0)
    ap.add_argument("--max-shift", type=float, default=0.05)
    ap.add_argument("--fov-margin", type=int, default=2)
    ap.add_argument("--tries", type=int, default=40)
    for k, v in ART_DEFAULTS.items():
        if isinstance(v, (int, float)):
            ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--exclude-units", nargs="*", default=[], metavar="UNIT",
                    help="target boxes that are not valid evaluation targets, written <pair slug>_u<k>, e.g. occluded_boxes_07_u1 (a camera cluster without a label / a phantom). "
                         "Whole scenes or categories are better excluded by renaming the folder with a leading underscore")
    ap.add_argument("--only", type=int, default=0, help="debug: process only the first N pairs")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


# ----------------------------------------------------------------------------------------------
def clip_range(p, max_range):
    return p[np.linalg.norm(p, axis=1) < max_range]


def split_clusters(points, eps, min_pts, vox):
    """Connected components (radius graph on a 1 cm voxel grid). -> list of point arrays."""
    keys = np.floor(points / vox).astype(np.int64)
    _, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    cen = points[first]
    pairs = cKDTree(cen).query_pairs(eps, output_type="ndarray")
    n = len(cen)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    out = []
    for c in np.unique(lab):
        sel = lab == c
        if sel.sum() >= min_pts:
            out.append(points[sel[inv]])
    return sorted(out, key=lambda x: -len(x))


def shadow_mask(gt, scan, ang_deg, margin):
    """GT point is 'shadowed' if the occluded scan has a return in (almost) the same direction that is clearly closer."""
    rg, rs = np.linalg.norm(gt, axis=1), np.linalg.norm(scan, axis=1)
    tree = cKDTree(scan / rs[:, None])
    dist, idx = tree.query(gt / rg[:, None], k=1, distance_upper_bound=2 * np.sin(np.deg2rad(ang_deg) / 2))
    found = np.isfinite(dist)
    sh = np.zeros(len(gt), bool)
    sh[found] = rs[idx[found]] < rg[found] - margin
    return sh


def load_pair(pair, a):
    odd = json.loads((pair.odd_dir / "recording_data.json").read_text())
    even = json.loads((pair.even_dir / "recording_data.json").read_text())
    hw = even["2d_data"]["image_dimensions"]
    rig, rig_e = Rig.from_calib(odd["calibration"], hw), Rig.from_calib(even["calibration"], hw)
    diag = {"calib_dpos_cm": float(np.linalg.norm(rig.t - rig_e.t) * 100),
            "calib_dyaw_deg": float(np.rad2deg(abs(odd["calibration"]["lidar_to_cam_extrinsic"]["lidar_yaw"]
                                                    - even["calibration"]["lidar_to_cam_extrinsic"]["lidar_yaw"])))}

    scan_odd = clip_range(read_pcd_xyz(pair.odd_dir / "pointcloud_lidar.pcd"), a.max_range)
    scan_even = clip_range(read_pcd_xyz(pair.even_dir / "pointcloud_lidar.pcd"), a.max_range)
    obj = np.asarray(even["3d_data_lidar"]["object_points"], float)
    if obj.ndim != 2 or len(obj) == 0:
        raise ValueError("no 3d_data_lidar.object_points in the even recording")
    obj = obj[:, :3]                                           # json rows are x, y, z, <4th column ignored>
    # frame check: the json LiDAR points must be (almost) a subset of the even .pcd -> same frame / same scan
    d, _ = cKDTree(scan_even).query(obj)
    diag["json_vs_pcd_med_mm"] = float(np.median(d) * 1000)
    gt = clip_range(obj, a.max_range) if a.gt_source == "object" else scan_even[scan_even[:, 2] > a.floor_z + a.ground_margin]

    cam = np.asarray(even["3d_data_camera"]["object_points"], float).reshape(-1, 3)
    diag["cam_frame"] = "lidar"
    if np.mean(cam[:, 2] < 0) <= 0.005:                       # same heuristic as augment_obb_dataset.py: camera frame
        cam = (cam - rig.t) @ rig.R.T
        diag["cam_frame"] = "camera->lidar"
    cam = cam[cam[:, 2] > a.floor_z + a.ground_margin]
    if len(cam) < a.cluster_min_pts:
        raise ValueError("camera cluster empty after ground removal")

    shadow = shadow_mask(gt, scan_odd, a.shadow_ang_deg, a.shadow_margin)
    units = []
    for ci, cl in enumerate(split_clusters(cam, a.cluster_eps, a.cluster_min_pts, a.cluster_voxel)):
        mg = np.array([a.crop_margin, a.crop_margin, a.crop_margin_z])
        lo, hi = cl.min(0) - mg, cl.max(0) + mg
        rs = np.linalg.norm(scan_odd, axis=1)
        tin, tout = ray_box_interval(np.zeros(3), scan_odd / rs[:, None], lo, hi)
        scan_c = scan_odd[(tout >= tin) & (tout > 0)]
        gsel = np.all((gt >= lo) & (gt <= hi), axis=1)
        units.append(dict(idx=ci, cluster=cl, scan=scan_c, gt=gt[gsel], gt_shadow=shadow[gsel], rig=rig))
    return units, diag


def pick_transform(cluster, rig, rng, a):
    pivot = np.array([rig.cam_center[0], rig.cam_center[1], 0.0])
    base = in_fov_fraction(rig, cluster, a.fov_margin)
    for _ in range(a.tries):
        phi = np.deg2rad(rng.uniform(-a.max_rot_deg, a.max_rot_deg))
        r, th = a.max_shift * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        shift = np.array([r * np.cos(th), r * np.sin(th), 0.0])
        Rz = rot_z(phi)
        if in_fov_fraction(rig, (cluster - pivot) @ Rz.T + pivot + shift, a.fov_margin) >= base - 1e-9:
            return (lambda X: (X - pivot) @ Rz.T + pivot + shift), phi, shift
    return None


def main():
    a = parse()
    cfg = {k: getattr(a, k) for k, v in ART_DEFAULTS.items() if isinstance(v, (int, float))}
    cfg.update({k: v for k, v in ART_DEFAULTS.items() if not isinstance(v, (int, float))})
    out = Path(a.work) / "stage0"
    banner(0, "augment recorded pairs (cluster + LiDAR rays + GT, rigid transform)")
    if (out / "DONE").exists() and not a.force:
        print(f"  {out}/DONE exists -> skipping (use --force to redo)")
        return
    rng = np.random.default_rng(a.seed)
    gates = Gates(0)

    step("1/5 discovering recording pairs")
    pairs = find_pairs(a.input)
    if a.only:
        pairs = pairs[:a.only]
    if not pairs:
        raise SystemExit(f"no (odd, even) recording pairs found under {a.input}")
    print(f"     {len(pairs)} pairs in {len({p.scene for p in pairs})} scenes: "
          + ", ".join(f"{c}={n}" for c, n in Counter(p.category for p in pairs).items()))
    is_virtual = lambda p: p.category.lower().startswith("virtual")                 # make_virtual_pairs.py output: training data only
    val_scenes = split_scenes({p.scene for p in pairs if not is_virtual(p)}, a.val_fraction, a.split_seed)

    rows, skipped, diags, rigid_err, first_tb = [], [], [], 0.0, None
    t0 = time.time()
    step("2/5 loading pairs, shadow masks, clusters  +  3/5 augmenting")
    bar = progress(pairs, total=len(pairs), desc="pairs")
    for pair in bar:
        try:
            units, diag = load_pair(pair, a)
        except Exception as e:                                   # one bad pair must not kill the run, but must be visible
            if first_tb is None:
                first_tb = traceback.format_exc()
            skipped.append((pair.scene, f"load error: {e!r}"))
            print(f"\n  [warn] {pair.scene}: {e!r}")
            continue
        diags.append({"scene": pair.scene, **diag})
        split = "val" if pair.scene in val_scenes else "train"
        for u in units:
            if f"{pair.slug}_u{u['idx']}" in a.exclude_units:
                skipped.append((f"{pair.scene}/u{u['idx']}", "excluded by --exclude-units"))
                continue
            ns = int(u["gt_shadow"].sum())
            if ns < a.min_shadow_pts:
                skipped.append((f"{pair.scene}/u{u['idx']}", f"only {ns} shadowed GT points"))
                continue
            rig = u["rig"]
            variants = [("orig", 0, None)] + [("aug", i, None) for i in range(a.n_aug)]
            made = 0
            for kind, i, _ in variants:
                if kind == "orig":
                    cl, sc, gt, org, info = u["cluster"], u["scan"], u["gt"], np.zeros(3), {}
                else:
                    tr = None
                    for _try in range(6):
                        tr = pick_transform(u["cluster"], rig, rng, a)
                        if tr is not None:
                            break
                    if tr is None:
                        continue
                    f, phi, shift = tr
                    sc, gt, org = f(u["scan"]), f(u["gt"]), f(np.zeros((1, 3)))[0]
                    # rigidity check: every range seen from the (moved) origin must be unchanged
                    rigid_err = max(rigid_err, float(np.abs(np.linalg.norm(sc - org, axis=1)
                                                            - np.linalg.norm(u["scan"], axis=1)).max()))
                    cl, ainfo = augment_cloud(f(u["cluster"]), rig, rng, cfg, cfg["extra_noise_scale"])
                    if len(cl) < 50:
                        continue
                    info = {"rotation_deg": round(float(np.rad2deg(phi)), 2),
                            "shift_x_cm": round(shift[0] * 100, 2), "shift_y_cm": round(shift[1] * 100, 2), **ainfo}
                sid = f"{pair.slug}_u{u['idx']}_{kind}{i:03d}"
                path = out / split / f"{sid}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, cluster=cl.astype(np.float32), scan=sc.astype(np.float32),
                                    gt=gt.astype(np.float32), gt_shadow=u["gt_shadow"], origin=np.asarray(org, np.float64),
                                    cam_K=rig.K, cam_R=rig.R, cam_t=rig.t, hw=np.array(rig.hw), floor_z=a.floor_z)
                rows.append({"sample_id": sid, "split": split, "scene": pair.scene, "category": pair.category,
                             "pair": pair.slug, "unit": u["idx"], "kind": kind, "n_cluster": len(cl),
                             "n_scan_rays": len(sc), "n_gt": len(gt), "n_shadow": ns, "path": str(path), **info})
                made += 1
            if made < 1 + a.n_aug:
                print(f"\n  [warn] {pair.scene}/u{u['idx']}: {made - 1}/{a.n_aug} augmented copies (cluster near the image border?)")
    print()

    step("4/5 writing index")
    if not rows:
        if first_tb:
            print("\nfirst error (full traceback):\n" + first_tb)
        raise SystemExit("no samples written - see the warnings above")
    write_csv(out / "index.csv", rows)
    write_csv(out / "pair_diagnostics.csv", diags)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2, default=str))

    step("5/5 summary")
    by = defaultdict(Counter)
    for r in rows:
        by[(r["split"])][r["kind"]] += 1
    print(f"     {'split':>6} {'orig':>6} {'aug':>6}")
    for sp in ("train", "val"):
        print(f"     {sp:>6} {by[sp]['orig']:>6} {by[sp]['aug']:>6}")
    print(f"     validation scenes: {sorted(val_scenes)}")
    orig = [r for r in rows if r["kind"] == "orig"]
    print(f"     target boxes: {len(orig)} | shadowed GT points per box: median {np.median([r['n_shadow'] for r in orig]):.0f}, "
          f"min {min(r['n_shadow'] for r in orig)} | rays per box: median {np.median([r['n_scan_rays'] for r in orig]):.0f}")
    if skipped:
        print(f"     skipped ({len(skipped)}):")
        for s, why in skipped[:15]:
            print(f"        {s}: {why}")

    n_units = len(orig) + len([s for s in skipped if "shadowed" in s[1]])
    jd = np.array([d["json_vs_pcd_med_mm"] for d in diags])
    cd = np.array([d["calib_dpos_cm"] for d in diags])
    gates.check("samples written", len(rows) > 0, f"{len(rows)} samples from {len(orig)} boxes")
    gates.check("json LiDAR points == even .pcd (same frame)", np.median(jd) < 50,
                f"median NN distance {np.median(jd):.2f} mm (max {jd.max():.1f} mm); >50 mm means the json and the pcd are in different frames")
    gates.check("rigid transform preserves LiDAR ranges", rigid_err < 1e-4, f"max |range change| = {rigid_err:.2e} m")
    gates.check("boxes with enough shadow", len(orig) >= 0.7 * max(1, n_units),
                f"{len(orig)}/{n_units} boxes have >= {a.min_shadow_pts} shadowed GT points "
                f"(if low: are odd/even swapped, or is --shadow-ang-deg too small?)", hard=False)
    gates.check("odd/even calibration agree", np.median(cd) < 3.0,
                f"median LiDAR-position difference {np.median(cd):.2f} cm (the odd calibration is used for both)", hard=False)
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"     stage 0 took {time.time() - t0:.0f}s")
    gates.finish()


if __name__ == "__main__":
    main()