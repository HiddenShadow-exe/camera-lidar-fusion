#!/usr/bin/env python3
"""Turn SINGLE recordings (one box, nothing occluding it) into occluded/GT pairs by adding a VIRTUAL occluder.

For every source recording and every variant:
  * a box-shaped occluder is placed between the LiDAR and the target (random distance, width, height, yaw, lateral offset
    -> anything from no overlap to a full occlusion),
  * every ray of the recorded scan that hits the occluder before its measured return is replaced by a return on the
    occluder surface -> the target points on those rays vanish from the "occluded" scan (= the shadow),
  * the unmodified recording is the GT ("even") recording.
The result is written in your dataset layout, as an extra category folder, so the whole pipeline runs on it unchanged:

    <out>/recording_01   odd  : modified scan (occluder added)      <- pointcloud_lidar.pcd (+ json copy)
    <out>/recording_02   even : the original recording (GT)
    <out>/recording_03 ...

Camera cluster, LiDAR object points and calibration come from the real recording, so the pipeline learns the real
camera-box-vs-LiDAR-surface error. NOT real: the occluder (a clean box, no sensor effects) and the odd scan having exactly
the rays of the GT scan. Therefore the category name must start with 'virtual': such scenes are never put in the validation
split and are never used for the final numbers (stage 0 / stage 6 --cv / stage 7 handle that).

    python make_virtual_pairs.py --input "dataset/_cardboard box" --out "dataset/virtual boxes" --k 3
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

from common import R_C2L, ray_box_interval, read_pcd_xyz, rot_z, write_pcd_ascii  # noqa: F401  (R_C2L: keep import parity)


def find_recordings(root):
    out = []
    for p in sorted(Path(root).rglob("recording_*")):
        if p.is_dir() and re.fullmatch(r"recording_(\d+)", p.name) and (p / "recording_data.json").exists() and (p / "pointcloud_lidar.pcd").exists():
            out.append(p)
    return sorted(out, key=lambda p: (str(p.parent), int(p.name.split("_")[1])))


def make_variant(scan, gt, rng, a):
    """-> (new_scan, occluder (c, dims, R), n_shadow_pts) or None."""
    cen = np.median(gt, axis=0)
    rt = float(np.percentile(np.linalg.norm(gt[:, :2] - cen[:2], axis=1), 95))
    dist = float(np.linalg.norm(cen[:2]))
    u = cen[:2] / dist
    v = np.array([-u[1], u[0]])
    rs = np.linalg.norm(scan, axis=1)
    dirs = scan / rs[:, None]
    for _ in range(a.tries):
        t, w, h = rng.uniform(*a.thick), rng.uniform(*a.width), rng.uniform(*a.height)
        d_oc = rng.uniform(a.front[0], a.front[1]) * (dist - rt)                 # distance of the occluder centre along the LiDAR -> target line
        if d_oc - t / 2 < 0.25:
            continue
        s = rng.uniform(-1, 1) * a.lateral * (w / 2 + rt)
        yaw = np.arctan2(u[1], u[0]) + rng.uniform(-a.yaw, a.yaw)
        c = np.array([*(u * d_oc + v * s), a.floor + h / 2])
        R = rot_z(yaw)
        dims = np.array([t, w, h])
        # the occluder must not touch the target
        if np.any(np.all(np.abs((gt - c) @ R) <= dims / 2 + 0.03, axis=1)):
            continue
        tin, tout = ray_box_interval((np.zeros(3) - c) @ R, dirs @ R, -dims / 2, dims / 2)
        hit = (tout >= tin) & (tin > 0) & (tin < rs - 0.02)
        # GT points whose ray is blocked
        rg = np.linalg.norm(gt, axis=1)
        tg, og = ray_box_interval((np.zeros(3) - c) @ R, (gt / rg[:, None]) @ R, -dims / 2, dims / 2)
        n_sh = int(((og >= tg) & (tg > 0) & (tg < rg - 0.05)).sum())
        if n_sh < a.min_shadow or hit.sum() < 200:
            continue
        new = scan.copy()
        tt = tin[hit] + rng.normal(0, a.noise, hit.sum())
        new[hit] = dirs[hit] * tt[:, None]
        return new, (c, dims, R), n_sh
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder with the single recordings (recording_XX/{recording_data.json, pointcloud_lidar.pcd}), searched recursively")
    ap.add_argument("--out", required=True, help="output category folder, e.g. 'dataset/virtual boxes' (name must start with 'virtual')")
    ap.add_argument("--k", type=int, default=3, help="virtual occluder variants per recording")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-range", type=float, default=5.0)
    ap.add_argument("--floor", type=float, default=-0.40, help="floor height in the LiDAR frame")
    ap.add_argument("--thick", type=float, nargs=2, default=(0.12, 0.30), help="occluder depth range [m]")
    ap.add_argument("--width", type=float, nargs=2, default=(0.35, 0.80), help="occluder width range [m]")
    ap.add_argument("--height", type=float, nargs=2, default=(0.25, 0.60), help="occluder height range [m]")
    ap.add_argument("--front", type=float, nargs=2, default=(0.30, 0.75), help="occluder centre at this fraction of the free distance in front of the target")
    ap.add_argument("--lateral", type=float, default=1.0, help="lateral offset in units of (occluder half width + target radius); 0 = always centred (full occlusion)")
    ap.add_argument("--yaw", type=float, default=0.35, help="occluder yaw jitter around 'facing the LiDAR' [rad]")
    ap.add_argument("--noise", type=float, default=0.004, help="range noise on the occluder returns [m]")
    ap.add_argument("--min-shadow", type=int, default=30)
    ap.add_argument("--tries", type=int, default=200)
    ap.add_argument("--only", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    if not out.name.lower().startswith("virtual"):
        raise SystemExit("the output folder name must start with 'virtual' (e.g. 'virtual boxes'): those scenes are excluded from validation / final numbers")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty - remove it first (the numbering must start at recording_01)")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    recs = find_recordings(a.input)
    if a.only:
        recs = recs[:a.only]
    print(f"{len(recs)} source recordings under {a.input}")
    n_out, report, skipped = 0, [], []
    for rec in recs:
        try:
            doc = json.loads((rec / "recording_data.json").read_text())
            gt = np.asarray(doc["3d_data_lidar"]["object_points"], float)[:, :3]
            for k in ("3d_data_camera", "calibration", "2d_data"):
                assert k in doc, f"no '{k}' in the json"
            assert len(doc["3d_data_camera"]["object_points"]) > 100, "empty camera cluster"
            scan = read_pcd_xyz(rec / "pointcloud_lidar.pcd")
        except Exception as e:
            skipped.append((rec, repr(e)))
            continue
        scan = scan[np.linalg.norm(scan, axis=1) < a.max_range]
        gt = gt[np.linalg.norm(gt, axis=1) < a.max_range]
        if len(gt) < 200:
            skipped.append((rec, f"only {len(gt)} LiDAR object points"))
            continue
        made = 0
        for _ in range(a.k):
            res = make_variant(scan, gt, rng, a)
            if res is None:
                continue
            new, (c, dims, R), n_sh = res
            n_out += 1
            made += 1
            odd, even = out / f"recording_{2 * n_out - 1:02d}", out / f"recording_{2 * n_out:02d}"
            odd.mkdir(), even.mkdir()
            shutil.copy(rec / "recording_data.json", even / "recording_data.json")
            shutil.copy(rec / "pointcloud_lidar.pcd", even / "pointcloud_lidar.pcd")
            doc_o = dict(doc)
            doc_o["virtual_occluder"] = {"source": str(rec), "center": c.tolist(), "dims": dims.tolist(), "yaw_rad": float(np.arctan2(R[1, 0], R[0, 0]))}
            (odd / "recording_data.json").write_text(json.dumps(doc_o))
            write_pcd_ascii(odd / "pointcloud_lidar.pcd", new)
            report.append(n_sh / len(gt))
        if made < a.k:
            skipped.append((rec, f"only {made}/{a.k} variants found (target too close to the LiDAR or too small a gap?)"))
    report = np.array(report) if report else np.zeros(1)
    print(f"written {n_out} virtual pairs to {out.resolve()}  ({n_out * 2} recording folders)")
    print(f"share of the GT hidden by the virtual occluder: min {report.min():.0%} | median {np.median(report):.0%} | max {report.max():.0%}"
          f" | fully hidden (>=98%): {np.mean(report >= 0.98):.0%}")
    for r, why in skipped[:20]:
        print(f"  [skipped] {r}: {why}")
    if skipped[20:]:
        print(f"  ... and {len(skipped) - 20} more")


if __name__ == "__main__":
    main()