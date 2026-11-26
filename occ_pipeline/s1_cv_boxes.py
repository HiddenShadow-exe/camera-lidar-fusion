#!/usr/bin/env python3
"""STAGE 1 - the initial ("traditional CV") box of every sample, same logic as get_boxes_3d() of the live script but
for ONE cluster and in the LiDAR frame:

    top plane = cluster points within --top-slab of the highest point (robust: --top-pct percentile instead of the max)
    -> project to the image -> minimum-area rectangle -> back-project the 4 corners at the top-plane depth
    -> box = rectangle x [floor_z, top_z]

Sanity gate (this is where a camera<->LiDAR frame mix-up shows up): the GT LiDAR points of the target lie on the true
box surface, so their distance to the CV box surface must be a few cm. If it is tens of cm the two sensors are not in
the same frame.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from common import (Gates, Rig, banner, box_sdf, min_area_rect, progress, read_csv, rot_z, step, write_csv)


def cv_box(cluster, rig, floor_z, ground_margin, top_slab, top_pct, min_pts=100):
    pts = cluster[cluster[:, 2] > floor_z + ground_margin]
    if len(pts) < min_pts:
        return None
    top_z = float(np.percentile(pts[:, 2], top_pct))
    top = pts[pts[:, 2] >= top_z - top_slab]
    if len(top) < 20:
        return None
    u, v, _ = rig.project(top)
    corners_px = min_area_rect(np.stack([u, v], 1))
    depth = rig.t[2] - top_z                                    # camera-frame depth of the (horizontal) top plane
    q = rig.unproject(corners_px[:, 0], corners_px[:, 1], np.full(4, depth))
    e1, e2 = q[1] - q[0], q[2] - q[1]
    dims_xy = np.array([np.linalg.norm(e1), np.linalg.norm(e2)])
    yaw = np.arctan2(e1[1], e1[0])
    if yaw >= np.pi / 4 or yaw < -np.pi / 4:                    # canonical yaw in [-45, 45) deg, swap the sides
        k = np.round(yaw / (np.pi / 2))
        yaw -= k * np.pi / 2
        if int(k) % 2:
            dims_xy = dims_xy[::-1]
    cxy = q[:, :2].mean(0)
    h = top_z - floor_z
    if h <= 0.02 or dims_xy.min() < 0.03:
        return None
    return np.array([cxy[0], cxy[1], (top_z + floor_z) / 2]), np.array([dims_xy[0], dims_xy[1], h]), rot_z(yaw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--ground-margin", type=float, default=0.08)
    ap.add_argument("--top-slab", type=float, default=0.10)
    ap.add_argument("--top-pct", type=float, default=99.0)
    ap.add_argument("--gate-sdf-cm", type=float, default=10.0, help="hard gate: median GT-to-CV-box surface distance [cm]")
    ap.add_argument("--min-support", type=int, default=100,
                    help="a unit (camera cluster) is evaluable only if >= this many GT LiDAR points lie within --support-dist of its CV box")
    ap.add_argument("--support-dist", type=float, default=0.10)
    ap.add_argument("--max-dim-dev", type=float, default=0.10,
                    help="drop an augmented sample whose CV box side lengths deviate more than this [m] from the real sample of the same unit")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = Path(a.work) / "stage1"
    banner(1, "initial CV box for every sample (fitted on the CAMERA cluster only)")
    if (out / "DONE").exists() and not a.force:
        print(f"  {out}/DONE exists -> skipping (use --force to redo)")
        return
    out.mkdir(parents=True, exist_ok=True)
    gates = Gates(1)
    rows = read_csv(Path(a.work) / "stage0" / "index.csv")
    step(f"1/4 fitting boxes for {len(rows)} samples (camera cluster only; LiDAR is only read for the support / frame diagnostics)")
    res, t0 = {}, time.time()
    for r in progress(rows, total=len(rows), desc="boxes"):
        z = np.load(r["path"])
        rig = Rig(z["cam_K"], z["cam_R"], z["cam_t"], z["hw"])
        box = cv_box(z["cluster"].astype(np.float64), rig, float(z["floor_z"]), a.ground_margin, a.top_slab, a.top_pct)
        d = {"sample_id": r["sample_id"], "unit": f"{r['pair']}/u{r['unit']}", "kind": r["kind"], "ok": 0 if box is None else 1}
        if box is not None:
            c, dm, Rm = box
            sdf = np.abs(box_sdf(z["gt"].astype(np.float64), c, dm, Rm))
            near = sdf < 0.2
            d.update(gt_sdf_med_cm=float(np.median(sdf[near]) * 100) if near.any() else np.nan, gt_near_frac=float(near.mean()),
                     gt_support=int((sdf < a.support_dist).sum()), dim_x=dm[0], dim_y=dm[1], dim_z=dm[2])
            res[r["sample_id"]] = box
        res.setdefault("_diag", []).append(d)
    diag = res.pop("_diag")
    by_id = {d["sample_id"]: d for d in diag}

    step("2/4 deciding which units / samples can be used")
    orig = {d["unit"]: d for d in diag if d["kind"] == "orig"}
    unit_reason = {}
    for u, d in orig.items():
        if not d["ok"]:
            unit_reason[u] = "CV box fit failed on the real cluster"
        elif d["gt_support"] < a.min_support:
            unit_reason[u] = (f"no LiDAR counterpart: only {d['gt_support']} GT points within {a.support_dist * 100:.0f} cm of the CV box "
                              f"(< {a.min_support}); {d['gt_near_frac']:.0%} of the GT points in its crop are within 20 cm")
    excluded, keep = [], []
    for d in diag:
        reason = unit_reason.get(d["unit"])
        if reason is None and not d["ok"]:
            reason = "CV box fit failed"
        if reason is None and d["kind"] != "orig":
            o = orig[d["unit"]]
            dev = max(abs(np.sort([d["dim_x"], d["dim_y"]]) - np.sort([o["dim_x"], o["dim_y"]])))
            if dev > a.max_dim_dev:
                reason = f"CV box side lengths deviate {dev * 100:.0f} cm from the real sample (> {a.max_dim_dev * 100:.0f} cm): fit artefact of the augmentation"
        d["excluded"] = reason or ""
        (excluded if reason else keep).append(d)
    write_csv(out / "diagnostics.csv", diag)
    if excluded:
        write_csv(out / "excluded.csv", excluded)
    n_units_bad = len(unit_reason)
    print(f"     {len(orig) - n_units_bad}/{len(orig)} units usable; excluded {len(excluded)}/{len(diag)} samples:")
    for u, why in unit_reason.items():
        print(f"       unit {u}: {why}")
    n_aug_dev = sum(1 for d in excluded if d["unit"] not in unit_reason and d["kind"] != "orig")
    if n_aug_dev:
        print(f"       + {n_aug_dev} augmented samples of otherwise usable units dropped (box side lengths deviate too much)")

    step("3/4 writing boxes.npz")
    ids = [d["sample_id"] for d in keep]
    np.savez(out / "boxes.npz", ids=np.array(ids), center=np.array([res[i][0] for i in ids]),
             dims=np.array([res[i][1] for i in ids]), R=np.array([res[i][2] for i in ids]))

    step("4/4 summary + gates")
    real = [d for d in keep if d["kind"] == "orig"]
    dims = np.array([[d["dim_x"], d["dim_y"], d["dim_z"]] for d in keep])
    sdf_o = np.array([d["gt_sdf_med_cm"] for d in real], float)
    print(f"     kept {len(keep)} samples from {len(real)} units; box sizes (m): x {np.median(dims[:, 0]):.2f}, y {np.median(dims[:, 1]):.2f}, "
          f"z {np.median(dims[:, 2]):.2f} (medians)")
    print(f"     GT LiDAR -> CV box surface, real samples: median {np.nanmedian(sdf_o):.1f} cm | p90 {np.nanpercentile(sdf_o, 90):.1f} cm | "
          f"max {np.nanmax(sdf_o):.1f} cm")
    gates.check("usable units", len(real) >= 3, f"{len(real)}/{len(orig)} units have a CV box and LiDAR support")
    gates.check("LiDAR and camera agree (frames)", np.nanmedian(sdf_o) <= a.gate_sdf_cm,
                f"median GT-to-box distance {np.nanmedian(sdf_o):.1f} cm (limit {a.gate_sdf_cm:.0f} cm); "
                f"tens of cm = camera/LiDAR frames or calibration are inconsistent")
    gates.check("box dimensions plausible", np.all((dims[:, :2].min(1) > 0.08) & (dims.max(1) < 1.5)),
                f"min xy side {dims[:, :2].min():.2f} m, max {dims.max():.2f} m", hard=False)
    gates.check("CV box close to the LiDAR surface", np.nanpercentile(sdf_o, 90) < 8,
                f"p90 {np.nanpercentile(sdf_o, 90):.1f} cm: errors beyond the shift-grid reach (+-n_steps*step) cannot be corrected", hard=False)
    gates.check("few samples excluded", len(excluded) <= 0.25 * len(diag), f"{len(excluded)}/{len(diag)} excluded (see {out}/excluded.csv)", hard=False)
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"     stage 1 took {time.time() - t0:.0f}s")
    gates.finish()


if __name__ == "__main__":
    main()