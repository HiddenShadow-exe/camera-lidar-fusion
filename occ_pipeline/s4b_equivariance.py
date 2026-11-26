#!/usr/bin/env python3
"""DIAGNOSTIC (not a pipeline stage) - does the achievable score depend on how the scene is rotated?

Takes ONE real sample (CV box, LiDAR scan, GT, LiDAR origin - no camera artifacts involved) and evaluates the complete
shift table after rigidly rotating everything by phi, in two ways:

  lidar-axis   rotation about the vertical axis through the LiDAR. Nothing physical changes (the LiDAR stays at the origin,
               ranges are identical); only the alignment of the scene with the WORLD-aligned voxel grid changes.
  camera-axis  the transform of stage 0 (vertical axis through the camera centre): additionally the LiDAR origin moves.

Both must give (almost) the same oracle score. If they do not:
  * lidar-axis already varies  -> the score depends on the voxel-grid alignment (use --voxel-frame box)
  * only camera-axis varies    -> the moved origin is handled differently from the origin at (0,0,0): that would be a bug
It also prints how low the GT points reach above the floor assumed for the CV box (a ground-filtered GT shows up here).

    python s4b_equivariance.py --work work --sample full_occlusion_boxes_01_u0_orig000 --voxel-size 0.02 --step 0.03 --n-steps 3
    python s4b_equivariance.py ... --voxel-frame world      # the old behaviour, for comparison
"""
import argparse
import time
from pathlib import Path

import numpy as np

from common import Gates, banner, progress, read_csv, rot_z, step
from shiftgrid import Backend, add_grid_args, best_candidate, evaluate_candidates, grid_from_args


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--sample", default=None, help="sample_id (default: first real sample in stage 1)")
    ap.add_argument("--n-rot", type=int, default=8, help="rotations per mode (0 .. 360 deg)")
    ap.add_argument("--mem-mb", type=float, default=1500)
    ap.add_argument("--backend", default="auto", choices=["auto", "numpy", "torch"])
    add_grid_args(ap)
    a = ap.parse_args()

    banner("4b", "rotation-equivariance diagnostic")
    bk, grid, vs = Backend(a.backend), grid_from_args(a), a.voxel_size
    boxes = np.load(Path(a.work) / "stage1" / "boxes.npz")
    ids = list(boxes["ids"])
    idx = {r["sample_id"]: r for r in read_csv(Path(a.work) / "stage0" / "index.csv")}
    sid = a.sample or next(s for s in ids if idx[s]["kind"] == "orig")
    i = ids.index(sid)
    c0, d0, R0 = boxes["center"][i], boxes["dims"][i], boxes["R"][i]
    z = np.load(idx[sid]["path"])
    origin0, scan0, gt0, sh = z["origin"], z["scan"].astype(np.float64), z["gt"].astype(np.float64), z["gt_shadow"]
    cam = -z["cam_R"] @ z["cam_t"]
    floor = float(z["floor_z"])
    step(f"backend {bk.name} | sample {sid} | {grid.describe()} | voxel {vs * 100:.1f} cm ({a.voxel_frame}-aligned), tolerance {a.tol_vox} voxel")
    print(f"     CV box: dims {np.round(d0, 3)} m, centre {np.round(c0, 3)}, bottom face at z = {c0[2] - d0[2] / 2:.3f} (floor assumed {floor:.2f})")
    near = np.abs(((gt0 - c0) @ R0)).__le__(d0 / 2 + 0.10).all(1)
    h = (gt0[near, 2] - floor) * 100
    print(f"     GT points near the box: {int(near.sum())}; height above the assumed floor [cm]: min {h.min():.1f} | p1 {np.percentile(h, 1):.1f} | "
          f"p5 {np.percentile(h, 5):.1f} | median {np.median(h):.1f}   (shadowed ones: min {(gt0[near & sh, 2] - floor).min() * 100:.1f})")

    l1, zero = grid.l1_all(), grid.zero_index()
    spread = {}
    for mode, pivot in (("lidar-axis", np.zeros(3)), ("camera-axis", np.array([cam[0], cam[1], 0.0]))):
        print(f"\n  rotation about the {mode} vertical axis" + ("  (only the voxel alignment changes)" if mode == "lidar-axis" else "  (stage 0's transform: LiDAR origin moves too)"))
        print(f"     {'phi':>6} {'oracle tol':>11} {'CV box tol':>11} {'oracle exact':>13}   best shift [cm]  lo_x hi_x lo_y hi_y lo_z hi_z")
        vals = []
        for phi in progress(np.linspace(0, 360, a.n_rot, endpoint=False), total=a.n_rot, desc=mode):
            Rz = rot_z(np.deg2rad(phi))
            f = lambda X: (np.atleast_2d(X) - pivot) @ Rz.T + pivot
            res = evaluate_candidates(bk, f(origin0)[0], f(scan0), (f(c0)[0], d0, Rz @ R0), f(gt0), sh, grid, vs, a.block_margin, a.tol_vox, a.mem_mb, frame=a.voxel_frame)
            bt, mt, _ = best_candidate(np.nan_to_num(res["score_tol"], nan=-1), l1)
            be, me, _ = best_candidate(np.nan_to_num(res["score_exact"], nan=-1), l1)
            vals.append((phi, mt, float(res["score_tol"][zero]), me, grid.decode(bt) * 100))
        for phi, mt, zt, me, shf in vals:
            print(f"     {phi:6.0f} {mt:11.3f} {zt:11.3f} {me:13.3f}   {np.round(shf, 1)}")
        o = np.array([v[1] for v in vals])
        spread[mode] = float(o.max() - o.min())
        print(f"     oracle(tol): mean {o.mean():.3f}, min {o.min():.3f}, max {o.max():.3f}  -> spread {spread[mode]:.3f}")

    print()
    g = Gates("4b")
    g.check("score independent of the voxel-grid alignment (lidar-axis)", spread["lidar-axis"] <= 0.10, f"spread {spread['lidar-axis']:.3f} (limit 0.10)", hard=False)
    g.check("score independent of the LiDAR origin position (camera-axis)", spread["camera-axis"] <= 0.10, f"spread {spread['camera-axis']:.3f} (limit 0.10)", hard=False)


if __name__ == "__main__":
    main()
