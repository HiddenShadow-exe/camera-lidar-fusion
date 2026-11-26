#!/usr/bin/env python3
"""DIAGNOSTIC (not a pipeline stage) - how much would a yaw correction of the CV box add on top of the face shifts?

The face shifts only move faces parallel to themselves, so a rotated CV box (yaw error) cannot be repaired by them.
For every real sample this rotates the CV box about its vertical axis through its centre by each angle in --yaws,
evaluates the COMPLETE shift table for each (same evaluator, same grid as stage 4) and prints the oracle score per angle.

  gain = best oracle over the angles - oracle at 0 deg      (how much a perfect yaw would add on top of the shifts)

Note: with --voxel-frame box the voxel grid turns with the box, so every angle also changes the voxel alignment a little
(a few 0.01 in score); only gains clearly above that are meaningful.

    python s4d_yaw_probe.py --work work --voxel-size 0.02 --step 0.02 --n-steps 4 --n-steps-z 2 --yaws -12,-8,-4,0,4,8,12
Runtime ~ (samples x angles x one stage-4 sample), e.g. 16 x 7 x 6 s = 11 min on your GPU.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from common import banner, read_csv, rot_z, step
from shiftgrid import Backend, add_grid_args, best_candidate, evaluate_candidates, grid_from_args


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--kind", default="orig", choices=["orig", "aug", "all"])
    ap.add_argument("--yaws", default="-12,-8,-4,0,4,8,12", help="angles in degrees (must contain 0)")
    ap.add_argument("--n-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--mem-mb", type=float, default=1500)
    ap.add_argument("--backend", default="auto", choices=["auto", "numpy", "torch"])
    add_grid_args(ap)
    a = ap.parse_args()
    banner("4d", "yaw probe")
    bk, grid, vs = Backend(a.backend), grid_from_args(a), a.voxel_size
    yaws = [float(x) for x in a.yaws.split(",")]
    assert 0.0 in yaws, "--yaws must contain 0"
    boxes = np.load(Path(a.work) / "stage1" / "boxes.npz")
    bid = {s: i for i, s in enumerate(boxes["ids"])}
    rows = [r for r in read_csv(Path(a.work) / "stage0" / "index.csv") if r["sample_id"] in bid and (a.kind == "all" or r["kind"] == a.kind)]
    if a.n_samples:
        rows = rows[:a.n_samples]
    step(f"backend {bk.name} | {len(rows)} samples x {len(yaws)} angles | {grid.describe()} | voxel {vs * 100:.1f} cm ({a.voxel_frame})")
    l1 = grid.l1_all()
    print(f"\n     {'sample':<40} " + " ".join(f"{y:>6.0f}" for y in yaws) + "   best yaw    gain   (oracle tol score per yaw angle [deg])")
    gains, best_yaws, t0 = [], [], time.time()
    for r in rows:
        i = bid[r["sample_id"]]
        c0, d0, R0 = boxes["center"][i], boxes["dims"][i], boxes["R"][i]
        z = np.load(r["path"])
        origin, scan, gt, sh = z["origin"], z["scan"].astype(np.float64), z["gt"].astype(np.float64), z["gt_shadow"]
        vals = []
        for y in yaws:
            res = evaluate_candidates(bk, origin, scan, (c0, d0, rot_z(np.deg2rad(y)) @ R0), gt, sh, grid, vs, a.block_margin, a.tol_vox, a.mem_mb, frame=a.voxel_frame)
            _, m, _ = best_candidate(np.nan_to_num(res["score_tol"], nan=-1), l1)
            vals.append(float(m))
        v = np.array(vals)
        g = float(v.max() - v[yaws.index(0.0)])
        gains.append(g)
        best_yaws.append(yaws[int(np.argmax(v - 1e-6 * np.abs(yaws)))])      # ties -> smallest |yaw|
        print(f"     {r['sample_id']:<40} " + " ".join(f"{x:>6.2f}" for x in v) + f"   {best_yaws[-1]:>+8.0f} {g:>7.3f}", flush=True)
    gains, by = np.array(gains), np.array(best_yaws)
    print(f"\n     gain from a perfect yaw: median {np.median(gains):.3f}, mean {gains.mean():.3f}, p90 {np.percentile(gains, 90):.3f}, "
          f"{np.mean(gains > 0.05):.0%} of samples gain > 0.05;  |best yaw| >= 4 deg in {np.mean(np.abs(by) >= 4):.0%}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()