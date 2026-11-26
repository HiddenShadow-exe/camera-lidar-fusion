#!/usr/bin/env python3
"""Tests of the graded score (shiftgrid.evaluate_candidates(..., graded=spec)):
   1. spec = one cube shell with weight 1  -> identical to the tolerance score (tol 1)
   2. sigma -> 0                            -> identical to the exact score
   3. brute force (KD-tree nearest cell, Gaussian weight) on random candidates   -> identical to the batched result
   4. (informational) exact <= graded <= tol; not guaranteed in general, graded reaches 2 voxels, tol only 1
Run on a stage-0/1 work folder:  python test_graded.py --work work"""
import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from common import read_csv, region_mask
from shiftgrid import Backend, ShiftGrid, envelope_half, evaluate_candidates, graded_spec

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="work")
ap.add_argument("--n-samples", type=int, default=2)
ap.add_argument("--steps", type=int, default=1)
a = ap.parse_args()

bk = Backend("numpy")
grid = ShiftGrid(a.steps, a.steps, 0.03, 0.05)
vs = 0.03
rows = read_csv(Path(a.work) / "stage0" / "index.csv")
boxes = np.load(Path(a.work) / "stage1" / "boxes.npz")
bid = {s: i for i, s in enumerate(boxes["ids"])}
rows = [r for r in rows if r["sample_id"] in bid][:a.n_samples]
ok = True
for r in rows:
    i = bid[r["sample_id"]]
    box = (boxes["center"][i], boxes["dims"][i], boxes["R"][i])
    z = np.load(r["path"])
    origin, scan, gt, sh = z["origin"], z["scan"].astype(np.float64), z["gt"].astype(np.float64), z["gt_shadow"]

    def run(spec):
        return evaluate_candidates(bk, origin, scan, box, gt, sh, grid, vs, None, 1, 500, frame="box", graded=spec)

    cube = [np.array([(0, 0, 0)])] + [np.array([o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)])]
    spec_cube = dict(a=np.array([0.0, 1.0]), shells=cube)
    res = run(spec_cube)
    d1 = np.nanmax(np.abs(res["score_graded"] - res["score_tol"]))
    spec0 = graded_spec(1e-3, 2)
    res0 = run(spec0)
    d2 = np.nanmax(np.abs(res0["score_graded"] - res0["score_exact"]))
    spec = graded_spec(1.0, 2)
    res = run(spec)
    sg, st, se = res["score_graded"], res["score_tol"], res["score_exact"]
    v = ~np.isnan(sg)
    ordered = bool(np.all(sg[v] <= st[v] + 1e-5) and np.all(sg[v] >= se[v] - 1e-5))

    # brute force on random valid candidates
    c0, d0, R0 = box
    H = envelope_half(d0, grid, vs)
    to_f = lambda p: (p - c0) @ R0
    cl = lambda p: np.unique(np.floor(to_f(p[region_mask(p, box, H)]) / vs).astype(np.int64), axis=0)
    G, S = cl(gt), cl(gt[sh])
    kG = cKDTree(G)
    from visualize_scene import regen_points
    rng = np.random.default_rng(0)
    idx = rng.choice(np.flatnonzero(v), 12, replace=False)
    d3 = 0.0
    for f in idx:
        P = np.unique(np.floor(to_f(regen_points(origin, scan, grid.box_from(box, f), vs)) / vs).astype(np.int64), axis=0)
        w = lambda d2: np.where(d2 <= 4, np.exp(-d2 / 2.0), 0.0)
        if len(P) == 0:
            continue
        dS = cKDTree(P).query(S)[0] ** 2
        dP = kG.query(P)[0] ** 2
        own = w(np.round(dS)).sum() / (len(S) + (1 - w(np.round(dP))).sum())
        d3 = max(d3, abs(own - sg[f]))
    good = d1 < 1e-5 and d2 < 1e-5 and d3 < 1e-4
    ok &= good
    print(f"{r['sample_id']}: cube==tol {d1:.1e} | sigma->0 == exact {d2:.1e} | brute force {d3:.1e} | exact <= graded <= tol (info): {ordered} | "
          f"S {len(S)} cells, {int(v.sum())} candidates -> {'OK' if good else 'FAIL'}")
print("ALL OK" if ok else "FAILED")
raise SystemExit(0 if ok else 1)