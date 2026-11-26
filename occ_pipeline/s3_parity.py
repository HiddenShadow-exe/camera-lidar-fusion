#!/usr/bin/env python3
"""STAGE 3 - parity test: the fast slab/voxel-grid evaluator must reproduce a slow, obviously-correct reference.

Reference = your Moller-Trumbore ray/triangle caster (12 outward-oriented triangles per box, back-face culling, no
noise) + python sets of voxel keys (the same logic as the 'e' branch of the live script). It is run on the same
rays / GT for a few hundred candidates and compared with the batched evaluator (numpy or torch/CUDA).
"""
import argparse
import time
from pathlib import Path

import numpy as np

from common import Gates, banner, obb_corners, progress, read_csv, ray_box_interval, region_mask, step
from shiftgrid import Backend, add_grid_args, best_candidate, envelope_half, evaluate_candidates, grid_from_args


def box_triangles(box):
    c, d, R = box
    cor = obb_corners(c, d, R)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    tris = []
    for k in range(3):
        a, b = [j for j in range(3) if j != k]
        for s in (-1, 1):
            idx = np.flatnonzero(signs[:, k] == s)
            F = cor[idx]
            loc = F - F.mean(0)
            order = np.argsort(np.arctan2(loc @ R[:, b], loc @ R[:, a]))
            F = F[order]
            for tri in ([F[0], F[1], F[2]], [F[0], F[2], F[3]]):
                t = np.array(tri)
                n = np.cross(t[1] - t[0], t[2] - t[0])
                if n @ (t.mean(0) - c) < 0:                      # enforce outward normal (as in the live script)
                    t = t[[0, 2, 1]]
                tris.append(t)
    return np.array(tris)


def moller_trumbore_first_hit(o, D, tris, eps=1e-9):
    """Closest FRONT-facing triangle hit per ray (back-face culling like ray_intersect_triangles_batch). -> t (N,), inf = miss"""
    best = np.full(len(D), np.inf)
    for v0, v1, v2 in tris:
        e1, e2 = v1 - v0, v2 - v0
        h = np.cross(D, e2)
        a = h @ e1
        front = a > eps                                          # A = -dot(dir, n): positive when the ray goes against the outward normal
        f = np.where(front, 1.0 / np.where(front, a, 1.0), 0.0)
        s = o - v0
        u = f * (h @ s)
        q = np.cross(s, e1)
        v = f * (D @ q)
        t = f * float(q @ e2)
        ok = front & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > eps)
        best = np.where(ok & (t < best), t, best)
    return best


def reference_scores(origin, scan, cand, cv, gt, gt_shadow, vs, m, tol, H, frame):
    """cand = candidate box (what the rays are cast at); cv = the CV box (defines the GT region and, for frame=box, the voxel grid)."""
    vec = scan - origin
    r = np.linalg.norm(vec, axis=1)
    ok = r > 1e-6
    vec, r = vec[ok], r[ok]
    D = vec / r[:, None]
    t = moller_trumbore_first_hit(origin, D, box_triangles(cand))
    blocked = np.isfinite(t) & (t > r + m)
    viol = int((np.isfinite(t) & (t < r - m)).sum())
    pts = origin + t[blocked][:, None] * D[blocked]
    tf = (lambda p: (p - cv[0]) @ cv[2]) if frame == "box" else (lambda p: p)
    keys = lambda p: set(map(tuple, np.floor(tf(p) / vs).astype(np.int64)))
    P = keys(pts)
    reg = region_mask(gt, cv, H)
    G, S = keys(gt[reg]), keys(gt[reg & gt_shadow])
    nS = len(S)
    se = len(P & S) / (nS + len(P - G))
    if tol > 0:
        offs = [(i, j, k) for i in range(-tol, tol + 1) for j in range(-tol, tol + 1) for k in range(-tol, tol + 1)]
        dil = lambda K: {(x + i, y + j, z + k) for (x, y, z) in K for (i, j, k) in offs}
        dP, dG = dil(P), dil(G)
        st = len(S & dP) / (nS + len(P - dG))
    else:
        st = se
    return se, st, len(P), viol


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--n-cands", type=int, default=40)
    ap.add_argument("--mem-mb", type=float, default=600)
    ap.add_argument("--backend", default="auto", choices=["auto", "numpy", "torch"])
    add_grid_args(ap)
    ap.set_defaults(n_steps=2)                                   # small grid: the fast evaluator runs on ALL its candidates
    a = ap.parse_args()
    banner(3, "evaluator parity test (fast slab evaluator vs Moller-Trumbore reference)")
    gates = Gates(3)
    bk, grid, vs = Backend(a.backend), grid_from_args(a), a.voxel_size
    m = vs if a.block_margin is None else a.block_margin
    step(f"backend: {bk.name} | grid: {grid.describe()} | voxels {a.voxel_frame}-aligned")
    rows = [r for r in read_csv(Path(a.work) / "stage0" / "index.csv") if r["kind"] == "orig"]
    boxes = np.load(Path(a.work) / "stage1" / "boxes.npz")
    bid = {s: i for i, s in enumerate(boxes["ids"])}
    rows = [r for r in rows if r["sample_id"] in bid]
    rng = np.random.default_rng(1)
    rows = [rows[i] for i in rng.permutation(len(rows))[:a.n_samples]]

    all_d, n_eq, n_tot, d_viol, d_pred, t_err = [], 0, 0, 0, 0, 0.0
    for r in progress(rows, total=len(rows), desc="parity samples"):
        z = np.load(r["path"])
        i = bid[r["sample_id"]]
        box = (boxes["center"][i], boxes["dims"][i], boxes["R"][i])
        origin, scan, gt, sh = z["origin"], z["scan"].astype(np.float64), z["gt"].astype(np.float64), z["gt_shadow"]
        res = evaluate_candidates(bk, origin, scan, box, gt, sh, grid, vs, a.block_margin, a.tol_vox, a.mem_mb, frame=a.voxel_frame)
        if res["nS"] == 0:
            print(f"\n  [warn] {r['sample_id']}: no shadow voxels in the envelope, skipped")
            continue
        H = envelope_half(box[1], grid, vs)
        valid = np.flatnonzero(~np.isnan(res["score_exact"]))
        cand = np.unique(np.concatenate([[grid.zero_index()], rng.choice(valid, min(a.n_cands, len(valid)), replace=False),
                                         [best_candidate(np.nan_to_num(res["score_tol"], nan=-1), grid.l1_all())[0]]]))
        for f in cand:
            b = grid.box_from(box, int(f))
            se, st, npr, viol = reference_scores(origin, scan, b, box, gt, sh, vs, m, a.tol_vox, H, a.voxel_frame)
            all_d.append(max(abs(se - res["score_exact"][f]), abs(st - res["score_tol"][f])))
            d_viol += abs(viol - res["n_viol"][f])
            d_pred += abs(npr - res["n_pred"][f])
            n_eq += int(all_d[-1] < 1e-9)
            n_tot += 1
        # direct check of the slab entry distances against Moller-Trumbore for a random candidate
        f = int(rng.choice(valid))
        b = grid.box_from(box, f)
        vec = scan - origin
        rr = np.linalg.norm(vec, axis=1)
        D = vec / rr[:, None]
        t_mt = moller_trumbore_first_hit(origin, D, box_triangles(b))
        tin, tout = ray_box_interval((origin - b[0]) @ b[2], D @ b[2], -b[1] / 2, b[1] / 2)
        hit = (tout >= tin) & (tin > 0)
        both = hit & np.isfinite(t_mt)
        t_err = max(t_err, float(np.abs(tin[both] - t_mt[both]).max()) if both.any() else 0.0)
        if (hit != np.isfinite(t_mt)).sum() > 0.01 * max(1, hit.sum()):
            t_err = max(t_err, 1.0)

    all_d = np.array(all_d)
    print(f"\n     compared {n_tot} (sample, candidate) pairs: exact equal {n_eq}/{n_tot}, "
          f"max |score diff| {all_d.max():.4f}, mean {all_d.mean():.5f}; sum |n_pred diff| {d_pred}, sum |n_viol diff| {d_viol}")
    gates.check("scores match the reference", np.mean(all_d <= 0.01) >= 0.95 and all_d.max() <= 0.05,
                f"{np.mean(all_d <= 0.01):.1%} of candidates within 0.01 (float32 vs float64 can flip single voxels at voxel borders)")
    gates.check("slab entry == Moller-Trumbore entry", t_err < 1e-5, f"max |t_slab - t_MT| = {t_err:.2e} m")
    gates.finish()


if __name__ == "__main__":
    main()