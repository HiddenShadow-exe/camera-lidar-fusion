#!/usr/bin/env python3
"""Face-shift grid and the fast batched evaluator (ray extension -> voxelisation -> comparison with GT).

Candidate = every face of the CV box moved independently along its OUTWARD normal (+ grows the box):
    per box axis k:  lo face at  -h_k - s_lo,  hi face at  +h_k + s_hi,   s in {-n..+n} * step
    6 shifts [lo_x, hi_x, lo_y, hi_y, lo_z, hi_z]  ->  (2n+1)^6 candidates (7^6 = 117 649 for n = 3)
The box slab test is separable per axis, so the (lo, hi) combinations of each axis are tabulated once and then
combined with max / min, which is far cheaper than 117k independent ray-box tests.

Evaluation of one candidate (the occluder never has to be known - the measured LiDAR range tells whether a ray was
blocked):
    * every LiDAR return of the OCCLUDED scan defines a ray (origin -> point) with measured range r
    * a ray is "blocked" if it enters the candidate box at t_in > r + margin  -> the box is behind what was measured
      -> a regenerated point  origin + t_in * dir  is created on the front face of the candidate
    * regenerated voxels P are compared with the ground-truth voxels G (LiDAR scan with the occluder removed).
      S (subset of G) = GT voxels that were in the shadow of the occluder (the part that has to be reconstructed)
          score = |P intersect S| / ( |S| + |P without G| )          (1 = perfect, 0 = nothing reconstructed)
      Voxels of P that lie in G but not in S are neutral. With tol > 0 the same score is computed with a tolerance of
      `tol` voxels (cube): recall via dilate(P), false positives via dilate(G).
    * n_viol = rays whose measured point lies BEHIND the entry of the candidate (the lidar saw through the box)
    * graded score (optional, see graded_spec): the hit / miss decision of both sides becomes a weight that decays with the
      distance (in voxels) to the nearest cell of the other set, w(d) = exp(-d^2 / (2 sigma^2)), 0 beyond `radius` voxels:
          recall    sum over s in S of  max_p w(s - p)                (every hidden GT cell is covered by its nearest regenerated cell)
          false pos sum over p in P of  1 - max_g w(p - g)            (every regenerated cell is matched to its nearest GT cell)
          graded    = recall / ( |S| + false pos )
      sigma -> 0 gives the exact score, w = 1 inside a cube gives the tolerance score. Because w is a monotone function of
      the squared offset, max_p w(s - p) = sum_j a_j * [s in D_j(P)] with nested dilations D_j and a_j = w_j - w_(j+1):
      the same batched dilation machinery as the tolerance score, just with a few more shells.
"""
from __future__ import annotations

import itertools

import numpy as np

from common import obb_corners, ray_box_interval, region_mask


# ----------------------------------------------------------------------------------------------
# backend shim: identical code path for numpy (CPU) and torch (CPU / CUDA)
# ----------------------------------------------------------------------------------------------
class Backend:
    def __init__(self, name="auto"):
        self.torch, self.dev = None, None
        if name in ("auto", "torch"):
            try:
                import torch
                if name == "torch" or torch.cuda.is_available():
                    self.torch = torch
                    self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            except ImportError:
                if name == "torch":
                    raise
        self.name = "numpy" if self.torch is None else f"torch[{self.dev}]"

    def a(self, x, kind="f"):
        if self.torch is None:
            return np.asarray(x, dtype={"f": np.float32, "i": np.int64, "b": np.bool_}[kind])
        t = self.torch
        return t.as_tensor(np.asarray(x), dtype={"f": t.float32, "i": t.int64, "b": t.bool}[kind], device=self.dev)

    def np(self, x):
        return x if self.torch is None else x.detach().cpu().numpy()

    def maximum(self, a, b):
        return np.maximum(a, b) if self.torch is None else self.torch.maximum(a, b)

    def minimum(self, a, b):
        return np.minimum(a, b) if self.torch is None else self.torch.minimum(a, b)

    def nonzero3(self, m):
        return np.nonzero(m) if self.torch is None else self.torch.nonzero(m, as_tuple=True)

    def floor_int(self, x):
        return np.floor(x).astype(np.int64) if self.torch is None else x.floor().long()

    def zeros_bool(self, n):
        return np.zeros(n, dtype=np.bool_) if self.torch is None else self.torch.zeros(n, dtype=self.torch.bool, device=self.dev)

    def rowsum(self, m):
        return m.reshape(m.shape[0], -1).sum(1)

    def copy(self, x):
        return x.copy() if self.torch is None else x.clone()

    def zeros_like(self, x):
        return np.zeros_like(x) if self.torch is None else self.torch.zeros_like(x)

    def shift_or(self, acc, src, off):
        """acc |= src shifted by the integer voxel offset off = (dx, dy, dz) along axes 1..3 of a (B, X, Y, Z) array (zero fill, in place)."""
        d_idx, s_idx = [slice(None)] * 4, [slice(None)] * 4
        for ax, o in zip((1, 2, 3), off):
            o = int(o)
            if o > 0:
                d_idx[ax], s_idx[ax] = slice(o, None), slice(None, -o)
            elif o < 0:
                d_idx[ax], s_idx[ax] = slice(None, o), slice(-o, None)
        acc[tuple(d_idx)] |= src[tuple(s_idx)]

    def dilate(self, occ, tol):
        """Cube dilation of a boolean (B, X, Y, Z) array by `tol` voxels (separable shifts, pure slicing)."""
        out = occ
        for ax in (1, 2, 3):
            acc = self.copy(out)
            for s in range(1, tol + 1):
                def sl(a, b):
                    idx = [slice(None)] * 4
                    idx[ax] = slice(a, b)
                    return tuple(idx)
                acc[sl(s, None)] |= out[sl(None, -s)]
                acc[sl(None, -s)] |= out[sl(s, None)]
            out = acc
        return out


# ----------------------------------------------------------------------------------------------
# shift grid
# ----------------------------------------------------------------------------------------------
class ShiftGrid:
    def __init__(self, n_steps=3, n_steps_z=None, step=0.03, min_dim=0.05):
        ns_z = n_steps if n_steps_z is None else n_steps_z
        self.ns = (int(n_steps), int(n_steps), int(ns_z))
        self.n = tuple(2 * s + 1 for s in self.ns)              # offsets per face
        self.C = tuple(n * n for n in self.n)                   # (lo, hi) combinations per axis
        self.step, self.min_dim = float(step), float(min_dim)
        self.total = int(np.prod(self.C))
        self.max_offset = np.array(self.ns, float) * self.step

    def describe(self):
        return (f"{self.total:,} candidates = {self.n[0]}x{self.n[0]} (x) * {self.n[1]}x{self.n[1]} (y) * "
                f"{self.n[2]}x{self.n[2]} (z) face offsets, step {self.step * 100:.1f} cm, "
                f"max +-{self.max_offset[0] * 100:.1f}/{self.max_offset[2] * 100:.1f} cm (xy/z)")

    def offsets(self, k):
        return (np.arange(self.n[k]) - self.ns[k]) * self.step

    def axis_bounds(self, k, h):
        """(lo, hi) positions in the box frame for every (lo-offset, hi-offset) combination of axis k."""
        s = self.offsets(k)
        s_lo, s_hi = np.repeat(s, self.n[k]), np.tile(s, self.n[k])
        return -h - s_lo, h + s_hi

    def zero_index(self):
        j = [self.ns[k] * self.n[k] + self.ns[k] for k in range(3)]
        return int((j[0] * self.C[1] + j[1]) * self.C[2] + j[2])

    def decode(self, flat):
        """flat index -> (..., 6) shifts [lo_x, hi_x, lo_y, hi_y, lo_z, hi_z] in metres (outward positive)."""
        js = np.unravel_index(np.asarray(flat), self.C)
        out = []
        for k, j in enumerate(js):
            s = self.offsets(k)
            out += [s[j // self.n[k]], s[j % self.n[k]]]
        return np.stack(out, -1)

    def encode(self, shifts):
        shifts = np.asarray(shifts)
        j = []
        for k in range(3):
            lo = np.rint(shifts[..., 2 * k] / self.step).astype(int) + self.ns[k]
            hi = np.rint(shifts[..., 2 * k + 1] / self.step).astype(int) + self.ns[k]
            j.append(lo * self.n[k] + hi)
        return (j[0] * self.C[1] + j[1]) * self.C[2] + j[2]

    def box_from(self, init_box, flat):
        c0, d0, R0 = init_box
        sh = self.decode(flat)
        lo, hi = -d0 / 2 - sh[0::2], d0 / 2 + sh[1::2]
        return c0 + R0 @ ((lo + hi) / 2), hi - lo, R0

    def valid_mask(self, d0):
        v = []
        for k in range(3):
            lo, hi = self.axis_bounds(k, d0[k] / 2)
            v.append((hi - lo) >= self.min_dim)
        return v

    def l1_all(self):
        return np.abs(self.decode(np.arange(self.total))).sum(-1)


def add_grid_args(ap):
    ap.add_argument("--voxel-size", type=float, default=0.03, help="voxel edge [m] used for the comparison")
    ap.add_argument("--n-steps", type=int, default=3, help="face offsets per direction (x/y faces)")
    ap.add_argument("--n-steps-z", type=int, default=None, help="face offsets per direction for top/bottom (default = n-steps)")
    ap.add_argument("--step", type=float, default=None, help="shift increment [m] (default = voxel size)")
    ap.add_argument("--min-dim", type=float, default=0.05, help="skip candidates thinner than this [m]")
    ap.add_argument("--tol-vox", type=int, default=1, help="tolerance of the 2nd (soft) score in voxels")
    ap.add_argument("--block-margin", type=float, default=None, help="ray counts as blocked if entry > range + margin [m] (default = voxel size)")
    ap.add_argument("--voxel-frame", choices=["box", "world"], default="box",
                    help="box: voxel grid aligned with the CV box (invariant to rigid motions of the scene; recommended for the labels). "
                         "world: aligned with the LiDAR frame (live-script protocol; the score depends on the scene's alignment)")


def grid_from_args(a):
    return ShiftGrid(a.n_steps, a.n_steps_z, a.step if a.step is not None else a.voxel_size, a.min_dim)


# ----------------------------------------------------------------------------------------------
# evaluator
# ----------------------------------------------------------------------------------------------
def graded_spec(sigma=1.0, radius=2):
    """Weights of the graded score. Cells are compared by their integer offset o to the nearest cell of the other set:
    w = exp(-|o|^2 / (2 sigma^2)) (sigma in voxels) for |o| <= radius voxels, 0 beyond.
    sigma = 1, radius = 2: exact 1.00 | face 0.61 | edge 0.37 | vertex 0.22 | 2 voxels straight 0.14 | everything farther 0."""
    R = int(radius)
    offs = np.array([o for o in itertools.product(range(-R, R + 1), repeat=3) if o[0] ** 2 + o[1] ** 2 + o[2] ** 2 <= R * R], np.int64)
    r2 = (offs ** 2).sum(1)
    levels = np.unique(r2)                                     # squared offsets that occur: 0, 1, 2, 3, 4 (radius 2)
    w = np.exp(-levels / (2.0 * float(sigma) ** 2))
    return dict(sigma=float(sigma), radius=R, r2=levels, w=w, a=w - np.append(w[1:], 0.0), shells=[offs[r2 == l] for l in levels])


def axis_slab_table(o_k, d_k, lo, hi, valid):
    """Entry/exit parameters along ONE axis for every (lo, hi) combination: (C, N) arrays. Invalid -> empty."""
    par = np.abs(d_k) < 1e-12
    ds = np.where(par, 1.0, d_k)
    t1, t2 = (lo[:, None] - o_k) / ds[None], (hi[:, None] - o_k) / ds[None]
    tmin, tmax = np.minimum(t1, t2), np.maximum(t1, t2)
    inside = (lo[:, None] <= o_k) & (o_k <= hi[:, None])
    tmin = np.where(par[None], np.where(inside, -np.inf, np.inf), tmin)
    tmax = np.where(par[None], np.where(inside, np.inf, -np.inf), tmax)
    tmin[~valid], tmax[~valid] = np.inf, -np.inf
    return tmin.astype(np.float32), tmax.astype(np.float32)


def envelope_half(d0, grid, vs):
    return d0 / 2 + grid.max_offset + vs


def evaluate_candidates(bk, origin, scan, box, gt, gt_shadow, grid, vs, block_margin=None, tol=1,
                        mem_mb=1500, on_progress=None, frame="box", graded=None):
    """Score ALL candidates of `grid` for one sample. Returns a dict of flat arrays of length grid.total.

    frame = "box":   voxels are aligned with the CV box (edges at multiples of vs from the box centre). A rigid motion of the
                     whole scene (rotation / translation) cannot change the table -> the labels do not depend on how the
                     scene happens to sit in the grid.
    frame = "world": voxels are aligned with the LiDAR frame, as in the live script. The score then depends on the alignment
                     of the scene with the grid (measured: oracle score 0.65 .. 0.88 for the SAME scene rotated)."""
    assert frame in ("box", "world")
    m = vs if block_margin is None else block_margin
    c0, d0, R0 = box
    H = envelope_half(d0, grid, vs)
    C0, C1, C2 = grid.C
    total = grid.total

    # ---- rays (box frame), pre-filtered by the envelope ------------------------------------------------------
    vec = scan - origin
    r_all = np.linalg.norm(vec, axis=1)
    ok = r_all > 1e-6
    vec, r_all = vec[ok], r_all[ok]
    d_w = vec / r_all[:, None]
    ol = (origin - c0) @ R0
    dl = d_w @ R0
    tin_e, tout_e = ray_box_interval(ol, dl, -H, H)
    keep = (tout_e >= tin_e) & (tout_e > 0)
    d_w, dl, r_all = d_w[keep], dl[keep], r_all[keep]
    N = len(r_all)

    # ---- dense voxel grid around the envelope, GT / shadow masks ---------------------------------------------
    if frame == "box":
        gmin = np.floor(-H / vs).astype(np.int64) - 1
        gmax = np.floor(H / vs).astype(np.int64) + 1
        to_frame = lambda p: (p - c0) @ R0
        o_use, d_use = ol, dl
    else:
        corners = obb_corners(c0, 2 * H, R0)
        gmin = np.floor(corners.min(0) / vs).astype(np.int64) - 1
        gmax = np.floor(corners.max(0) / vs).astype(np.int64) + 1
        to_frame = lambda p: p
        o_use, d_use = origin, d_w
    dims = gmax - gmin + 1
    nvox = int(np.prod(dims))

    def vox_mask(pts):
        pts = pts[region_mask(pts, box, H)]
        idx = np.floor(to_frame(pts) / vs).astype(np.int64) - gmin
        idx = idx[np.all((idx >= 0) & (idx < dims), axis=1)]
        mk = np.zeros(tuple(dims), bool)
        mk[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return mk

    Gm, Sm = vox_mask(gt), vox_mask(gt[gt_shadow])
    nS, nG = int(Sm.sum()), int(Gm.sum())
    out = dict(n_rays=N, nS=nS, nG=nG, grid_dims=tuple(int(x) for x in dims), nvox=nvox)

    score_e = np.full((C0, C1, C2), np.nan, np.float32)
    score_t = np.full((C0, C1, C2), np.nan, np.float32)
    n_pred = np.zeros((C0, C1, C2), np.int32)
    n_viol = np.zeros((C0, C1, C2), np.int32)
    score_g = np.full((C0, C1, C2), np.nan, np.float32) if graded is not None else None
    v0, v1, v2 = grid.valid_mask(d0)
    valid = v0[:, None, None] & v1[None, :, None] & v2[None, None, :]

    if nS == 0 or N == 0:
        score_e[valid] = 0.0
        score_t[valid] = 0.0
        out.update(score_exact=score_e.reshape(-1), score_tol=score_t.reshape(-1), n_pred=n_pred.reshape(-1),
                   n_viol=n_viol.reshape(-1))
        if graded is not None:
            score_g[valid] = 0.0
            out["score_graded"] = score_g.reshape(-1)
        return out

    # ---- tables + backend arrays ---------------------------------------------------------------------------
    tmn, tmx = [], []
    for k, v in zip(range(3), (v0, v1, v2)):
        lo, hi = grid.axis_bounds(k, d0[k] / 2)
        a, b = axis_slab_table(ol[k], dl[:, k], lo, hi, v)
        tmn.append(bk.a(a))
        tmx.append(bk.a(b))
    o_b, d_b, r_b = bk.a(o_use), bk.a(d_use), bk.a(r_all)
    gmin_b, dims_b = bk.a(gmin, "i"), bk.a(dims, "i")
    Gm_b, Sm_b = bk.a(Gm, "b"), bk.a(Sm, "b")
    Gd_b = bk.dilate(Gm_b[None], tol)[0] if tol > 0 else Gm_b
    X, Y, Z = (int(x) for x in dims)

    Gds = None
    if graded is not None:                                           # G dilated by the nested shells (one volume per sample)
        acc_g = bk.copy(Gm_b[None])
        Gds = [bk.copy(acc_g)]
        for sh in graded["shells"][1:]:
            for off in sh:
                bk.shift_or(acc_g, Gm_b[None], off)
            Gds.append(bk.copy(acc_g))

    per_cand = (7 if graded is not None else 6) * nvox + 48 * N
    B1 = int(max(1, min(C1, (mem_mb * 1e6) // (C2 * per_cand))))
    out["chunk_candidates"] = B1 * C2

    for j0 in range(C0):
        if on_progress:
            on_progress(j0, C0)
        if not v0[j0]:
            continue
        tin0, tout0 = tmn[0][j0], tmx[0][j0]
        for s1 in range(0, C1, B1):
            sl = slice(s1, min(C1, s1 + B1))
            b1 = sl.stop - sl.start
            tin = bk.maximum(bk.maximum(tin0[None, None, :], tmn[1][sl][:, None, :]), tmn[2][None, :, :])
            tout = bk.minimum(bk.minimum(tout0[None, None, :], tmx[1][sl][:, None, :]), tmx[2][None, :, :])
            hit = (tout >= tin) & (tin > 0)
            viol = bk.rowsum((hit & (tin < r_b - m)).reshape(b1 * C2, N))
            blocked = hit & (tin > r_b + m)
            bi, ci, ni = bk.nonzero3(blocked)
            B = b1 * C2
            occ = bk.zeros_bool(B * nvox)
            if len(bi) > 0:
                pts = o_b + tin[bi, ci, ni][:, None] * d_b[ni]
                vox = bk.floor_int(pts / vs) - gmin_b
                inb = ((vox >= 0) & (vox < dims_b)).all(1)
                vox, cand = vox[inb], (bi * C2 + ci)[inb]
                occ[cand * nvox + (vox[:, 0] * Y + vox[:, 1]) * Z + vox[:, 2]] = True
            occ = occ.reshape(B, X, Y, Z)
            npr = bk.np(bk.rowsum(occ)).astype(np.int64)
            in_g = bk.np(bk.rowsum(occ & Gm_b)).astype(np.int64)
            in_s = bk.np(bk.rowsum(occ & Sm_b)).astype(np.int64)
            se = in_s / (nS + (npr - in_g))
            if tol > 0:
                rec = bk.np(bk.rowsum(bk.dilate(occ, tol) & Sm_b)).astype(np.int64)
                fpt = bk.np(bk.rowsum(occ & ~Gd_b)).astype(np.int64)
                st = rec / (nS + fpt)
            else:
                st = se
            if graded is not None:
                a_ = graded["a"]
                rec_w = a_[0] * bk.np(bk.rowsum(occ & Sm_b)).astype(np.float64)
                hit_g = a_[0] * bk.np(bk.rowsum(occ & Gds[0])).astype(np.float64)
                Dm = bk.copy(occ)
                for j in range(1, len(a_)):
                    for off in graded["shells"][j]:
                        bk.shift_or(Dm, occ, off)
                    rec_w += a_[j] * bk.np(bk.rowsum(Dm & Sm_b)).astype(np.float64)
                    hit_g += a_[j] * bk.np(bk.rowsum(occ & Gds[j])).astype(np.float64)
                score_g[j0, sl, :] = (rec_w / (nS + (npr - hit_g))).astype(np.float32).reshape(b1, C2)
            score_e[j0, sl, :] = se.reshape(b1, C2)
            score_t[j0, sl, :] = st.reshape(b1, C2)
            n_pred[j0, sl, :] = npr.reshape(b1, C2)
            n_viol[j0, sl, :] = bk.np(viol).reshape(b1, C2)
    score_e[~valid] = np.nan
    score_t[~valid] = np.nan
    out.update(score_exact=score_e.reshape(-1), score_tol=score_t.reshape(-1), n_pred=n_pred.reshape(-1),
               n_viol=n_viol.reshape(-1))
    if graded is not None:
        score_g[~valid] = np.nan
        out["score_graded"] = score_g.reshape(-1)
    return out


def best_candidate(scores, l1):
    """argmax with ties (plateaus of exact hit/miss!) broken towards the smallest total shift."""
    mx = np.nanmax(scores)
    ties = np.flatnonzero(scores >= mx - 1e-9)
    return int(ties[np.argmin(l1[ties])]), float(mx), int(len(ties))