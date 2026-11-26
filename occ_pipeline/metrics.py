#!/usr/bin/env python3
"""Numpy-only conventions and metrics shared by the model (stage 6) and the evaluation (stage 7).

Flat candidate index  f = (j0 * C1 + j1) * C2 + j2   where  j_k = lo_offset_idx * n_k + hi_offset_idx   (see ShiftGrid)
So a score table of shape (total,) reshapes to (C0, C1, C2), and the model predicts three categorical distributions
(one per box axis, over its C_k = n_k^2 (lo, hi) combinations). The argmax of the product distribution is the
per-axis argmax, so predicting a flat index = combining the three per-axis argmaxes.
"""
import numpy as np


def table_to_grid(tab, C):
    """(..., total) -> (..., C0, C1, C2)"""
    return np.asarray(tab).reshape(*np.asarray(tab).shape[:-1], *C)


def flat_from_axes(j, C):
    """per-axis indices j = (j0, j1, j2) (arrays of equal shape) -> flat candidate index"""
    return (np.asarray(j[0]) * C[1] + np.asarray(j[1])) * C[2] + np.asarray(j[2])


def axes_from_flat(f, C):
    return np.unravel_index(np.asarray(f), C)


def expected_loss(p0, p1, p2, loss_grid):
    """E_p[loss] for the factorised distribution. p_k: (B, C_k); loss_grid: (B, C0, C1, C2)."""
    return np.einsum("bi,bj,bk,bijk->b", p0, p1, p2, loss_grid)


def clean_scores(tab):
    """float scores with invalid (NaN) candidates -> -inf, so they never win an argmax."""
    t = np.asarray(tab, dtype=np.float32)
    return np.where(np.isnan(t), -np.inf, t)


def sample_metrics(tab, chosen, zero_idx, l1, tie_tol=1e-3):
    """Per-sample metrics for one score table (total,) and a chosen flat index.
    regret = best - chosen (>= 0). in_tie = chosen is optimal up to tie_tol. The oracle is tie-broken towards the smallest shift."""
    t = clean_scores(tab)
    best = float(t.max())
    ties = np.flatnonzero(t >= best - 1e-9)
    oracle = int(ties[np.argmin(l1[ties])])
    s = float(t[chosen])
    return {"score": s, "best": best, "regret": best - s, "zero": float(t[zero_idx]), "zero_regret": best - float(t[zero_idx]),
            "in_tie": bool(s >= best - tie_tol), "oracle_idx": oracle, "n_ties": int(len(ties))}


def best_constant(tabs, return_mean=False):
    """Single candidate that maximises the MEAN score over all (training) samples: the 'no learning' baseline."""
    acc = np.zeros(tabs.shape[1], np.float64)
    cnt = np.zeros(tabs.shape[1], np.float64)
    for i in range(0, len(tabs), 64):
        t = np.asarray(tabs[i:i + 64], np.float32)
        ok = ~np.isnan(t)
        acc += np.where(ok, t, 0).sum(0)
        cnt += ok.sum(0)
    mean = np.where(cnt >= 0.5 * len(tabs), acc / np.maximum(cnt, 1), -np.inf)
    return (int(np.argmax(mean)), mean) if return_mean else int(np.argmax(mean))


def bootstrap_ci(values, groups, n=2000, seed=0):
    """95% CI of the mean, resampling whole GROUPS (scenes) - augmented copies of one scene are not independent."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    ug = np.unique(groups)
    if len(ug) < 2:
        return float(values.mean()), float("nan"), float("nan")
    per = [values[groups == g] for g in ug]
    rng = np.random.default_rng(seed)
    means = [np.concatenate([per[i] for i in rng.integers(0, len(per), len(per))]).mean() for _ in range(n)]
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))