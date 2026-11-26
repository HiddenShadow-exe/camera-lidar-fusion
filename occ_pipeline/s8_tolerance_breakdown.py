#!/usr/bin/env python3
"""STAGE 8 - exact match and 1-voxel-neighbourhood breakdown of OUR pipeline's choices (CV box, best constant, model, oracle).

For every real sample and every method the chosen face shifts define a box; the LiDAR rays blocked by the occluder are extended
into it (same rule as stage 4), the regenerated points are voxelised and compared with the ground truth on the stage-4 voxel grid
(same voxel size / frame as work/stage4/config.json):
    P = regenerated voxels        S = GT voxels that the occluder hid (the part that has to be recovered)       G = all GT voxels
    precision side  every cell of P  -> nearest cell of G     (hit = the regenerated cell is on / next to real GT)
    recall side     every cell of S  -> nearest cell of P     (hit = the hidden GT cell has a regenerated cell on / next to it)
The class of the nearest cell: exact | face | edge | vertex | > 1 voxel away, and the four cumulative levels
    exact  |  exact+face  |  exact+face+edge  |  exact+face+edge+vertex      (same definitions as occupancy_tolerance_eval.py)
P / R / F1 are means over samples. The stacked bars are pooled cell counts. Parity check: the exact score recomputed here must equal
the value in the stage-4 table for the chosen candidate.

Needs occupancy_tolerance_eval.py in the same folder (cell classes + stacked-bar helper), numpy, scipy, matplotlib.
    python s8_tolerance_breakdown.py --work work_virt --split cv        # after s6_train.py --cv 5
Outputs in <work>/stage8<tag>/: 1_exact_match.png, 2_tolerance_levels.png, tolerance_breakdown.csv, tolerance_per_sample.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import Gates, banner, progress, read_csv, region_mask, step, write_csv
from metrics import best_constant, bootstrap_ci, clean_scores
from occupancy_tolerance_eval import (CLASSES, CLASS_COLORS, HIT_MISS_COLORS, LEVELS, LEVEL_NAMES, _label_bars, _stack,
                                      cell_classes, composition, prf_level)
from s7_evaluate import feasible_constant, load
from shiftgrid import ShiftGrid, best_candidate, envelope_half
from visualize_scene import regen_points


def choices_for(work, split, pred, grid):
    """Same choices as stage 7 (tol-score based constant / oracle). Returns meta, tabs, {method: flat idx}, fold."""
    meta, tabs = load(work, split)
    n = len(meta)
    zero, l1 = grid.zero_index(), grid.l1_all()
    ch = {"CV box (shift 0)": np.full(n, zero)}
    zp = np.load(pred) if Path(pred).exists() else None
    fold = np.zeros(n, int)
    T = tabs["tol"]
    if split == "cv":
        if zp is None or "fold" not in zp.files:
            raise SystemExit(f"{pred} not found / no folds: run  python s6_train.py --work {work} --cv 5  first")
        fo = dict(zip(zp["ids"], zp["fold"]))
        fold = np.array([fo[m["sample_id"]] for m in meta])
        ci = np.zeros(n, np.int64)
        use_virtual = bool(zp["use_virtual"]) if "use_virtual" in zp.files else True
        for f in np.unique(fold[fold >= 0]):
            _, mu = best_constant(T[(fold != f) & ((fold >= 0) | use_virtual)], return_mean=True)
            ci[fold == f] = feasible_constant(T[fold == f], mu)
    else:
        _, trt = load(work, "train")
        _, mu = best_constant(trt["tol"], return_mean=True)
        ci = feasible_constant(T, mu)
    ch["best constant"] = ci
    if zp is not None:
        order = {s_: i for i, s_ in enumerate(zp["ids"])}
        if all(m["sample_id"] in order for m in meta):
            ch["model"] = np.array([zp["idx"][order[m["sample_id"]]] for m in meta])
        else:
            print(f"  [warn] {pred} does not cover all samples - model skipped")
    ch["oracle (tol score)"] = np.array([best_candidate(np.asarray(tabs["tol"][i], np.float32), l1)[0] for i in range(n)])
    ch["oracle (exact score)"] = np.array([best_candidate(np.asarray(tabs["exact"][i], np.float32), l1)[0] for i in range(n)])
    if "graded" in tabs:
        ch["oracle (graded score)"] = np.array([best_candidate(np.asarray(tabs["graded"][i], np.float32), l1)[0] for i in range(n)])
    return meta, tabs, ch, fold


def cells(points, to_frame, vs):
    if len(points) == 0:
        return np.zeros((0, 3), np.int64)
    return np.unique(np.floor(to_frame(np.asarray(points, np.float64)) / vs).astype(np.int64), axis=0)


def pack(cells_arr):
    return set(map(tuple, cells_arr.tolist()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--split", default="cv", help="cv | val | train")
    ap.add_argument("--pred", default=None, help="npz with idx/ids (default work/stage6/pred_<split>.npz)")
    ap.add_argument("--all-kinds", action="store_true", help="also the augmented copies (default: real samples only)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N samples")
    ap.add_argument("--tag", default="", help="suffix used by s6_train.py --tag (reads pred_<split><tag>.npz, writes stage8<tag>/)")
    a = ap.parse_args()

    banner(8, f"exact / 1-voxel-tolerance breakdown on the '{a.split}' split")
    gates = Gates(8)
    work = Path(a.work)
    cfg = json.loads((work / "stage4" / "config.json").read_text())
    grid = ShiftGrid(cfg["n_steps"], cfg["n_steps_z"], cfg["step"], cfg["min_dim"])
    vs, margin, frame = cfg["voxel_size"], cfg["block_margin"], cfg["voxel_frame"]
    pred = a.pred or str(work / "stage6" / f"pred_{a.split}{a.tag}.npz")
    meta, tabs, ch, fold = choices_for(a.work, a.split, pred, grid)
    methods = list(ch)
    s0 = {r["sample_id"]: r for r in read_csv(work / "stage0" / "index.csv")}
    boxes = np.load(work / "stage1" / "boxes.npz")
    bid = {s: i for i, s in enumerate(boxes["ids"])}
    sel = [i for i, m in enumerate(meta) if fold[i] >= 0 and (a.all_kinds or m["kind"] == "orig")]
    if a.limit:
        sel = sel[:a.limit]
    step(f"{len(sel)} samples x {len(methods)} methods | voxel {vs * 100:.1f} cm ({frame}-aligned), block margin {margin * 100:.1f} cm | "
         f"methods: {', '.join(methods)}")

    per_stats = {m: [] for m in methods}      # method -> list of dicts (per sample)
    rows, parity = [], []
    for i in progress(sel, total=len(sel), desc="samples"):
        sid = meta[i]["sample_id"]
        b = bid[sid]
        cv = (boxes["center"][b], boxes["dims"][b], boxes["R"][b])
        c0, d0, R0 = cv
        z = np.load(s0[sid]["path"])
        origin, scan = z["origin"], z["scan"].astype(np.float64)
        gt, shadow = z["gt"].astype(np.float64), z["gt_shadow"]
        H = envelope_half(d0, grid, vs)
        to_frame = (lambda p: (p - c0) @ R0) if frame == "box" else (lambda p: p)
        gin = gt[region_mask(gt, cv, H)]
        sin = gt[shadow & region_mask(gt, cv, H)]
        G, S = cells(gin, to_frame, vs), cells(sin, to_frame, vs)
        Gset, Sset = pack(G), pack(S)
        for m in methods:
            f = int(ch[m][i])
            P = cells(regen_points(origin, scan, grid.box_from(cv, f), margin), to_frame, vs)
            Pset = pack(P)
            own = len(Pset & Sset) / (len(Sset) + len(Pset - Gset)) if len(Sset) else 0.0
            parity.append(abs(own - float(np.asarray(tabs["exact"][i], np.float32)[f])))
            cls_pg, d_pg = cell_classes(P, G, vs * 100)          # regenerated cell -> nearest GT cell
            cls_sp, d_sp = cell_classes(S, P, vs * 100)          # hidden GT cell -> nearest regenerated cell
            st = {"cls_pg": cls_pg, "cls_gp": cls_sp, "d_pg": d_pg, "d_gp": d_sp, "scene": meta[i]["scene"]}
            per_stats[m].append(st)
            row = {"sample_id": sid, "scene": meta[i]["scene"], "category": meta[i]["category"], "method": m, "chosen": f,
                   "shift_cm": " ".join(f"{x:+.0f}" for x in grid.decode(f) * 100), "P_cells": len(Pset), "S_cells": len(Sset),
                   "G_cells": len(Gset), "exact_score_recomputed": own}
            for k in range(4):
                p, r, f1 = prf_level(st, k)
                row.update({f"P_{LEVEL_NAMES[k]}": p, f"R_{LEVEL_NAMES[k]}": r, f"F1_{LEVEL_NAMES[k]}": f1})
            rows.append(row)

    # ---- aggregate ----------------------------------------------------------------------------------------------
    scenes = np.array([s["scene"] for s in per_stats[methods[0]]])
    agg = {}
    for m in methods:
        st = per_stats[m]
        agg[m] = {"prf": [np.mean([prf_level(s, k) for s in st], axis=0) for k in range(4)],
                  "comp_pg": composition(st, "pg"), "comp_sp": composition(st, "gp")}

    print(f"\n  {len(sel)} samples in {len(set(scenes))} scene(s).  P = share of regenerated cells that are a hit, R = share of hidden GT cells that are a hit "
          f"(%, mean over samples)")
    print(f"  {'method':<22}" + "".join(f"{n:>34}" for n in LEVEL_NAMES))
    for m in methods:
        print(f"  {m:<22}" + "".join(f"{agg[m]['prf'][k][0] * 100:>10.1f} /{agg[m]['prf'][k][1] * 100:>5.1f} /{agg[m]['prf'][k][2] * 100:>5.1f}     " for k in range(4)))
    print("      (each cell: P / R / F1)")
    print(f"\n  F1 with 95% CI over scenes:")
    for m in methods:
        txt = []
        for k in range(4):
            v = np.array([prf_level(s, k)[2] for s in per_stats[m]])
            mu, lo, hi = bootstrap_ci(v, scenes, a.bootstrap)
            txt.append(f"{mu * 100:5.1f} [{lo * 100:4.1f},{hi * 100:4.1f}]" if np.isfinite(lo) else f"{mu * 100:5.1f}")
        print(f"  {m:<22}" + "".join(f"{t:>34}" for t in txt))
    print("\n  neighbourhood breakdown of the pooled cells [% exact / face / edge / vertex / far]")
    for m in methods:
        fmt = lambda c: " / ".join(f"{x:5.1f}" for x in (c / c.sum() * 100 if c.sum() else np.zeros(5)))
        print(f"  {m:<22} regenerated->GT {fmt(agg[m]['comp_pg'])}   hidden GT->regenerated {fmt(agg[m]['comp_sp'])}")

    # ---- figures ------------------------------------------------------------------------------------------------
    out = work / f"stage8{a.tag}"
    out.mkdir(exist_ok=True)
    x = np.arange(len(methods))
    xl = [m.replace(" (", "\n(") for m in methods]
    bw = 0.27
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(max(17, 2.2 * len(methods) + 8), 6.8))
    for k, (name, color, dx) in enumerate((("Precision", "C0", -bw), ("Recall", "C1", 0), ("F1", "C2", bw))):
        bars = a0.bar(x + dx, [agg[m]["prf"][0][k] * 100 for m in methods], bw, label=name, color=color)
        _label_bars(a0, bars)
    a0.set_ylim(0, 115); a0.set_yticks(range(0, 101, 20)); a0.set_ylabel("%")
    a0.set_xticks(x); a0.set_xticklabels(xl, fontsize=8)
    a0.set_title("Precision / Recall / F1 (exact cell match)", fontsize=10)
    a0.legend(fontsize=8, loc="upper right"); a0.grid(axis="y", alpha=.3)
    hm = lambda key: [np.array([agg[m][key][0], agg[m][key][1:].sum()]) for m in methods]
    _stack(a1, hm("comp_pg"), ["hit (exact GT cell)", "miss (= false positive)"], xl, HIT_MISS_COLORS,
           "Where the regenerated cells find their GT", "% of regenerated cells")
    _stack(a2, hm("comp_sp"), ["hit (exact regenerated cell)", "miss"], xl, HIT_MISS_COLORS,
           "Where the hidden GT cells find a regenerated cell", "% of hidden GT cells (S)")
    fig.suptitle(f"Exact match  -  voxel {vs * 100:g} cm, {len(sel)} samples", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "1_exact_match.png", dpi=150)
    plt.close(fig)

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(max(19, 2.4 * len(methods) + 9), 7.4))
    lx = np.arange(4)
    wb = 0.8 / len(methods)
    for j, m in enumerate(methods):
        bars = a0.bar(lx + (j - (len(methods) - 1) / 2) * wb, [agg[m]["prf"][k][2] * 100 for k in range(4)], wb, label=m)
        _label_bars(a0, bars)
    a0.set_ylim(0, 135); a0.set_yticks(range(0, 101, 20)); a0.set_ylabel("F1 (%)")
    a0.set_xticks(lx); a0.set_xticklabels(LEVELS, fontsize=8)
    a0.set_title("F1 per tolerance level", fontsize=10)
    a0.legend(fontsize=7, loc="upper left", ncol=2); a0.grid(axis="y", alpha=.3)
    _stack(a1, [agg[m]["comp_pg"] for m in methods], CLASSES, xl, CLASS_COLORS,
           "Regenerated -> GT: neighbourhood breakdown", "% of regenerated cells")
    _stack(a2, [agg[m]["comp_sp"] for m in methods], CLASSES, xl, CLASS_COLORS,
           "Hidden GT -> regenerated: neighbourhood breakdown", "% of hidden GT cells (S)")
    fig.suptitle(f"Tolerance levels (1-voxel neighbourhood)  -  voxel {vs * 100:g} cm, {len(sel)} samples", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "2_tolerance_levels.png", dpi=150)
    plt.close(fig)

    write_csv(out / "tolerance_per_sample.csv", rows)
    summ = []
    for m in methods:
        for k in range(4):
            summ.append({"method": m, "level": LEVEL_NAMES[k], "precision": agg[m]["prf"][k][0], "recall": agg[m]["prf"][k][1], "f1": agg[m]["prf"][k][2],
                         **{f"pg_{n}_%": agg[m]["comp_pg"][i] / max(agg[m]["comp_pg"].sum(), 1) * 100 for i, n in enumerate(("exact", "face", "edge", "vertex", "far"))},
                         **{f"sp_{n}_%": agg[m]["comp_sp"][i] / max(agg[m]["comp_sp"].sum(), 1) * 100 for i, n in enumerate(("exact", "face", "edge", "vertex", "far"))}})
    write_csv(out / "tolerance_breakdown.csv", summ)

    # ---- gates --------------------------------------------------------------------------------------------------
    parity = np.array(parity)
    gates.check("exact score recomputed here equals the stage-4 table", parity.mean() < 0.01 and np.percentile(parity, 95) < 0.03,
                f"mean |diff| {parity.mean():.4f}, p95 {np.percentile(parity, 95):.4f}, max {parity.max():.3f} (float16 tables + float32/64 voxel-edge cases)")
    mono = all(agg[m]["prf"][k][2] <= agg[m]["prf"][k + 1][2] + 1e-9 for m in methods for k in range(3))
    gates.check("F1 does not decrease with a more lenient tolerance", mono, "levels are nested")
    print(f"\n  figures + csv: {out}")
    gates.finish()


if __name__ == "__main__":
    main()