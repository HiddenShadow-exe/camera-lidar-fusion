#!/usr/bin/env python3
"""
Occupancy evaluation with 1-voxel-neighbourhood tolerances (companion of evaluate_dataset.py / the occupancy script;
keep it in the same folder. It only IMPORTS your evaluation module, nothing of it is modified).

For every voxel size (default 3 cm) the predicted cells are compared with the GT cells on the voxel grid. For each cell
the NEAREST cell of the other set decides the class
    exact   same cell                                  (offset 0,0,0)
    face    shares a face                              (offset sum 1)
    edge    shares an edge                             (offset sum 2)
    vertex  shares a vertex                            (offset sum 3)
    far     more than 1 cell away in some axis
and four evaluation levels are built cumulatively from these classes
    exact                          hit = exact
    exact + face                   hit = exact | face
    exact + face + edge            hit = exact | face | edge
    exact + face + edge + vertex   hit = exact | face | edge | vertex      (= your "relaxed 1-voxel" neighbourhood)
Precision = share of predicted cells that are a hit, recall = share of GT cells that are a hit, F1 as usual. Per category
the P / R / F1 are means over scenes (as in your occupancy script), the stacked bars are pooled cell counts.

Outputs (default <OUTPUT_DIR>/occupancy_tolerance):
    1_exact_match_<v>cm.png          P / R / F1 and the hit / miss split, exact match only
    2_tolerance_levels_<v>cm.png     P / R / F1 for the four levels + stacked exact / face / edge / vertex / far split
    tolerance_metrics.csv            per category, voxel size and level
    per_scene_tolerance.csv          per scene
    (and the same numbers printed on the console)

Usage:
    python occupancy_tolerance_eval.py --dataset ../dataset --pipeline lidar-ray-extension-voxelized.py
    python occupancy_tolerance_eval.py --voxel-sizes 3 5            # several voxel sizes -> one pair of figures each
    python occupancy_tolerance_eval.py --eval-module dataset_evaluation   # if your evaluation module has that name
"""
import argparse
import csv
import importlib
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["Exact match", "1 voxel (shared face)", "1 voxel (shared edge)", "1 voxel (shared vertex)", "> 1 voxel away"]
CLASS_COLORS = ["#1a9850", "#91cf60", "#d9ef8b", "#fee08b", "#d73027"]
LEVELS = ["Exact", "Exact + face", "Exact + face\n+ edge", "Exact + face\n+ edge + vertex"]
LEVEL_NAMES = ["exact", "exact+face", "exact+face+edge", "exact+face+edge+vertex"]
HIT_MISS_COLORS = ["#1a9850", "#d73027"]


# ----------------------------------------------------------------------------------------------
# Import of the user's evaluation module
# ----------------------------------------------------------------------------------------------
def import_eval(name):
    sys.path.insert(0, str(Path.cwd()))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    names = [name] if name else ["evaluate_dataset", "dataset_evaluation"]
    last = None
    for n in names:
        try:
            return importlib.import_module(n)
        except ModuleNotFoundError as e:
            if e.name != n:          # the module exists but one of ITS imports is missing (e.g. open3d)
                raise
            last = e
    raise SystemExit(f"Could not import the evaluation module {names}: {last}\n"
                     f"Put this script next to it or pass --eval-module <name>.")


# ----------------------------------------------------------------------------------------------
# Cells and classes
# ----------------------------------------------------------------------------------------------
def occupied_cells(points, voxel_size):
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.unique(np.floor(np.asarray(points, dtype=np.float64) / voxel_size).astype(np.int64), axis=0)


def cell_classes(A, B, voxel_cm):
    """For every cell of A: class (0 exact, 1 face, 2 edge, 3 vertex, 4 far) of the nearest cell of B, and the
    Euclidean distance to it in cm (inf if B is empty)."""
    n = len(A)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    if len(B) == 0:
        return np.full(n, 4, dtype=np.int64), np.full(n, np.inf)
    d, j = cKDTree(B).query(A, k=1)
    off = np.abs(B[j] - A)
    cls = np.where(off.max(1) > 1, 4, off.sum(1))          # max 0 -> sum 0 (exact); max 1 -> sum 1 / 2 / 3
    return cls.astype(np.int64), d * voxel_cm


def scene_stats(pred_points, gt_points, voxel_cm):
    v = voxel_cm / 100.0
    P, G = occupied_cells(pred_points, v), occupied_cells(gt_points, v)
    cls_pg, d_pg = cell_classes(P, G, voxel_cm)
    cls_gp, d_gp = cell_classes(G, P, voxel_cm)
    return {"cls_pg": cls_pg, "cls_gp": cls_gp, "d_pg": d_pg, "d_gp": d_gp, "n_pred": len(P), "n_gt": len(G)}


def prf_level(st, level):
    """Precision / recall / F1 when classes 0..level count as a hit."""
    p = float((st["cls_pg"] <= level).mean()) if len(st["cls_pg"]) else 0.0
    r = float((st["cls_gp"] <= level).mean()) if len(st["cls_gp"]) else 0.0
    return p, r, (2 * p * r / (p + r) if p + r > 0 else 0.0)


def composition(stats, side):
    """Pooled cell counts per class (5 values)."""
    tot = np.zeros(5)
    for s in stats:
        tot += np.bincount(s["cls_" + side], minlength=5)[:5]
    return tot


def error_stats(stats, side):
    d = [s["d_" + side][np.isfinite(s["d_" + side])] for s in stats]
    d = [x for x in d if len(x)]
    if not d:
        return float("nan"), float("nan")
    d = np.concatenate(d)
    return float(d.mean()), float(d.max())


def summarize(stats):
    out = {"n_scenes": len(stats)}
    for k in range(4):
        out[f"prf{k}"] = np.mean([prf_level(s, k) for s in stats], axis=0) if stats else np.zeros(3)
    out["comp_pg"], out["comp_gp"] = composition(stats, "pg"), composition(stats, "gp")
    out["err_pg"], out["err_gp"] = error_stats(stats, "pg"), error_stats(stats, "gp")
    return out


# ----------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------
def _label_bars(ax, bars, fmt="{:.0f}", min_h=0.0):
    for rect in bars:
        h = rect.get_height()
        if h > min_h:
            ax.annotate(fmt.format(h), (rect.get_x() + rect.get_width() / 2, h), textcoords="offset points",
                        xytext=(0, 2), ha="center", fontsize=7)


def _stack(ax, comps, labels, xlabels, colors, title, ylabel):
    n_bins = len(colors)
    fr = np.array([c / c.sum() * 100 if c.sum() else np.zeros(n_bins) for c in comps])
    bottom = np.zeros(len(comps))
    x = np.arange(len(comps))
    for k in range(n_bins):
        ax.bar(x, fr[:, k], 0.6, bottom=bottom, color=colors[k], label=labels[k], edgecolor="white", lw=.6)
        for xi, (b, h) in enumerate(zip(bottom, fr[:, k])):
            if h >= 4:
                ax.text(xi, b + h / 2, f"{h:.0f}", ha="center", va="center", fontsize=8)
        bottom += fr[:, k]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2 if n_bins > 3 else 1)


def _wrap(name, sep):
    import textwrap
    return textwrap.fill(name, 14) + ("\n(not in avg.)" if sep else "")


def plot_exact(groups, order, sep_flags, voxel_cm, out_path, all_name):
    """Figure 1: exact match only. x = groups (pooled + each category)."""
    xl = [_wrap(g, sep_flags[g]) for g in order]
    x = np.arange(len(order))
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(max(16, 1.6 * len(order) + 10), 6.8))
    bw = 0.27
    for k, (name, color, dx) in enumerate((("Precision", "C0", -bw), ("Recall", "C1", 0), ("F1", "C2", bw))):
        vals = [groups[g]["prf0"][k] * 100 for g in order]
        bars = a0.bar(x + dx, vals, bw, label=name, color=color)
        for rect, g in zip(bars, order):
            if sep_flags[g]:
                rect.set_hatch("//"); rect.set_alpha(.6)
        _label_bars(a0, bars)
    a0.set_ylim(0, 115); a0.set_yticks(range(0, 101, 20)); a0.set_ylabel("%")
    a0.set_xticks(x); a0.set_xticklabels(xl, fontsize=8)
    a0.set_title("Precision / Recall / F1 (exact cell match)", fontsize=10)
    a0.legend(fontsize=8, loc="upper right"); a0.grid(axis="y", alpha=.3)

    comp = lambda side: [np.array([groups[g]["comp_" + side][0], groups[g]["comp_" + side][1:].sum()]) for g in order]
    _stack(a1, comp("pg"), ["hit (exact GT cell)", "miss (= false positive)"], xl, HIT_MISS_COLORS,
           "Where the predicted cells find their GT", "% of predicted cells")
    _stack(a2, comp("gp"), ["hit (exact predicted cell)", "miss"], xl, HIT_MISS_COLORS,
           "Where the GT cells find their prediction", "% of GT cells")
    fig.suptitle(f"Exact match  -  voxel {voxel_cm:g} cm", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.925, f"first group = {all_name}; hatched = not in the average", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_levels(groups, order, sep_flags, voxel_cm, out_path, pooled_name):
    """Figure 2: the four tolerance levels (P/R/F1) + stacked neighbourhood breakdown per group."""
    xl = [_wrap(g, sep_flags[g]) for g in order]
    x = np.arange(len(LEVELS))
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(max(18, 1.6 * len(order) + 12), 7.2))

    bw = 0.27
    for k, (name, color, dx) in enumerate((("Precision", "C0", -bw), ("Recall", "C1", 0), ("F1", "C2", bw))):
        vals = [groups[pooled_name][f"prf{lv}"][k] * 100 for lv in range(4)]
        bars = a0.bar(x + dx, vals, bw, label=name, color=color)
        _label_bars(a0, bars)
    styles = ["--", ":", "-."]
    j = 0
    for g in order:                       # separate categories: dashed lines on top of the pooled bars
        if sep_flags[g]:
            for k, color in enumerate(("C0", "C1", "C2")):
                a0.plot(x + (k - 1) * bw, [groups[g][f"prf{lv}"][k] * 100 for lv in range(4)], styles[j % 3] + "o",
                        color=color, lw=1.6, ms=4, mfc="white")
            a0.plot([], [], "k" + styles[j % 3], label=f"{g}: not in the average")
            j += 1
    a0.set_ylim(0, 135); a0.set_yticks(range(0, 101, 20)); a0.set_ylabel("%")
    a0.set_xticks(x); a0.set_xticklabels(LEVELS, fontsize=8)
    a0.set_title(f"Precision / Recall / F1 per tolerance ({pooled_name})", fontsize=10)
    a0.legend(fontsize=8, loc="upper center", ncol=2); a0.grid(axis="y", alpha=.3)

    _stack(a1, [groups[g]["comp_pg"] for g in order], CLASSES, xl, CLASS_COLORS,
           "Pred -> GT: neighbourhood breakdown", "% of predicted cells")
    _stack(a2, [groups[g]["comp_gp"] for g in order], CLASSES, xl, CLASS_COLORS,
           "GT -> Pred: neighbourhood breakdown", "% of GT cells")
    fig.suptitle(f"Tolerance levels (1-voxel neighbourhood)  -  voxel {voxel_cm:g} cm", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-module", default=None, help="module with your evaluation code (default: evaluate_dataset, "
                                                       "then dataset_evaluation)")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--pipeline", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--offset-mode", choices=["geometry", "prediction"], default=None)
    ap.add_argument("--voxel-sizes", nargs="+", type=float, default=[3.0], metavar="CM", help="default: 3")
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs per category (quick test)")
    args = ap.parse_args()

    ev = import_eval(args.eval_module)
    is_sep = getattr(ev, "is_separate", lambda c: False)
    all_name = getattr(ev, "all_label", lambda: "ALL")()
    dataset = args.dataset or ev.DATASET_DIR
    pipeline = args.pipeline or ev.PIPELINE_SOURCE
    out = Path(args.output or (Path(ev.OUTPUT_DIR) / "occupancy_tolerance"))
    out.mkdir(parents=True, exist_ok=True)
    mode = args.offset_mode or ev.CALIB_OFFSET_MODE
    voxels = sorted(set(args.voxel_sizes))

    funcs, _ = ev.load_pipeline(pipeline)
    print(f"evaluation module: {ev.__name__} | voxel sizes {voxels} cm | offset mode {mode}")

    categories = sorted(p for p in Path(dataset).iterdir() if p.is_dir() and not p.name.startswith("_"))
    store = {}                       # (category, v_cm) -> list of scene stats
    scene_rows = []
    n_err = 0
    for cat_dir in categories:
        category = cat_dir.name
        pairs = ev.find_pairs(cat_dir)
        if args.limit:
            pairs = pairs[:args.limit]
        print(f"\n=== {category}: {len(pairs)} pair(s) ===")
        for occ_dir, gt_dir in pairs:
            t0 = time.time()
            offset = ev.get_scene_offset(ev.scene_key(category, occ_dir.name))
            try:
                scene = ev.load_scene(occ_dir)
                gt_raw = ev.load_gt_points(gt_dir)
                base = ev.compute_hits(scene, funcs, offset, mode)        # independent of the voxel size
            except Exception as e:
                n_err += 1
                print(f"  [error] {occ_dir.name}: {e}")
                continue
            line = []
            for v_cm in voxels:
                v = v_cm / 100.0
                pred = (ev.voxelize_hits(base["hit_points"], v, offset, mode)
                        if len(base["hit_points"]) else np.empty((0, 3)))
                gt = ev.filter_dense_voxels(gt_raw, v)
                st = scene_stats(pred, gt, v_cm)
                store.setdefault((category, v_cm), []).append(st)
                row = {"category": category, "scene": occ_dir.name, "voxel_cm": v_cm,
                       "pred_cells": st["n_pred"], "gt_cells": st["n_gt"]}
                for k in range(4):
                    p, r, f = prf_level(st, k)
                    row.update({f"P_{LEVEL_NAMES[k]}": p, f"R_{LEVEL_NAMES[k]}": r, f"F1_{LEVEL_NAMES[k]}": f})
                scene_rows.append(row)
                line.append(f"v={v_cm:g}: F1 exact {row['F1_exact']:.2f} -> 1-voxel {row['F1_exact+face+edge+vertex']:.2f}")
            print(f"  {occ_dir.name}: " + " | ".join(line) + f" | {time.time() - t0:.1f}s")

    if not store:
        raise SystemExit("No scenes evaluated.")

    cats = sorted({c for c, _ in store})
    main_cats = [c for c in cats if not is_sep(c)]
    sep_flags = {c: is_sep(c) for c in cats}
    sep_flags[all_name] = False

    csv_rows = []
    for v_cm in voxels:
        groups = {}
        if main_cats:
            groups[all_name] = summarize([s for c in main_cats for s in store.get((c, v_cm), [])])
        for c in cats:
            groups[c] = summarize(store.get((c, v_cm), []))
        order = list(groups.keys())
        pooled_name = all_name if all_name in groups else order[0]

        print("\n" + "=" * 112)
        print(f"voxel {v_cm:g} cm   (P / R / F1 in %, mean over scenes; 'miss' = 100 - P resp. 100 - R)")
        print(f"{'group':<34}{'scenes':>6}  " + "  ".join(f"{n:<26}" for n in LEVEL_NAMES))
        for g in order:
            s = groups[g]
            cells = "  ".join(f"{s[f'prf{k}'][0] * 100:5.1f} /{s[f'prf{k}'][1] * 100:5.1f} /{s[f'prf{k}'][2] * 100:5.1f}      "
                              for k in range(4))
            print(f"{g:<34}{s['n_scenes']:>6}  {cells}")
        print("\n  neighbourhood breakdown of the pooled cells [% exact / face / edge / vertex / far]")
        for g in order:
            s = groups[g]
            fmt = lambda c: " / ".join(f"{x:5.1f}" for x in (c / c.sum() * 100 if c.sum() else np.zeros(5)))
            print(f"  {g:<34} pred->GT {fmt(s['comp_pg'])}   GT->pred {fmt(s['comp_gp'])}   "
                  f"mean/max error pred->GT {s['err_pg'][0]:.1f}/{s['err_pg'][1]:.1f} cm, "
                  f"GT->pred {s['err_gp'][0]:.1f}/{s['err_gp'][1]:.1f} cm")

        for g in order:
            s = groups[g]
            for k in range(4):
                csv_rows.append({"group": g, "voxel_cm": v_cm, "level": LEVEL_NAMES[k], "scenes": s["n_scenes"],
                                 "precision": s[f"prf{k}"][0], "recall": s[f"prf{k}"][1], "f1": s[f"prf{k}"][2],
                                 **{f"pred_to_gt_{n}_%": (s["comp_pg"][i] / s["comp_pg"].sum() * 100 if s["comp_pg"].sum() else 0)
                                    for i, n in enumerate(("exact", "face", "edge", "vertex", "far"))},
                                 **{f"gt_to_pred_{n}_%": (s["comp_gp"][i] / s["comp_gp"].sum() * 100 if s["comp_gp"].sum() else 0)
                                    for i, n in enumerate(("exact", "face", "edge", "vertex", "far"))},
                                 "mean_err_pred_to_gt_cm": s["err_pg"][0], "max_err_pred_to_gt_cm": s["err_pg"][1],
                                 "mean_err_gt_to_pred_cm": s["err_gp"][0], "max_err_gt_to_pred_cm": s["err_gp"][1]})

        plot_exact(groups, order, sep_flags, v_cm, out / f"1_exact_match_{v_cm:g}cm.png", all_name)
        plot_levels(groups, order, sep_flags, v_cm, out / f"2_tolerance_levels_{v_cm:g}cm.png", pooled_name)

    for name, rows in (("tolerance_metrics.csv", csv_rows), ("per_scene_tolerance.csv", scene_rows)):
        with open(out / name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    print(f"\nDone ({n_err} scene error(s)). Results in: {out.resolve()}")


if __name__ == "__main__":
    main()