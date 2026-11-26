#!/usr/bin/env python3
"""STAGE 9 - the four box sources on the FINAL voxelised pcd completion, at every tolerance (paper figure + diagnostic parts).

Approaches (all evaluated by the same ray extension, the same voxel grid and the same ground truth):
    1  CV + learned face shifts (ours)    stage-6 model, out-of-fold choice
    2  PointNet OBB                       train_obb.py, trained with the same scene folds (s9a_pointnet_boxes.py)
    3  Traditional CV                     the camera-cluster box, shift 0
    4  Manual label                       your hand-drawn box, 3d_data_lidar.obb_3d (s9b_manual_boxes.py)
    +  reference "best shift (uses GT)"   best candidate of the shift grid for the graded score = what a perfect shift model could reach
For every box the LiDAR rays blocked by the occluder are extended to the box's front face (stage-4 rule), the regenerated points are voxelised on the
voxel grid of stage 4 (aligned with the CV box, so all methods share one grid) and every cell is classified by its nearest cell of the other set:
    precision side  regenerated cell -> nearest GT cell        recall side  hidden GT cell (S) -> nearest regenerated cell
    exact | face | edge | vertex | > 1 voxel,   cumulative levels  exact | +face | +face+edge | +face+edge+vertex  (= your occupancy evaluation)
P / R / F1 are means over boxes (95% CI by bootstrap over scenes). Extra numbers: distance of the cells to the other set in cm (mean, 95th percentile,
maximum per box -> Hausdorff distance), the stage-4 scores (exact / tol / graded), 3D IoU of every box with the manual label.
Only boxes that exist for ALL compared methods are used (same samples everywhere).

    python s9_compare_methods.py --work work_virt --pred work_virt/stage6/pred_cv_graded.npz
Outputs in <work>/stage9<tag>/:  figure_methods_compact.png / .pdf (the paper figure), parts/*.png (the same panels one by one, large),
    diag_per_box.png (box x method heat map), comparison_summary.csv, comparison_per_box.csv, FIGURE_CAPTION.txt
"""
import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import Gates, banner, progress, read_csv, region_mask, step, write_csv
from metrics import bootstrap_ci
from occupancy_tolerance_eval import CLASS_COLORS, CLASSES, cell_classes, composition, prf_level
from s8_tolerance_breakdown import cells, choices_for, pack
from shiftgrid import ShiftGrid, envelope_half
from visualize_scene import regen_points

LEVEL_TXT = ["Exact", "≤ 1 voxel\n(face)", "≤ 1 voxel\n(face + edge)", "≤ 1 voxel\n(face + edge + vertex)"]
LEVEL_SHORT = ["exact", "+face", "+face+edge", "+face+edge+vertex"]
CAP_CM = 30.0                                  # distance used when one of the two cell sets is empty

STYLE = {                                      # key -> (display name, tick label, colour, hatch)
    "ours": ("CV + learned face shifts (ours)", "CV + learned\nshifts (ours)", "#1f77b4", None),
    "pointnet": ("PointNet OBB", "PointNet\nOBB", "#ff7f0e", None),
    "cv": ("Traditional CV", "Traditional\nCV", "#8c8c8c", None),
    "manual": ("Manual label", "Manual\nlabel", "#2ca02c", None),
    "const": ("Best constant shift", "Best constant\nshift", "#9467bd", None),
    "ceiling": ("Best shift in grid (uses GT)", "Best shift\n(uses GT)", "#d9d9d9", "///"),
}


# ----------------------------------------------------------------------------------------------
def obb_iou(b1, b2, h=0.01):
    """3D IoU of two arbitrary oriented boxes (c, d, R): fraction of box 1 inside box 2 on a regular grid of spacing h (no yaw-only assumption)."""
    (c1, d1, R1), (c2, d2, R2) = b1, b2
    corners = c1 + (np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * d1 / 2) @ R1.T
    lo, hi = corners.min(0), corners.max(0)
    g = [np.arange(lo[k] + h / 2, hi[k], h) for k in range(3)]
    P = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    in1 = np.all(np.abs((P - c1) @ R1) <= d1 / 2, axis=1)
    if not in1.any():
        return 0.0
    P = P[in1]
    frac = float(np.mean(np.all(np.abs((P - c2) @ R2) <= d2 / 2, axis=1)))
    v1, v2 = float(np.prod(d1)), float(np.prod(d2))
    inter = v1 * frac
    return inter / (v1 + v2 - inter)


def graded_w(d_cm, vs_cm, sigma, radius):
    d2 = np.rint((np.asarray(d_cm) / vs_cm) ** 2)
    d2 = np.where(np.isfinite(d2), d2, 1e9)
    return np.where(d2 <= radius * radius + 1e-9, np.exp(-d2 / (2.0 * sigma ** 2)), 0.0)


def dist_stats(d):
    d = np.where(np.isfinite(d), d, CAP_CM)
    if len(d) == 0:
        return 0.0, 0.0, 0.0
    return float(d.mean()), float(np.percentile(d, 95)), float(d.max())


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--pred", required=True, help="stage-6 cross-validation file of OUR model (e.g. work/stage6/pred_cv_graded.npz)")
    ap.add_argument("--pointnet", default=None, help="default <work>/stage9/pointnet_boxes.npz (s9a_pointnet_boxes.py)")
    ap.add_argument("--manual", default=None, help="default <work>/stage9/manual_boxes.npz (s9b_manual_boxes.py)")
    ap.add_argument("--with-constant", action="store_true", help="add the 'best constant shift' baseline")
    ap.add_argument("--no-ceiling", action="store_true", help="leave out the grey 'best shift (uses GT)' reference")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--tag", default="", help="suffix of the output folder stage9<tag>")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N boxes")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--voxel-size", type=float, default=None,
                    help="RE-SCORE the same boxes at another voxel size [m] (sensitivity check, e.g. 0.02 or 0.05). No stage-4 rerun and no retraining: the boxes of every method "
                         "stay as they are. The stage-4 tables are tied to the original voxel size, so the grey best-shift reference, the constant baseline and the table parity "
                         "gate are switched off, and the output goes to stage9<tag>_vox<N>cm")
    a = ap.parse_args()

    banner(9, "four box sources on the final voxelised completion")
    gates = Gates(9)
    work = Path(a.work)
    cfg = json.loads((work / "stage4" / "config.json").read_text())
    grid = ShiftGrid(cfg["n_steps"], cfg["n_steps_z"], cfg["step"], cfg["min_dim"])
    vs, margin, frame = cfg["voxel_size"], cfg["block_margin"], cfg["voxel_frame"]
    rescored = a.voxel_size is not None and abs(a.voxel_size - vs) > 1e-9
    if rescored:
        print(f"  [note] re-scoring at voxel {a.voxel_size * 100:g} cm (stage-4 tables are for {vs * 100:g} cm): reference / constant / parity switched off")
        vs, a.no_ceiling, a.with_constant = a.voxel_size, True, False
        a.tag = f"{a.tag}_vox{a.voxel_size * 100:g}cm"
    gsig, grad = cfg.get("graded", {}).get("sigma", 1.0), cfg.get("graded", {}).get("radius", 2)
    meta, tabs, ch, fold = choices_for(a.work, "cv", a.pred, grid)
    boxes = np.load(work / "stage1" / "boxes.npz")
    bid = {s: i for i, s in enumerate(boxes["ids"])}
    s0 = {r["sample_id"]: r for r in read_csv(work / "stage0" / "index.csv")}
    out = work / f"stage9{a.tag}"
    (out / "parts").mkdir(parents=True, exist_ok=True)

    # ---- which boxes does every method have? ------------------------------------------------------------------------
    step("1/4 collecting the boxes of every method")
    ext = {}
    for key, path, default in (("pointnet", a.pointnet, "pointnet_boxes.npz"), ("manual", a.manual, "manual_boxes.npz")):
        p = Path(path) if path else work / "stage9" / default
        if not p.exists():
            raise SystemExit(f"{p} not found - run {'s9a_pointnet_boxes.py' if key == 'pointnet' else 's9b_manual_boxes.py'} first")
        z = np.load(p)
        ext[key] = {s: (z["center"][i], z["dims"][i], z["R"][i]) for i, s in enumerate(z["ids"])}
        if key == "pointnet" and "dry_run" in z.files and bool(z["dry_run"]):
            print("  [warn] pointnet_boxes.npz is a DRY RUN (classical prior box, no training): the PointNet column is NOT the trained model")
    shift_methods = {"ours": "model", "cv": "CV box (shift 0)"}
    if a.with_constant:
        shift_methods["const"] = "best constant"
    ceil_name = "oracle (graded score)" if "oracle (graded score)" in ch else "oracle (tol score)"
    if not a.no_ceiling:
        shift_methods["ceiling"] = ceil_name
    for k, v in shift_methods.items():
        if v not in ch:
            raise SystemExit(f"'{v}' is not available (is {a.pred} complete?)")
    order = [k for k in ("ours", "pointnet", "cv", "manual", "const", "ceiling") if k in shift_methods or k in ext]
    sel = [i for i, m in enumerate(meta) if fold[i] >= 0 and m["kind"] == "orig" and m["sample_id"] in bid
           and all(m["sample_id"] in ext[k] for k in ext)]
    n_real = sum(1 for i, m in enumerate(meta) if fold[i] >= 0 and m["kind"] == "orig" and m["sample_id"] in bid)
    for k in ext:
        print(f"     {STYLE[k][0]:<34} boxes for {sum(1 for i in range(len(meta)) if meta[i]['sample_id'] in ext[k] and meta[i]['kind'] == 'orig')}/{n_real} real samples")
    if a.limit:
        sel = sel[:a.limit]
    print(f"     -> {len(sel)} boxes in {len({meta[i]['scene'] for i in sel})} scenes are available for ALL methods and used below")
    gates.check("enough common boxes", len(sel) >= 3, f"{len(sel)} boxes")

    # ---- evaluation ------------------------------------------------------------------------------------------------
    step(f"2/4 regenerating + scoring {len(sel)} boxes x {len(order)} methods (voxel {vs * 100:.1f} cm, {frame}-aligned)")
    stats = {k: [] for k in order}
    rows, parity, reach = [], [], {k: 0 for k in ext}
    for i in progress(sel, total=len(sel), desc="boxes"):
        sid = meta[i]["sample_id"]
        b = bid[sid]
        cv = (boxes["center"][b], boxes["dims"][b], boxes["R"][b])
        c0, d0, R0 = cv
        z = np.load(s0[sid]["path"])
        origin, scan = z["origin"], z["scan"].astype(np.float64)
        gt, shadow = z["gt"].astype(np.float64), z["gt_shadow"]
        H = envelope_half(d0, grid, vs)
        to_frame = (lambda p: (p - c0) @ R0) if frame == "box" else (lambda p: p)
        G = cells(gt[region_mask(gt, cv, H)], to_frame, vs)
        S = cells(gt[shadow & region_mask(gt, cv, H)], to_frame, vs)
        gmin, gmax = np.floor(-H / vs).astype(np.int64) - 1, np.floor(H / vs).astype(np.int64) + 1
        Gset = pack(G)
        boxes_i = {}
        for k in order:
            if k in shift_methods:
                boxes_i[k] = grid.box_from(cv, int(ch[shift_methods[k]][i]))
            else:
                boxes_i[k] = ext[k][sid]
        for k in ext:                                                       # does the box leave the region of the shift grid (+- reach around the CV box)?
            cb, db, Rb = boxes_i[k]
            cn = (cb + (np.array([[x, y, z_] for x in (-1, 1) for y in (-1, 1) for z_ in (-1, 1)]) * db / 2) @ Rb.T - c0) @ R0
            if np.any(cn.min(0) < -(d0 / 2 + grid.max_offset)) or np.any(cn.max(0) > d0 / 2 + grid.max_offset):
                reach[k] += 1
        man = boxes_i.get("manual", ext["manual"][sid])
        for k in order:
            P = cells(regen_points(origin, scan, boxes_i[k], margin), to_frame, vs)
            if frame == "box" and len(P):
                P = P[np.all((P >= gmin) & (P <= gmax), axis=1)]            # same dense grid as the stage-4 evaluator
            cls_pg, d_pg = cell_classes(P, G, vs * 100)
            cls_sp, d_sp = cell_classes(S, P, vs * 100)
            nS, nP = len(S), len(P)
            ex = float((cls_sp == 0).sum() / (nS + (cls_pg != 0).sum())) if nS else 0.0
            tl = float((cls_sp <= 3).sum() / (nS + (cls_pg == 4).sum())) if nS else 0.0
            gr = float(graded_w(d_sp, vs * 100, gsig, grad).sum() / (nS + (1 - graded_w(d_pg, vs * 100, gsig, grad)).sum())) if nS else 0.0
            if k in shift_methods and not rescored:
                f = int(ch[shift_methods[k]][i])
                for nm, v in (("exact", ex), ("tol", tl), ("graded", gr)):
                    if nm in tabs:
                        parity.append(abs(v - float(np.asarray(tabs[nm][i], np.float32)[f])))
            st = {"cls_pg": cls_pg, "cls_gp": cls_sp, "d_pg": d_pg, "d_gp": d_sp, "scene": meta[i]["scene"], "sample_id": sid,
                  "exact": ex, "tol": tl, "graded": gr, "nP": nP, "nS": nS}
            mp, p95p, mxp = dist_stats(d_pg)
            mg, p95g, mxg = dist_stats(d_sp)
            st.update(chamfer=(mp + mg) / 2, hausdorff=max(mxp, mxg), p95=max(p95p, p95g), max_pg=mxp, max_gp=mxg, mean_pg=mp, mean_gp=mg)
            st["iou_manual"] = 1.0 if k == "manual" else obb_iou(boxes_i[k], man)
            st["centre_err_manual_cm"] = float(np.linalg.norm(boxes_i[k][0] - man[0]) * 100)
            stats[k].append(st)
            row = {"sample_id": sid, "scene": meta[i]["scene"], "category": meta[i]["category"], "method": STYLE[k][0], "n_regenerated": nP, "n_hidden_gt": nS,
                   "score_exact": ex, "score_tol": tl, "score_graded": gr, "chamfer_cm": st["chamfer"], "hausdorff_cm": st["hausdorff"],
                   "max_regen_to_gt_cm": mxp, "max_hidden_gt_to_regen_cm": mxg, "iou_with_manual": st["iou_manual"], "centre_err_to_manual_cm": st["centre_err_manual_cm"]}
            for lv in range(4):
                p, r, f1 = prf_level(st, lv)
                row.update({f"P_{LEVEL_SHORT[lv]}": p, f"R_{LEVEL_SHORT[lv]}": r, f"F1_{LEVEL_SHORT[lv]}": f1})
            rows.append(row)

    # ---- aggregation ---------------------------------------------------------------------------------------------------
    step("3/4 aggregating")
    scenes = np.array([s["scene"] for s in stats[order[0]]])
    n = len(scenes)
    agg = {}
    for k in order:
        st = stats[k]
        f1 = np.array([[prf_level(s, lv)[2] for s in st] for lv in range(4)])           # (4, n)
        agg[k] = {"prf": np.array([np.mean([prf_level(s, lv) for s in st], axis=0) for lv in range(4)]), "f1": f1,
                  "ci": [bootstrap_ci(f1[lv], scenes, a.bootstrap) for lv in range(4)],
                  "pg": composition(st, "pg"), "gp": composition(st, "gp"),
                  **{m: np.array([s[m] for s in st]) for m in ("exact", "tol", "graded", "chamfer", "hausdorff", "p95", "max_pg", "max_gp", "iou_manual", "centre_err_manual_cm")}}

    name = lambda k: STYLE[k][0]
    print(f"\n  F1 (%) per tolerance level, mean over {n} boxes [95% CI over scenes]")
    print(f"  {'method':<36}" + "".join(f"{t:>24}" for t in LEVEL_SHORT))
    for k in order:
        print(f"  {name(k):<36}" + "".join(f"{agg[k]['prf'][lv][2] * 100:>10.1f} [{agg[k]['ci'][lv][1] * 100:4.1f},{agg[k]['ci'][lv][2] * 100:4.1f}]  " for lv in range(4)))
    print(f"\n  precision / recall (%) at 'exact' and at '+face+edge+vertex'")
    for k in order:
        print(f"  {name(k):<36} exact P {agg[k]['prf'][0][0] * 100:5.1f}  R {agg[k]['prf'][0][1] * 100:5.1f}   | <=1 voxel P {agg[k]['prf'][3][0] * 100:5.1f}  R {agg[k]['prf'][3][1] * 100:5.1f}")
    print(f"\n  cell distance to the other set (cm): chamfer = mean of both directions, Hausdorff = max over both directions, per box")
    print(f"  {'method':<36}{'chamfer mean':>13}{'Hausd. median':>14}{'Hausd. mean':>12}{'Hausd. MAX':>11}{'IoU w/ label':>14}{'centre err cm':>14}")
    for k in order:
        g = agg[k]
        print(f"  {name(k):<36}{g['chamfer'].mean():>13.2f}{np.median(g['hausdorff']):>14.1f}{g['hausdorff'].mean():>12.1f}{g['hausdorff'].max():>11.1f}"
              f"{g['iou_manual'].mean():>14.3f}{g['centre_err_manual_cm'].mean():>14.1f}")
    print(f"\n  stage-4 scores (mean over boxes; 1 = perfect):  exact | tol | graded")
    for k in order:
        print(f"  {name(k):<36} {agg[k]['exact'].mean():.3f} | {agg[k]['tol'].mean():.3f} | {agg[k]['graded'].mean():.3f}")
    ref = "cv"
    print(f"\n  paired F1 difference to '{name(ref)}' at +face+edge+vertex and exact (percentage points, [95% CI over scenes]); negative = worse than the CV box")
    for k in order:
        if k == ref:
            continue
        txt = []
        for lv in (0, 3):
            mu, lo, hi = bootstrap_ci(agg[k]["f1"][lv] - agg[ref]["f1"][lv], scenes, a.bootstrap)
            txt.append(f"{mu * 100:+6.1f} [{lo * 100:+5.1f},{hi * 100:+5.1f}]")
        print(f"  {name(k):<36} exact {txt[0]}   |  <=1 voxel {txt[1]}")
    if "ours" in order and "pointnet" in order:
        for lv in (0, 3):
            mu, lo, hi = bootstrap_ci(agg["ours"]["f1"][lv] - agg["pointnet"]["f1"][lv], scenes, a.bootstrap)
            print(f"  ours - PointNet, F1 {LEVEL_SHORT[lv]:<18} {mu * 100:+6.1f} [{lo * 100:+5.1f},{hi * 100:+5.1f}]")
    print(f"\n  worst boxes per method (F1 at +face+edge+vertex):")
    for k in order:
        w = np.argsort(agg[k]["f1"][3])[:3]
        print(f"  {name(k):<36} " + "  ".join(f"{stats[k][j]['sample_id']} {agg[k]['f1'][3][j] * 100:.0f}% (Hausd. {agg[k]['hausdorff'][j]:.0f} cm)" for j in w))
    for k in ext:
        if reach[k]:
            print(f"  [note] {name(k)}: {reach[k]}/{n} boxes extend beyond the reach of the shift grid around the CV box (+-{grid.max_offset[0] * 100:.0f}/{grid.max_offset[2] * 100:.0f} cm); "
                  f"their regenerated cells outside the evaluation region are not counted")

    # ---- figures -------------------------------------------------------------------------------------------------------
    step("4/4 figures")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
    col = lambda k: STYLE[k][2]
    tick = lambda k: STYLE[k][1]

    def bars_style(rect, k):
        if STYLE[k][3]:
            rect.set_hatch(STYLE[k][3]); rect.set_edgecolor("#9a9a9a"); rect.set_linewidth(.6)

    def p_f1(ax, small=False):
        w = 0.82 / len(order)
        for j, k in enumerate(order):
            xs = np.arange(4) + (j - (len(order) - 1) / 2) * w
            v = np.array([agg[k]["prf"][lv][2] for lv in range(4)]) * 100
            lo = v - np.array([agg[k]["ci"][lv][1] for lv in range(4)]) * 100
            hi = np.array([agg[k]["ci"][lv][2] for lv in range(4)]) * 100 - v
            bars = ax.bar(xs, v, w, color=col(k), label=name(k), yerr=np.vstack([np.nan_to_num(lo), np.nan_to_num(hi)]),
                          error_kw=dict(lw=.8, capsize=1.5, ecolor="#333"))
            for r in bars:
                bars_style(r, k)
            for x_, v_ in zip(xs, v):
                ax.text(x_, 1.5, f"{v_:.0f}", ha="center", va="bottom", fontsize=6.5 if small else 7.5, color="white" if STYLE[k][3] is None else "#333", fontweight="bold")
        ax.set_xticks(range(4)); ax.set_xticklabels(LEVEL_TXT, fontsize=8)
        ax.set_ylim(0, 124); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel("F1 of the completed voxels (%)")
        ax.set_title("F1 by tolerance level")
        ax.grid(axis="y", alpha=.3)
        ax.legend(fontsize=7, loc="upper center", frameon=False, ncol=3, columnspacing=1.0, handlelength=1.4)

    def p_stack(ax, side, title):
        fr = np.array([agg[k][side] / max(agg[k][side].sum(), 1) * 100 for k in order])
        bottom = np.zeros(len(order))
        xs = np.arange(len(order))
        for c in range(5):
            bars = ax.bar(xs, fr[:, c], 0.65, bottom=bottom, color=CLASS_COLORS[c], edgecolor="white", lw=.6)
            for xi, (bt, h) in enumerate(zip(bottom, fr[:, c])):
                if h >= 5:
                    ax.text(xi, bt + h / 2, f"{h:.0f}", ha="center", va="center", fontsize=7.5)
            bottom += fr[:, c]
        ax.set_xticks(xs); ax.set_xticklabels([tick(k) for k in order], fontsize=7.5)
        ax.set_ylim(0, 100); ax.set_ylabel("% of cells"); ax.set_title(title)

    def p_box(ax, vals, title, ylabel, kfmt="{:.0f}", ylim=None, hlines=(), annotate="max"):
        data = [vals[k] for k in order]
        bp = ax.boxplot(data, positions=range(len(order)), widths=.55, showfliers=False, patch_artist=True, medianprops=dict(color="black", lw=1.2),
                        whiskerprops=dict(lw=.8), capprops=dict(lw=.8), boxprops=dict(lw=.8))
        rng = np.random.default_rng(0)
        for j, k in enumerate(order):
            bp["boxes"][j].set_facecolor(col(k)); bp["boxes"][j].set_alpha(.45)
            ax.scatter(j + rng.uniform(-.17, .17, len(data[j])), data[j], s=9, color=col(k), edgecolor="#222", lw=.4, zorder=3)
            ax.scatter([j], [data[j].mean()], marker="D", s=26, color="white", edgecolor="black", zorder=4)
            ref_v = data[j].max() if annotate == "max" else data[j].min()
            ax.annotate((("max " if annotate == "max" else "min ") + kfmt).format(ref_v), (j, ref_v), textcoords="offset points",
                        xytext=(0, 4 if annotate == "max" else -11), ha="center", fontsize=7)
        for hl, lab in hlines:
            ax.axhline(hl, color="#888", lw=.7, ls="--")
            ax.text(len(order) - .45, hl, lab, fontsize=6.5, va="bottom", ha="right", color="#666")
        ax.set_xticks(range(len(order))); ax.set_xticklabels([tick(k) for k in order], fontsize=7.5)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.3)
        if ylim:
            ax.set_ylim(*ylim)

    def p_table(ax):
        ax.axis("off")
        head = ["method", "F1\nexact", "F1\n≤1 voxel", "chamfer\n(cm)", "Hausdorff\nmed / max\n(cm)", "IoU with\nlabel"]
        cell = []
        for k in order:
            g = agg[k]
            cell.append([name(k).replace(" (uses GT)", "\n(uses GT)").replace(" (ours)", "\n(ours)"), f"{g['prf'][0][2] * 100:.0f}", f"{g['prf'][3][2] * 100:.0f}", f"{g['chamfer'].mean():.1f}",
                         f"{np.median(g['hausdorff']):.0f} / {g['hausdorff'].max():.0f}", "—" if k == "manual" else f"{g['iou_manual'].mean():.2f}"])
        t = ax.table(cellText=cell, colLabels=head, cellLoc="center", colWidths=[.36, .11, .12, .11, .17, .13], bbox=[0, 0.05, 1, 0.80])
        t.auto_set_font_size(False); t.set_fontsize(7)
        for (r, c), cl in t.get_celld().items():
            cl.set_linewidth(.4)
            if r == 0:
                cl.set_facecolor("#eeeeee"); cl.set_text_props(fontweight="bold")
            elif c == 0:
                cl.set_text_props(ha="left"); cl._loc = "left"
                cl.set_facecolor(matplotlib.colors.to_rgba(col(order[r - 1]), .25))
        ax.set_title("Summary (mean over boxes)", loc="left")

    leg = [Patch(facecolor=CLASS_COLORS[c], label=CLASSES[c]) for c in range(5)]
    f1_vs = lambda lv: {k: agg[k]["f1"][lv] * 100 for k in order}
    haus = {k: agg[k]["hausdorff"] for k in order}
    f1_lim = lambda lv: (max(-3, min(float(agg[k]["f1"][lv].min()) for k in order) * 100 - 14), 104)
    vcm = vs * 100
    haus_lines = [(vcm, "1 voxel"), (2 * vcm, "2 voxels")]

    fig = plt.figure(figsize=(16, 9.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, width_ratios=[1.35, 1, 1], height_ratios=[1, 1, 0.07])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
           fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    p_f1(axs[0], small=True)
    p_stack(axs[1], "pg", "Precision side: regenerated cell → GT")
    p_stack(axs[2], "gp", "Recall side: hidden GT cell → regenerated")
    p_box(axs[3], f1_vs(3), "Per-box F1 (≤ 1 voxel, incl. vertex)", "F1 (%)", annotate="min", ylim=f1_lim(3))
    p_box(axs[4], haus, "Worst error per box (Hausdorff distance)", "cm", ylim=(0, max(15, float(max(h.max() for h in haus.values())) * 1.15)), hlines=haus_lines)
    p_table(axs[5])
    for ax, ch_ in zip(axs, "abcdef"):
        ax.text(-0.02, 1.07, f"({ch_})", transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")
    axl = fig.add_subplot(gs[2, 1:])
    axl.axis("off")
    axl.legend(handles=leg, loc="center", ncol=5, fontsize=8, frameon=False, title="class of the nearest cell of the other set (panels b, c)", title_fontsize=8)
    fig.savefig(out / "figure_methods_compact.png", dpi=a.dpi, bbox_inches="tight")
    fig.savefig(out / "figure_methods_compact.pdf", bbox_inches="tight")
    plt.close(fig)

    def single(fn, fname, size=(9, 5.6), legend_classes=False):
        fig, ax = plt.subplots(figsize=size, constrained_layout=True)
        fn(ax)
        if legend_classes:
            ax.legend(handles=leg, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
        fig.savefig(out / "parts" / fname, dpi=200)
        plt.close(fig)

    single(lambda ax: p_f1(ax), "a_f1_by_tolerance.png", (11, 6))
    single(lambda ax: p_stack(ax, "pg", "Precision side: regenerated cell → nearest GT cell"), "b_precision_side.png", (8.5, 6), True)
    single(lambda ax: p_stack(ax, "gp", "Recall side: hidden GT cell → nearest regenerated cell"), "c_recall_side.png", (8.5, 6), True)
    single(lambda ax: p_box(ax, f1_vs(3), "Per-box F1 (≤ 1 voxel, incl. vertex)", "F1 (%)", annotate="min", ylim=f1_lim(3)), "d_per_box_f1_1voxel.png", (8.5, 5.6))
    single(lambda ax: p_box(ax, f1_vs(0), "Per-box F1 (exact cell match)", "F1 (%)", annotate="min", ylim=f1_lim(0)), "d2_per_box_f1_exact.png", (8.5, 5.6))
    single(lambda ax: p_box(ax, haus, "Worst error per box (Hausdorff distance)", "cm", ylim=(0, max(15, float(max(h.max() for h in haus.values())) * 1.15)), hlines=haus_lines),
           "e_worst_error_per_box.png", (8.5, 5.6))
    single(p_table, "f_summary_table.png", (10, 3.2))
    fig, axs3 = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)          # P / R / F1 per tolerance level
    for ax, (mi, nm) in zip(axs3, enumerate(("Precision", "Recall", "F1"))):
        w = 0.82 / len(order)
        for j, k in enumerate(order):
            bars = ax.bar(np.arange(4) + (j - (len(order) - 1) / 2) * w, [agg[k]["prf"][lv][mi] * 100 for lv in range(4)], w, color=col(k), label=name(k))
            for r in bars:
                bars_style(r, k)
        ax.set_xticks(range(4)); ax.set_xticklabels(LEVEL_TXT, fontsize=8); ax.set_ylim(0, 105); ax.set_title(nm + " by tolerance level"); ax.grid(axis="y", alpha=.3)
    axs3[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.savefig(out / "parts" / "a2_precision_recall_f1_by_tolerance.png", dpi=170)
    plt.close(fig)

    # box x method heat maps (what is going on in a bad box?)
    rank = np.argsort(agg["cv"]["f1"][3]) if "cv" in agg else np.arange(n)
    fig, axs2 = plt.subplots(1, 3, figsize=(5 + 3.2 * len(order), 0.34 * n + 2.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1, 1]))
    for ax, (mat, ttl) in zip(axs2, ((lambda k: agg[k]["f1"][3] * 100, "F1, ≤ 1 voxel (%)"), (lambda k: agg[k]["f1"][0] * 100, "F1, exact (%)"),
                                      (lambda k: agg[k]["hausdorff"], "Hausdorff distance (cm)"))):
        M = np.array([mat(k)[rank] for k in order]).T
        im = ax.imshow(M, aspect="auto", cmap="RdYlGn" if "F1" in ttl else "RdYlGn_r", vmin=(40 if "1 voxel" in ttl else 0), vmax=100 if "F1" in ttl else max(10, np.percentile(M, 95)))
        ax.set_xticks(range(len(order))); ax.set_xticklabels([tick(k) for k in order], fontsize=7, rotation=0)
        ax.set_yticks(range(n))
        ax.set_yticklabels([stats[order[0]][j]["sample_id"].replace("_orig000", "") for j in rank] if ax is axs2[0] else [], fontsize=6.5)
        for r_ in range(M.shape[0]):
            for c_ in range(M.shape[1]):
                ax.text(c_, r_, f"{M[r_, c_]:.0f}", ha="center", va="center", fontsize=6.5)
        ax.set_title(ttl, fontsize=9)
    fig.suptitle("Per box and method (rows sorted by the traditional-CV box, worst first)", fontsize=10)
    fig.savefig(out / "diag_per_box.png", dpi=170)
    plt.close(fig)

    write_csv(out / "comparison_per_box.csv", rows)
    write_csv(out / "comparison_summary.csv", [
        {"method": name(k), "n_boxes": n, **{f"F1_{LEVEL_SHORT[lv]}": agg[k]["prf"][lv][2] for lv in range(4)},
         **{f"F1_{LEVEL_SHORT[lv]}_ci_lo": agg[k]["ci"][lv][1] for lv in range(4)}, **{f"F1_{LEVEL_SHORT[lv]}_ci_hi": agg[k]["ci"][lv][2] for lv in range(4)},
         **{f"P_{LEVEL_SHORT[lv]}": agg[k]["prf"][lv][0] for lv in range(4)}, **{f"R_{LEVEL_SHORT[lv]}": agg[k]["prf"][lv][1] for lv in range(4)},
         "score_exact": agg[k]["exact"].mean(), "score_tol": agg[k]["tol"].mean(), "score_graded": agg[k]["graded"].mean(),
         "chamfer_cm": agg[k]["chamfer"].mean(), "hausdorff_median_cm": float(np.median(agg[k]["hausdorff"])), "hausdorff_max_cm": float(agg[k]["hausdorff"].max()),
         "iou_with_manual": agg[k]["iou_manual"].mean(), **{f"pg_{c}": agg[k]["pg"][j] / max(agg[k]["pg"].sum(), 1) * 100 for j, c in enumerate(("exact", "face", "edge", "vertex", "far"))},
         **{f"gp_{c}": agg[k]["gp"][j] / max(agg[k]["gp"].sum(), 1) * 100 for j, c in enumerate(("exact", "face", "edge", "vertex", "far"))}} for k in order])
    cap = (f"Figure. Completion of LiDAR points hidden by an occluder, using four different boxes of the target ({n} boxes, {len(set(scenes))} scenes, {vs * 100:g} cm voxels): "
           f"the box fitted to the camera cluster by traditional CV, the same box corrected by learned face shifts (ours; out-of-fold), a PointNet that regresses the box from the "
           f"cluster (out-of-fold), and the manually labelled box. Blocked LiDAR rays are extended to the front face of each box and compared with the ground truth scan taken "
           f"without the occluder. (a) F1 of the voxelised completion when a hit must be an exact voxel or may be a neighbouring voxel (face; face and edge; face, edge and vertex), "
           f"mean over boxes with 95% CI over scenes. (b, c) Where the regenerated cells find their nearest ground-truth cell (precision side) and where the hidden ground-truth cells "
           f"find their nearest regenerated cell (recall side). (d) Per-box F1 at 1-voxel tolerance; diamonds = mean. (e) Largest distance between a cell and its nearest cell of the other set "
           f"(Hausdorff distance) per box. (f) Summary." + ("" if a.no_ceiling else " The grey bars are the best candidate of the shift grid chosen with the ground truth (an upper reference, not a method)."))
    (out / "FIGURE_CAPTION.txt").write_text(cap + "\n")

    # ---- gates -----------------------------------------------------------------------------------------------------
    parity = np.array(parity)
    if len(parity):
        gates.check("scores recomputed here equal the stage-4 tables (exact / tol / graded of the shift-based methods)", parity.mean() < 0.01 and np.percentile(parity, 95) < 0.03,
                    f"mean |diff| {parity.mean():.4f}, p95 {np.percentile(parity, 95):.4f}, max {parity.max():.3f}")
    mono = all(agg[k]["prf"][lv][2] <= agg[k]["prf"][lv + 1][2] + 1e-9 for k in order for lv in range(3))
    gates.check("F1 does not decrease with a more lenient tolerance", mono, "levels are nested")
    if "ceiling" in order and "cv" in order:
        gates.check("best-shift reference is not worse than the CV box (graded score)", agg["ceiling"]["graded"].mean() >= agg["cv"]["graded"].mean() - 1e-6,
                    f"{agg['ceiling']['graded'].mean():.3f} vs {agg['cv']['graded'].mean():.3f}")
    print(f"\n  files in {out}:  figure_methods_compact.png/.pdf, parts/, diag_per_box.png, comparison_summary.csv, comparison_per_box.csv, FIGURE_CAPTION.txt")
    gates.finish()


if __name__ == "__main__":
    main()