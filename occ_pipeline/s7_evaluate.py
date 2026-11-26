#!/usr/bin/env python3
"""STAGE 7 - regret and baselines (numpy only; reads the model's choices from stage 6 if they exist).

For every validation sample the stage-4 table contains the exact outcome of ALL candidates, so no pipeline has to be
re-run here:
    regret = best score in the table - score of the chosen candidate          (0 = the model found an optimal shift)
Compared choices (all use the same tables):
    oracle          argmax of the table (ties -> smallest shift)                 regret 0 by definition
    CV box          shift 0 = the unmodified traditional-CV box
    best constant   ONE shift for all samples, picked on the TRAIN tables        <- if the model cannot beat this, it has learned nothing
    random          expected score of a uniformly random valid candidate
    model           argmax of the predicted distribution (work/stage6/pred_val.npz)
Reported for the tolerance score, the exact hit/miss score and (if stage 4 computed it) the distance-weighted 'graded' score, on the real un-augmented validation samples
('orig', the honest number) and on all validation samples (augmented copies of the same val scenes), with a 95% bootstrap
CI over scenes. Also: how often the choice is optimal (within 0.001), and the distance of the chosen face shifts to the
nearest optimal candidate [cm].
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import Gates, banner, read_csv, step, write_csv
from metrics import best_constant, bootstrap_ci, clean_scores, sample_metrics
from shiftgrid import ShiftGrid


def load(work, split):
    d = Path(work) / "stage5"
    if split == "cv":                                   # train + val pooled (scene-level cross-validation, see stage 6 --cv)
        parts = [load(work, s_) for s_ in ("train", "val") if (d / f"{s_}_meta.csv").exists()]
        keys = [w for w in ("tol", "exact", "graded") if all(w in p[1] for p in parts)]
        return sum((p[0] for p in parts), []), {w: np.concatenate([np.asarray(p[1][w]) for p in parts]) for w in keys}
    meta = read_csv(d / f"{split}_meta.csv")
    tabs = {"tol": np.load(d / f"{split}_tab_tol.npy", mmap_mode="r"), "exact": np.load(d / f"{split}_tab_exact.npy", mmap_mode="r")}
    if (d / f"{split}_tab_graded.npy").exists():
        tabs["graded"] = np.load(d / f"{split}_tab_graded.npy", mmap_mode="r")
    return meta, tabs


def feasible_constant(tab, mu):
    """The 'one shift for all' baseline, made fair: for every sample take the candidate with the best TRAIN mean score among those
    that are valid for THAT sample (the model cannot choose an impossible, too thin box either)."""
    out = np.zeros(len(tab), np.int64)
    for i in range(len(tab)):
        ok = ~np.isnan(np.asarray(tab[i], np.float32))
        out[i] = int(np.argmax(np.where(ok, mu, -np.inf)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--split", default="val", help="val | train | cv (all scenes pooled; needs work/stage6/pred_cv.npz from  s6_train.py --cv K)")
    ap.add_argument("--pred", default=None, help="npz with 'idx' and 'ids' (default work/stage6/pred_<split>.npz)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--tag", default="", help="suffix used by  s6_train.py --tag  (reads pred_<split><tag>.npz, writes stage7_per_sample<tag>.csv)")
    a = ap.parse_args()

    banner(7, f"regret evaluation on the '{a.split}' split")
    gates = Gates(7)
    cfg = json.loads((Path(a.work) / "stage4" / "config.json").read_text())
    grid = ShiftGrid(cfg["n_steps"], cfg["n_steps_z"], cfg["step"], cfg["min_dim"])
    zero, l1, total = grid.zero_index(), grid.l1_all(), grid.total
    meta, tabs = load(a.work, a.split)
    n = len(meta)
    step(f"{n} samples ({sum(m['kind'] == 'orig' for m in meta)} real) | {grid.describe()}")

    # ---- the choices that are compared --------------------------------------------------------------------------
    choices = {"CV box (shift 0)": np.full(n, zero)}
    pred_path = Path(a.pred) if a.pred else Path(a.work) / "stage6" / f"pred_{a.split}{a.tag}.npz"
    zp = np.load(pred_path) if pred_path.exists() else None
    cv = a.split == "cv"
    if cv:
        if zp is None or "fold" not in zp.files:
            raise SystemExit(f"--split cv needs {pred_path}: run  python s6_train.py --work {a.work} --cv 5  first")
        fold_of = dict(zip(zp["ids"], zp["fold"]))
        fold = np.array([fold_of[m["sample_id"]] for m in meta])
        use_virtual = bool(zp["use_virtual"]) if "use_virtual" in zp.files else True           # --no-virtual runs: the constant must not see them either
        if not use_virtual:
            print("  [info] model trained with --no-virtual: virtual-occluder samples are also left out of the best-constant fit")
        const_idx, const_raw = {}, {}
        for w in tabs:
            ci, cu = np.zeros(n, np.int64), np.zeros(n, np.int64)
            for f in np.unique(fold[fold >= 0]):
                top, mu = best_constant(tabs[w][(fold != f) & ((fold >= 0) | use_virtual)], return_mean=True)    # fitted on the same training data as the model (incl. virtual samples), WITHOUT the held-out scenes
                cu[fold == f] = top
                ci[fold == f] = feasible_constant(tabs[w][fold == f], mu)
            const_idx[w], const_raw[w] = ci, cu
        cname = "best constant (out-of-fold)"
        step("best constant shift per fold, tol [cm] lo_x hi_x lo_y hi_y lo_z hi_z: " + " | ".join(
            f"f{f}: {np.round(grid.decode(const_raw['tol'][fold == f][0]) * 100, 0).astype(int).tolist()}" for f in np.unique(fold[fold >= 0])))
    else:
        trmeta, trt = load(a.work, "train")
        if zp is not None and "use_virtual" in zp.files and not bool(zp["use_virtual"]):
            keep = np.array([not m["category"].lower().startswith("virtual") for m in trmeta])
            trt = {w: np.asarray(trt[w])[keep] for w in trt}
            print("  [info] model trained with --no-virtual: virtual-occluder samples are also left out of the best-constant fit")
        bc = {w: best_constant(trt[w], return_mean=True) for w in tabs}
        const = {w: bc[w][0] for w in bc}
        const_idx = {w: feasible_constant(tabs[w], bc[w][1]) for w in bc}
        cname = "best constant (train)"
        sh = {w: grid.decode(const[w]) * 100 for w in const}
        step("best constant shift from the train tables [cm]  lo_x hi_x lo_y hi_y lo_z hi_z:  "
             f"tol {np.round(sh['tol'], 1)} | exact {np.round(sh['exact'], 1)}")
    choices[cname] = None                                            # filled per score kind below
    model_idx = None
    if zp is not None:
        order = {s_: i for i, s_ in enumerate(zp["ids"])}
        if all(m["sample_id"] in order for m in meta):
            model_idx = np.array([zp["idx"][order[m["sample_id"]]] for m in meta])
            choices["model"] = model_idx
        else:
            print(f"  [warn] {pred_path} does not cover all '{a.split}' samples (was stage 6 run with --smoke?) - model skipped")
    else:
        print(f"  [info] no predictions at {pred_path}: only baselines are evaluated (run stage 6 first)")

    if cv and (fold < 0).any():                                      # virtual-occluder samples were training data only: evaluate on the real scenes
        sel = np.flatnonzero(fold >= 0)
        print(f"  [info] {int((fold < 0).sum())} virtual-occluder training samples left out of the evaluation; {len(sel)} real-scene samples evaluated")
        meta = [meta[i] for i in sel]
        tabs = {w: tabs[w][sel] for w in tabs}
        n = len(meta)
        const_idx = {w: v[sel] for w, v in const_idx.items()}
        choices = {k_: (None if v is None else np.asarray(v)[sel]) for k_, v in choices.items()}
    kinds = np.array([m["kind"] for m in meta])
    scenes = np.array([m["scene"] for m in meta])
    rows, summary = [], {}
    for which in tabs:
        T = tabs[which]
        for name, idx in choices.items():
            if idx is None:
                idx = const_idx[which]
            per = []
            for i in range(n):
                t = np.asarray(T[i], np.float32)
                ch = int(idx[i])
                m = sample_metrics(t, ch, zero, l1)
                if not np.isfinite(clean_scores(t)[ch]):
                    m["score"], m["regret"] = 0.0, m["best"]              # invalid candidate chosen = worst case
                    m["invalid"] = 1
                tie = np.flatnonzero(clean_scores(t) >= m["best"] - 1e-3)
                dist = np.abs(grid.decode(tie) - grid.decode(ch)).max(1)   # worst face distance to each optimal candidate
                m["shift_err_cm"] = float(dist.min() * 100)
                per.append(m)
                rows.append({"sample_id": meta[i]["sample_id"], "kind": kinds[i], "scene": scenes[i], "category": meta[i]["category"],
                             "score_type": which, "method": name, "chosen": ch, **m})
            summary[(which, name)] = per
    write_csv(Path(a.work) / f"stage7_per_sample{a.tag}.csv", rows)

    # ---- report -------------------------------------------------------------------------------------------------
    names = list(choices) + ["random (expected)"]
    for subset, mask in (("REAL (un-augmented) samples", kinds == "orig"), ("ALL samples (incl. augmented copies)", np.ones(n, bool))):
        if mask.sum() == 0:
            continue
        print(f"\n  {subset}: {int(mask.sum())} samples in {len(set(scenes[mask]))} scene(s)")
        for which in tabs:
            T = tabs[which]
            best = np.array([summary[(which, list(choices)[0])][i]["best"] for i in range(n)])
            print(f"    score type: {which}    (mean oracle score {best[mask].mean():.3f})")
            print(f"      {'method':<24} {'mean score':>10} {'mean regret':>12} {'95% CI (scenes)':>20} {'optimal':>8} {'shift err [cm]':>15}")
            for name in names:
                if name == "random (expected)":
                    sc = np.array([np.nanmean(np.asarray(T[i], np.float32)) for i in range(n)])
                    reg = best - sc
                    print(f"      {name:<24} {sc[mask].mean():>10.3f} {reg[mask].mean():>12.4f} {'':>20} {'':>8} {'':>15}")
                    continue
                per = summary[(which, name)]
                sc = np.array([p["score"] for p in per])
                reg = np.array([p["regret"] for p in per])
                opt = np.array([p["in_tie"] for p in per])
                err = np.array([p["shift_err_cm"] for p in per])
                mu, lo, hi = bootstrap_ci(reg[mask], scenes[mask], a.bootstrap)
                ci = f"[{lo:.4f}, {hi:.4f}]" if np.isfinite(lo) else "(1 scene: no CI)"
                print(f"      {name:<24} {sc[mask].mean():>10.3f} {mu:>12.4f} {ci:>20} {opt[mask].mean():>8.0%} {err[mask].mean():>15.2f}")

    real_i = np.flatnonzero(kinds == "orig")
    if 0 < len(real_i) <= 40:
        print("\n  PER REAL SAMPLE (tol score; regret = oracle - score):")
        print(f"      {'sample':<40} {'oracle':>6} {'CV box':>7} {'const':>7} {'model':>7}   regret: CV box / const / model")
        cn = [k for k in choices if k.startswith("best constant")][0]
        for i in real_i:
            g = lambda nm: summary[("tol", nm)][i]
            mo = g("model") if "model" in choices else None
            print(f"      {meta[i]['sample_id']:<40} {g('CV box (shift 0)')['best']:>6.2f} {g('CV box (shift 0)')['score']:>7.2f} {g(cn)['score']:>7.2f} "
                  f"{(mo['score'] if mo else float('nan')):>7.2f}   {g('CV box (shift 0)')['regret']:.2f} / {g(cn)['regret']:.2f} / {(mo['regret'] if mo else float('nan')):.2f}")

    # ---- gates --------------------------------------------------------------------------------------------------
    zero_ok = max(abs(summary[("tol", "CV box (shift 0)")][i]["score"] - float(meta[i]["zero_tol"])) for i in range(n))
    gates.check("CV-box score equals the stage-4 value", zero_ok < 2e-3, f"max difference {zero_ok:.1e}")
    gates.check("regret is never negative", all(p["regret"] >= -1e-6 for p in summary[("tol", "CV box (shift 0)")]), "best >= chosen for all samples")
    inv_by = {}
    for (w_, name_), per in summary.items():
        inv_by[name_] = inv_by.get(name_, 0) + sum(p.get("invalid", 0) for p in per)
    inv = sum(inv_by.values())
    gates.check("no invalid candidates chosen", inv == 0, f"{inv} choices pointed to a candidate thinner than --min-dim (per method, both score types: {inv_by})")
    if "model" in choices:
        real = kinds == "orig"
        mask = real if real.sum() >= 3 else np.ones(n, bool)
        rm = np.mean([summary[("tol", "model")][i]["regret"] for i in np.flatnonzero(mask)])
        rc = np.mean([summary[("tol", "CV box (shift 0)")][i]["regret"] for i in np.flatnonzero(mask)])
        rk = np.mean([summary[("tol", cname)][i]["regret"] for i in np.flatnonzero(mask)])
        gates.check("model beats the CV box", rm < rc, f"regret {rm:.4f} vs {rc:.4f}", hard=False)
        gates.check("model beats the best constant shift", rm < rk, f"regret {rm:.4f} vs {rk:.4f} (if not: the model has not learned anything per-sample)", hard=False)
    print(f"\n  per-sample results: {a.work}/stage7_per_sample{a.tag}.csv")
    gates.finish()


if __name__ == "__main__":
    main()