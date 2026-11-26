#!/usr/bin/env python3
"""STAGE 5 - pack everything the model needs into a few arrays (train / val), so training never touches the 1000s of npz files.

Per sample:
    points  (P, 3) float16   camera cluster in the frame of the CV box:  (x - c) @ R   [m]  (P random points, with replacement if fewer)
    ctx     (9,)   float32   LiDAR origin in box frame (3) | camera centre in box frame (3) | CV box dims (3)
    tab_tol / tab_exact  (total,) float16   stage-4 score of every face-shift candidate (NaN = invalid candidate)
    tab_graded           same for the distance-weighted score (only if stage 4 computed it for every sample)
The model therefore works in the CV-box frame in which the six face shifts are defined; nothing about the world pose is needed.

Output: work/stage5/{train,val}_{points,ctx,tab_tol,tab_exact[,tab_graded]}.npy (+ meta CSV)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import Gates, banner, progress, read_csv, step, write_csv
from shiftgrid import ShiftGrid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--n-points", type=int, default=2048, help="stored points per cluster (training draws a random subset)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = Path(a.work) / "stage5"
    banner(5, "pack the dataset for the model")
    if (out / "DONE").exists() and not a.force:
        print(f"  {out}/DONE exists -> skipping (use --force to redo)")
        return
    out.mkdir(parents=True, exist_ok=True)
    gates = Gates(5)
    t0 = time.time()

    cfg = json.loads((Path(a.work) / "stage4" / "config.json").read_text())
    grid = ShiftGrid(cfg["n_steps"], cfg["n_steps_z"], cfg["step"], cfg["min_dim"])
    total = grid.total
    step(f"grid from stage 4: {grid.describe()}")

    s0 = {r["sample_id"]: r for r in read_csv(Path(a.work) / "stage0" / "index.csv")}
    s4 = {r["sample_id"]: r for r in read_csv(Path(a.work) / "stage4" / "index.csv") if r["status"] == "ok"}
    boxes = np.load(Path(a.work) / "stage1" / "boxes.npz")
    bid = {s: i for i, s in enumerate(boxes["ids"])}
    ids = [s for s in s4 if s in s0 and s in bid]
    step(f"{len(ids)} samples have a stage-1 box and a stage-4 table")
    has_g = bool(ids) and all(s4[s_].get("best_graded", "") not in ("", None) for s_ in ids)
    step("graded score tables: " + ("found for every sample -> packed as tab_graded" if has_g else "not available for every sample (rerun stage 4 without --graded-sigma 0)"))

    meta_all, zero_check, plateau = {}, [], []
    scenes = {"train": set(), "val": set()}
    for split in ("train", "val"):
        sel = [s for s in ids if s0[s]["split"] == split]
        n = len(sel)
        if n == 0:
            print(f"  [warn] split '{split}' is empty")
            continue
        step(f"{split}: packing {n} samples")
        pts = np.zeros((n, a.n_points, 3), np.float16)
        ctx = np.zeros((n, 9), np.float32)
        tt = np.lib.format.open_memmap(out / f"{split}_tab_tol.npy", mode="w+", dtype=np.float16, shape=(n, total))
        te = np.lib.format.open_memmap(out / f"{split}_tab_exact.npy", mode="w+", dtype=np.float16, shape=(n, total))
        tg = np.lib.format.open_memmap(out / f"{split}_tab_graded.npy", mode="w+", dtype=np.float16, shape=(n, total)) if has_g else None
        if not has_g:
            (out / f"{split}_tab_graded.npy").unlink(missing_ok=True)
        meta = []
        for k, sid in enumerate(progress(sel, total=n, desc=f"{split}")):
            z = np.load(s0[sid]["path"])
            i = bid[sid]
            c, R = boxes["center"][i], boxes["R"][i]
            loc = (z["cluster"].astype(np.float64) - c) @ R
            rng = np.random.default_rng([a.seed, k, 0 if split == "train" else 1])
            pts[k] = loc[rng.choice(len(loc), a.n_points, replace=len(loc) < a.n_points)]
            cam = -z["cam_R"] @ z["cam_t"]
            ctx[k] = np.concatenate([(z["origin"] - c) @ R, (cam - c) @ R, boxes["dims"][i]])
            tab = np.load(Path(a.work) / "stage4" / "tables" / split / f"{sid}.npz")
            tt[k], te[k] = tab["score_tol"], tab["score_exact"]
            if has_g:
                tg[k] = tab["score_graded"]
            r = s4[sid]
            meta.append({"sample_id": sid, "kind": r["kind"], "scene": r["scene"], "category": r["category"],
                         "best_tol": int(r["best_tol"]), "best_exact": int(r["best_exact"]),
                         "zero_tol": float(r["zero_tol"]), "zero_exact": float(r["zero_exact"]),
                         **({"best_graded": int(r["best_graded"]), "zero_graded": float(r["zero_graded"])} if has_g else {})})
            scenes[split].add(r["scene"])
            if k % max(1, n // 20) == 0:
                x = tab["score_tol"].astype(np.float32)
                plateau.append(float(np.mean(x >= np.nanmax(x) - 1e-3)))
        tt.flush(); te.flush(); del tt, te
        if has_g:
            tg.flush(); del tg
        np.save(out / f"{split}_points.npy", pts)
        np.save(out / f"{split}_ctx.npy", ctx)
        write_csv(out / f"{split}_meta.csv", meta)
        meta_all[split] = (pts, ctx, meta)

    step("checks")
    for split, (pts, ctx, meta) in meta_all.items():
        tt = np.load(out / f"{split}_tab_tol.npy", mmap_mode="r")
        for k in np.random.default_rng(0).choice(len(meta), min(20, len(meta)), replace=False):
            zero_check.append(abs(float(tt[k, grid.zero_index()]) - meta[k]["zero_tol"]))
        gates.check(f"{split}: finite inputs", bool(np.isfinite(pts.astype(np.float32)).all() and np.isfinite(ctx).all()),
                    f"{len(meta)} samples, points {pts.shape}, ctx {ctx.shape}")
        loc = pts.astype(np.float32)
        gates.check(f"{split}: clusters sit on the CV box", float(np.median(np.abs(loc).max(axis=(1, 2)))) < 1.0,
                    f"median max |coordinate| in the box frame = {np.median(np.abs(loc).max(axis=(1, 2))):.2f} m (box frame, so ~half box size + CV error)")
    gates.check("train / val scenes disjoint", not (scenes["train"] & scenes["val"]),
                f"{len(scenes['train'])} train scenes, {len(scenes['val'])} val scenes")
    gates.check("val split not empty", "val" in meta_all, "model selection and the final evaluation need real validation scenes", hard=False)
    gates.check("table zero-shift entry matches stage 4", bool(zero_check) and max(zero_check) < 2e-3,
                f"max |difference| = {max(zero_check) if zero_check else float('nan'):.1e} (float16 rounding)")
    print(f"     typical plateau: {np.mean(plateau) * 100:.2f}% of all candidates are within 0.001 of the best tolerance score "
          f"({np.mean(plateau) * total:.0f} candidates) -> ties are broken towards the smallest shift")
    (out / "config.json").write_text(json.dumps({**cfg, "n_points": a.n_points}, indent=2))
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"     stage 5 took {time.time() - t0:.0f}s")
    gates.finish()


if __name__ == "__main__":
    main()