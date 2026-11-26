#!/usr/bin/env python3
"""STAGE 4 - run the whole pipeline (ray extension -> voxelisation -> comparison with GT) for EVERY face-shift
candidate of EVERY sample and store the complete score table.

Per sample  work/stage4/tables/<split>/<sample_id>.npz :
    score_exact  float16 [n_cand]   exact voxel hit/miss score      (NaN = invalid candidate, thinner than --min-dim)
    score_tol    float16 [n_cand]   same with a --tol-vox voxel tolerance
    score_graded float16 [n_cand]   distance-weighted score (see shiftgrid.graded_spec; --graded-sigma 0 switches it off): a regenerated
                                    cell next to a GT cell counts less the farther away it is (exact 1.00, face .61, edge .37, vertex .22, 2 voxels .14, beyond 0)
    n_pred       uint16  [n_cand]   regenerated voxels
    n_viol       uint16  [n_cand]   rays whose measured point lies behind the candidate's entry (LiDAR saw through it)
Loss = 1 - score. Flat candidate index <-> six face shifts: ShiftGrid.decode / encode (see shiftgrid.py).
The best candidate = argmax; plateaus (ties) are resolved towards the smallest total shift.
Resumable: samples whose table already exists are skipped.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import Gates, banner, progress, read_csv, step, write_csv
from shiftgrid import Backend, add_grid_args, best_candidate, evaluate_candidates, graded_spec, grid_from_args


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--mem-mb", type=float, default=1500, help="memory budget per chunk (GPU or RAM)")
    ap.add_argument("--backend", default="auto", choices=["auto", "numpy", "torch"])
    ap.add_argument("--min-shadow-vox", type=int, default=5, help="skip samples with fewer shadow voxels")
    ap.add_argument("--only", type=int, default=0, help="debug: first N samples only")
    ap.add_argument("--gate-min-oracle", type=float, default=0.10, help="hard gate: median oracle score of the real samples")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--graded-sigma", type=float, default=1.0, help="graded score: width of the Gaussian weight in voxels (0 = do not compute it)")
    ap.add_argument("--graded-radius", type=int, default=2, help="graded score: cells farther than this many voxels get weight 0")
    add_grid_args(ap)
    a = ap.parse_args()

    banner(4, "loss tables (all face-shift candidates x all samples)")
    out = Path(a.work) / "stage4"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    gates = Gates(4)
    bk, grid, vs = Backend(a.backend), grid_from_args(a), a.voxel_size
    cfg = dict(voxel_size=vs, n_steps=grid.ns[0], n_steps_z=grid.ns[2], step=grid.step, min_dim=grid.min_dim,
               tol_vox=a.tol_vox, block_margin=vs if a.block_margin is None else a.block_margin, total=grid.total, voxel_frame=a.voxel_frame)
    cfg_path = out / "config.json"
    old = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
    if old is not None and not a.force and {k: v for k, v in old.items() if k != "graded"} != cfg:
        raise SystemExit(f"existing tables were made with another config:\n  {old}\n  now: {cfg}\n"
                         f"use --force (or another --work) to redo them")
    spec = graded_spec(a.graded_sigma, a.graded_radius) if a.graded_sigma > 0 else None
    cfg_path.write_text(json.dumps({**cfg, "graded": None if spec is None else {"sigma": spec["sigma"], "radius": spec["radius"]}}, indent=2))
    step(f"backend: {bk.name} | {grid.describe()}")
    step(f"voxel {vs * 100:.1f} cm ({a.voxel_frame}-aligned) | tolerance {a.tol_vox} voxel | block margin {cfg['block_margin'] * 100:.1f} cm")
    if spec is not None:
        step("graded score weights by squared offset in voxels: " + "  ".join(f"{int(r)}:{w:.2f}" for r, w in zip(spec["r2"], spec["w"])) +
             f"  (sigma {spec['sigma']:g}, beyond {spec['radius']} voxels 0; tables without it are recomputed)")

    rows = read_csv(Path(a.work) / "stage0" / "index.csv")
    boxes = np.load(Path(a.work) / "stage1" / "boxes.npz")
    bid = {s: i for i, s in enumerate(boxes["ids"])}
    rows = [r for r in rows if r["sample_id"] in bid]
    if a.only:
        rows = rows[:a.only]
    l1 = grid.l1_all()
    zero = grid.zero_index()
    step(f"{len(rows)} samples x {grid.total:,} candidates = {len(rows) * grid.total / 1e6:.0f} M evaluations")

    index, t0, new = [], time.time(), 0
    bar = progress(rows, total=len(rows), desc="samples")
    for r in bar:
        tp = out / "tables" / r["split"] / f"{r['sample_id']}.npz"
        tp.parent.mkdir(parents=True, exist_ok=True)
        i = bid[r["sample_id"]]
        box = (boxes["center"][i], boxes["dims"][i], boxes["R"][i])
        row = {"sample_id": r["sample_id"], "split": r["split"], "kind": r["kind"], "scene": r["scene"], "category": r["category"]}
        have = None
        if tp.exists() and not a.force:
            z = np.load(tp)
            if spec is None or ("score_graded" in z.files and abs(float(z["graded_sigma"]) - spec["sigma"]) < 1e-9
                                and int(z["graded_radius"]) == spec["radius"]):
                have = z
        sg = None
        if have is not None:
            se, st, nS = have["score_exact"].astype(np.float32), have["score_tol"].astype(np.float32), int(have["nS"])
            sg = have["score_graded"].astype(np.float32) if spec is not None else None
        else:
            z = np.load(r["path"])
            t1 = time.time()
            res = evaluate_candidates(bk, z["origin"], z["scan"].astype(np.float64), box, z["gt"].astype(np.float64),
                                      z["gt_shadow"], grid, vs, a.block_margin, a.tol_vox, a.mem_mb, frame=a.voxel_frame, graded=spec)
            nS = res["nS"]
            if nS < a.min_shadow_vox:
                row.update(status=f"skipped: {nS} shadow voxels in the envelope")
                index.append(row)
                continue
            se = res["score_exact"].astype(np.float16).astype(np.float32)      # same rounding as a resumed run
            st = res["score_tol"].astype(np.float16).astype(np.float32)
            extra = {}
            if spec is not None:
                sg = res["score_graded"].astype(np.float16).astype(np.float32)
                extra = dict(score_graded=sg.astype(np.float16), graded_sigma=spec["sigma"], graded_radius=spec["radius"])
            np.savez_compressed(tp, score_exact=se.astype(np.float16), score_tol=st.astype(np.float16),
                                n_pred=np.minimum(res["n_pred"], 65535).astype(np.uint16),
                                n_viol=np.minimum(res["n_viol"], 65535).astype(np.uint16), nS=nS, nG=res["nG"],
                                n_rays=res["n_rays"], **extra)
            new += 1
            bar.set_postfix_str(f"{time.time() - t1:.1f}s/sample, {res['n_rays']} rays, {nS} S-vox")
        be, mxe, nte = best_candidate(se, l1)
        bt, mxt, ntt = best_candidate(st, l1)
        sh_t = grid.decode(bt)
        on_edge = bool(np.any(np.abs(sh_t) >= grid.max_offset.repeat(2) - 1e-9))
        if sg is not None:
            bg, mxg, ntg = best_candidate(sg, l1)
            row.update(best_graded=bg, best_graded_score=mxg, n_ties_graded=ntg, zero_graded=float(sg[zero]))
        row.update(status="ok", n_shadow_vox=nS, best_exact=be, best_exact_score=mxe, n_ties_exact=nte,
                   best_tol=bt, best_tol_score=mxt, n_ties_tol=ntt, zero_exact=float(se[zero]), zero_tol=float(st[zero]),
                   best_tol_on_grid_edge=int(on_edge), **{f"shift_tol_{n}": float(v) for n, v in
                                                          zip(["lo_x", "hi_x", "lo_y", "hi_y", "lo_z", "hi_z"], sh_t)})
        index.append(row)
    write_csv(out / "index.csv", index)

    step("summary + gates")
    ok = [r for r in index if r["status"] == "ok"]
    real = [r for r in ok if r["kind"] == "orig"]
    sk = [r for r in index if r["status"] != "ok"]
    print(f"     evaluated {len(ok)} samples ({new} new), skipped {len(sk)}; {time.time() - t0:.0f}s")
    for name, rr in (("real", real), ("augmented", [r for r in ok if r["kind"] != "orig"])):
        if not rr:
            continue
        zt, bt_, ze, be_ = (np.array([r[k] for r in rr], float) for k in ("zero_tol", "best_tol_score", "zero_exact", "best_exact_score"))
        print(f"     {name:>9}: score(tol)   CV box (shift 0) median {np.median(zt):.3f} | oracle median {np.median(bt_):.3f} | "
              f"median regret {np.median(bt_ - zt):.3f}")
        print(f"     {'':>9}: score(exact) CV box (shift 0) median {np.median(ze):.3f} | oracle median {np.median(be_):.3f} | "
              f"median regret {np.median(be_ - ze):.3f}")
        if spec is not None:
            zg, bg_ = (np.array([r[k] for r in rr], float) for k in ("zero_graded", "best_graded_score"))
            print(f"     {'':>9}: score(graded) CV box (shift 0) median {np.median(zg):.3f} | oracle median {np.median(bg_):.3f} | "
                  f"median regret {np.median(bg_ - zg):.3f}")
    edge = np.mean([r["best_tol_on_grid_edge"] for r in ok]) if ok else 1.0
    shifts = np.array([[r[f"shift_tol_{n}"] for n in ("lo_x", "hi_x", "lo_y", "hi_y", "lo_z", "hi_z")] for r in ok]) * 100
    print("     oracle shifts [cm] (median / p10 / p90):  lo_x hi_x lo_y hi_y lo_z hi_z")
    print("       median", np.round(np.median(shifts, 0), 1), "\n       p10   ", np.round(np.percentile(shifts, 10, 0), 1),
          "\n       p90   ", np.round(np.percentile(shifts, 90, 0), 1))

    oracle_real = np.median([r["best_tol_score"] for r in real]) if real else 0.0
    headroom = np.median([r["best_tol_score"] - r["zero_tol"] for r in ok]) if ok else 0.0
    gates.check("samples evaluated", len(ok) > 0, f"{len(ok)} ok / {len(sk)} skipped")
    gates.check("oracle box reconstructs the shadow", oracle_real >= a.gate_min_oracle,
                f"median oracle score on real samples {oracle_real:.3f} (hard limit {a.gate_min_oracle}); near 0 means rays, GT and boxes are "
                f"not in one frame - check the stage 0/1 gates")
    gates.check("oracle clearly better than a typical shift-0", oracle_real >= 0.3 or headroom > 0.02,
                f"oracle {oracle_real:.3f}, median headroom over the CV box {headroom:.3f}", hard=False)
    gates.check("optimum inside the grid", edge <= 0.25, f"{edge:.0%} of samples have their optimum on the grid boundary "
                f"(raise --n-steps / --n-steps-z or --step if high)", hard=False)
    gates.check("few skipped samples", len(sk) <= 0.2 * len(index), f"{len(sk)}/{len(index)} skipped", hard=False)
    gates.finish()


if __name__ == "__main__":
    main()