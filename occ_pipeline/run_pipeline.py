#!/usr/bin/env python3
"""Runs stages 0-4 in order, prints a progress overview and stops at the first failed gate.

    python run_pipeline.py --input dataset --work work --n-aug 20
    python run_pipeline.py --input dataset --work work --from-stage 4 --to-stage 4 --n-steps 3   # only the loss tables
    python run_pipeline.py --input dataset --work work --from-stage 6 --epochs 80              # only train + evaluate

Every stage can also be run on its own (python s0_augment_pairs.py --help). Stages 0/1 are skipped when their DONE
marker exists (--force redoes them); stage 4 resumes per sample.
"""
import argparse
import os
import subprocess
import sys
import time

STAGES = [(0, "s0_augment_pairs.py", "augment pairs"), (1, "s1_cv_boxes.py", "initial CV boxes"),
          (2, "s2_check_grid.py", "shift grid check"), (3, "s3_parity.py", "evaluator parity"),
          (4, "s4_loss_tables.py", "loss tables"), (5, "s5_build_dataset.py", "pack dataset"),
          (6, "s6_train.py", "train model (needs torch)"), (7, "s7_evaluate.py", "regret evaluation")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--from-stage", type=int, default=0)
    ap.add_argument("--to-stage", type=int, default=7)
    ap.add_argument("--n-aug", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--voxel-size", type=float, default=0.03)
    ap.add_argument("--n-steps", type=int, default=3)
    ap.add_argument("--n-steps-z", type=int, default=None)
    ap.add_argument("--step", type=float, default=None)
    ap.add_argument("--tol-vox", type=int, default=1)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--mem-mb", type=float, default=1500)
    ap.add_argument("--only", type=int, default=0, help="debug: first N pairs (stage 0) / samples (stage 4)")
    ap.add_argument("--exclude-units", nargs="*", default=[], help="stage 0: units to drop, e.g. occluded_boxes_07_u1")
    ap.add_argument("--epochs", type=int, default=60, help="stage 6")
    ap.add_argument("--loss", default="expected", help="stage 6: expected | ce")
    ap.add_argument("--score", default="tol", help="stage 6: tol | exact | mix | graded")
    ap.add_argument("--smoke", action="store_true", help="stage 6: 2 epochs on 64 samples")
    ap.add_argument("--graded-sigma", type=float, default=1.0, help="stage 4: width (voxels) of the distance-weighted score, 0 = off")
    ap.add_argument("--graded-radius", type=int, default=2, help="stage 4: weight 0 beyond this many voxels")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    grid = ["--voxel-size", str(a.voxel_size), "--n-steps", str(a.n_steps), "--tol-vox", str(a.tol_vox)]
    if a.n_steps_z is not None:
        grid += ["--n-steps-z", str(a.n_steps_z)]
    if a.step is not None:
        grid += ["--step", str(a.step)]
    extra = {0: ["--input", os.path.abspath(a.input), "--n-aug", str(a.n_aug), "--seed", str(a.seed)] + (["--only", str(a.only)] if a.only else [])
             + (["--exclude-units", *a.exclude_units] if a.exclude_units else []),
             1: [], 2: grid, 5: ["--force"] if a.force else [],
             6: ["--epochs", str(a.epochs), "--loss", a.loss, "--score", a.score] + (["--smoke"] if a.smoke else []),
             7: [],
             3: grid + ["--backend", a.backend],
             4: grid + ["--backend", a.backend, "--mem-mb", str(a.mem_mb), "--graded-sigma", str(a.graded_sigma),
                        "--graded-radius", str(a.graded_radius)] + (["--only", str(a.only)] if a.only else [])}
    if a.force:
        extra[0].append("--force")
        extra[1].append("--force")
    todo = [s for s in STAGES if a.from_stage <= s[0] <= a.to_stage]
    t0 = time.time()
    for n, script, name in todo:
        print(f"\n######## [{n - todo[0][0] + 1}/{len(todo)}] stage {n}: {name} ########", flush=True)
        t1 = time.time()
        here = os.path.dirname(os.path.abspath(__file__))
        rc = subprocess.call([sys.executable, "-u", os.path.join(here, script), "--work", os.path.abspath(a.work)] + extra[n], cwd=here)
        if rc != 0:
            print(f"\n!!! stage {n} ({name}) failed with exit code {rc} - pipeline stopped. Fix it, then resume with "
                  f"--from-stage {n}", flush=True)
            sys.exit(rc)
        print(f"######## stage {n} finished in {time.time() - t1:.0f}s ########", flush=True)
    print(f"\nAll requested stages done in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()