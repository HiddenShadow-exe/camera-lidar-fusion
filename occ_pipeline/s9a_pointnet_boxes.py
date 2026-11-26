#!/usr/bin/env python3
"""STAGE 9a - boxes from the PointNet OBB model (train_obb.py), trained with OUR scene folds, for the real boxes of the pipeline.

Approach 2 of the comparison: "camera cluster -> OBB" regressed by the PointNet of train_obb.py (labels: your manual OBBs).
To feed it through the whole pipeline it must never predict a box of a scene it was trained on. So this script runs the same
scene-level K-fold cross-validation as stage 6 and re-uses ITS folds (read from the stage-6 prediction file):
    fold f:  train_obb.fit() on every sample of dataset_aug whose scene is NOT in fold f (all recordings, augmented copies, synthetic
             boxes; scenes that are not part of our evaluation - e.g. the single-box recordings - are always training data),
             then predict the OBB of every real box of fold f from its camera cluster (stage-0 cluster, LiDAR frame).
The predicted box (centre, length/width/height, yaw about z) is written to work/stage9/pointnet_boxes.npz and evaluated by
s9_compare_methods.py exactly like the CV box: LiDAR rays blocked by the occluder are extended to its front face, voxelised,
compared with the GT.   train_obb.py itself is imported unchanged (put it next to this script or pass --train-obb-dir).

    python s9a_pointnet_boxes.py --work work_virt --pred work_virt/stage6/pred_cv_graded.npz --aug-data dataset_aug
    (needs torch + a GPU for reasonable speed; one fold = one train_obb training run; finished folds are kept and skipped on re-runs)
    python s9a_pointnet_boxes.py ... --dry-run       # no torch, no training: writes the classical prior box of train_obb.py (plumbing test only)
"""
import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from common import Gates, banner, progress, read_csv, rot_z, step, write_csv


def import_train_obb(directory):
    for d in (directory, Path.cwd(), Path(__file__).resolve().parent):
        if d and (Path(d) / "train_obb.py").exists():
            sys.path.insert(0, str(Path(d).resolve()))
            break
    try:
        return importlib.import_module("train_obb")
    except ModuleNotFoundError as e:
        if e.name != "train_obb":
            raise
        raise SystemExit("train_obb.py not found: put it next to this script or pass --train-obb-dir <folder>")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="work")
    ap.add_argument("--pred", required=True, help="stage-6 cross-validation file (pred_cv<tag>.npz): only its 'ids' and 'fold' are used, so PointNet and our model share the folds")
    ap.add_argument("--aug-data", default="dataset_aug", help="folder written by augment_obb_dataset.py (train/val/<category>/<sample>/recording_data.json with obb_label)")
    ap.add_argument("--train-obb-dir", default=None, help="folder containing train_obb.py (default: next to this script / current folder)")
    ap.add_argument("--out", default=None, help="default <work>/stage9")
    ap.add_argument("--epochs", type=int, default=60, help="train_obb.py default")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--align-jitter-deg", type=float, default=10.0)
    ap.add_argument("--eval-every", type=int, default=10, help="print the held-out IoU every N epochs (display only, never used to pick a model)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--folds", default=None, help="comma list of folds to run (default all)")
    ap.add_argument("--force", action="store_true", help="retrain folds that are already finished")
    ap.add_argument("--use-odd", action="store_true",
                    help="also train on the ODD (occluded) recordings. Default: EVEN recordings only, because only those carry a correct hand label "
                         "(the label of an odd recording is the auto-computed visible-face box and fits the camera points badly)")
    ap.add_argument("--dry-run", action="store_true", help="no torch / no training: boxes = classical prior box of train_obb.py (plumbing test)")
    a = ap.parse_args()

    banner("9a", "PointNet OBB (train_obb.py) with the cross-validation folds of stage 6")
    gates = Gates("9a")
    work = Path(a.work)
    out = Path(a.out) if a.out else work / "stage9"
    out.mkdir(parents=True, exist_ok=True)
    T = import_train_obb(a.train_obb_dir)
    if not a.dry_run and T.torch is None:
        raise SystemExit("PyTorch is not installed (use --dry-run to test the plumbing only)")
    torch = T.torch

    # ---- our samples and folds -----------------------------------------------------------------------------------
    step("1/4 reading folds and the real samples of the pipeline")
    zp = np.load(a.pred)
    if "fold" not in zp.files:
        raise SystemExit(f"{a.pred} has no 'fold' array: it must come from  s6_train.py --cv K")
    fold_id = dict(zip(zp["ids"], zp["fold"]))
    s0 = {r["sample_id"]: r for r in read_csv(work / "stage0" / "index.csv")}
    boxes = np.load(work / "stage1" / "boxes.npz")
    have_box = set(boxes["ids"])
    ours = [sid for sid in zp["ids"] if sid in s0 and s0[sid]["kind"] == "orig" and fold_id[sid] >= 0 and sid in have_box]
    fold_of_scene = {s0[sid]["scene"]: int(fold_id[sid]) for sid in zp["ids"] if sid in s0 and fold_id[sid] >= 0}
    K = max(fold_of_scene.values()) + 1
    print(f"     {len(ours)} real boxes in {len({s0[s]['scene'] for s in ours})} scenes, {K} folds")
    cfg0 = json.loads((work / "stage0" / "config.json").read_text())
    floor_z = float(cfg0.get("floor_z", -0.4))

    step("2/4 loading the training samples of train_obb.py")
    if a.dry_run:
        samples = []
        print("     (dry run: no training samples needed)")
    else:
        samples = T.load_samples(a.aug_data)
        if not a.use_odd:
            def rec_no(it):
                m = re.search(r"recording_(\d+)", Path(it["path"]).parent.name) or re.search(r"recording_(\d+)", str(it["path"]))
                return int(m.group(1)) if m else None
            n0 = len(samples)
            samples = [it for it in samples if rec_no(it) is None or rec_no(it) % 2 == 0]
            unk = sum(1 for it in samples if rec_no(it) is None)
            print(f"     even recordings only: kept {len(samples)} of {n0} samples ({n0 - len(samples)} from odd recordings dropped; "
                  f"{unk} without a recording number kept) - use --use-odd to keep the odd ones")
            gates.check("training samples left after the even-only filter", len(samples) > 50, f"{len(samples)} samples")
        scenes_aug = {s["scene"] for s in samples}
        missing = sorted(set(fold_of_scene) - scenes_aug)
        gates.check("every evaluated scene exists in dataset_aug", not missing,
                    f"{len(set(fold_of_scene) - set(missing))}/{len(fold_of_scene)} scenes found" + (f"; MISSING: {missing[:6]}" if missing else "")
                    + "  (a missing scene is simply never part of the training data; check that dataset_aug was built from the same recordings / category names)", hard=False)
        print(f"     {len(samples)} samples, {len(scenes_aug)} scenes; "
              f"{sum(1 for s in samples if s['scene'] not in fold_of_scene)} samples of scenes outside our evaluation (always training data)")

    step(f"3/4 {'prior boxes' if a.dry_run else 'training one PointNet per fold and predicting the held-out real boxes'}")
    device = None if a.dry_run else torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    if device is not None:
        print(f"     device: {device}")
    run_folds = sorted(int(x) for x in a.folds.split(",")) if a.folds else list(range(K))
    t0 = time.time()
    for f in run_folds:
        fpath = out / f"pointnet_fold{f}.npz"
        if fpath.exists() and not a.force:
            print(f"     fold {f + 1}/{K}: {fpath.name} exists -> skipped (use --force to retrain)")
            continue
        te_ids = [sid for sid in ours if int(fold_id[sid]) == f]
        te_scenes = sorted({s0[s]["scene"] for s in te_ids})
        model = None
        if not a.dry_run:
            tr = [s for s in samples if fold_of_scene.get(s["scene"], -1) != f]
            te = [s for s in samples if fold_of_scene.get(s["scene"], -1) == f]
            print(f"\n     fold {f + 1}/{K}: train {len(tr)} samples ({len({s['scene'] for s in tr})} scenes) | held out {len(te)} samples of "
                  f"{len(te_scenes)} scenes: {', '.join(te_scenes[:4])}{'...' if len(te_scenes) > 4 else ''}")
            if not te:
                print("       [warn] no dataset_aug samples for the held-out scenes: the display IoU uses training samples")
            ns = argparse.Namespace(epochs=a.epochs, batch=a.batch, lr=a.lr, wd=a.wd, width=a.width, dropout=a.dropout, n_points=a.n_points,
                                    floor_z=floor_z, align_jitter_deg=a.align_jitter_deg, eval_every=a.eval_every, select_best=False,
                                    workers=a.workers, seed=a.seed + f, cpu=a.cpu, smoke=False)
            torch.manual_seed(a.seed + f)
            np.random.seed(a.seed + f)
            model, _, _ = T.fit(tr, te if te else tr[:16], ns, device, tag=f"[fold {f + 1}] ")
            torch.save({"state_dict": model.state_dict(), "width": a.width, "dropout": a.dropout, "floor_z": floor_z, "n_points": a.n_points,
                        "unit": T.UNIT}, out / f"pointnet_fold{f}.pth")
            model.eval()
        rec = {"ids": [], "center": [], "dims": [], "yaw": [], "prior_center": [], "prior_dims": [], "prior_yaw": []}
        for sid in te_ids:
            z = np.load(s0[sid]["path"])
            pts = z["cluster"].astype(np.float64)
            cam = -np.asarray(z["cam_R"], float) @ np.asarray(z["cam_t"], float)             # camera centre in the LiDAR frame (= train_obb.cam_center)
            item, origin, psi = T.prepare(pts, cam, float(z["floor_z"]) if "floor_z" in z.files else floor_z, a.n_points, np.random.default_rng(0))
            pc0, pd0, py0 = T.to_world(origin, psi, item["prior_c"], item["prior_d"], 0.0)   # the classical box train_obb starts from
            if a.dry_run:
                c, d, yaw = pc0, pd0, py0
            else:
                with torch.no_grad():
                    b = {k: torch.as_tensor(v)[None].to(device) for k, v in item.items()}
                    pc, pd, py = T.decode(b, *model(b["pts"], b["cond"]))
                c, d, yaw = T.to_world(origin, psi, pc[0].cpu().numpy(), pd[0].cpu().numpy(), py[0].item())
            for k, v in (("ids", sid), ("center", c), ("dims", d), ("yaw", yaw), ("prior_center", pc0), ("prior_dims", pd0), ("prior_yaw", py0)):
                rec[k].append(v)
        np.savez(fpath, ids=np.array(rec["ids"]), center=np.array(rec["center"]), dims=np.array(rec["dims"]), yaw=np.array(rec["yaw"]),
                 prior_center=np.array(rec["prior_center"]), prior_dims=np.array(rec["prior_dims"]), prior_yaw=np.array(rec["prior_yaw"]))
        print(f"     fold {f + 1}/{K}: {len(te_ids)} boxes written to {fpath.name}   ({time.time() - t0:.0f}s since start)")

    step("4/4 merging the folds")
    parts = [np.load(out / f"pointnet_fold{f}.npz") for f in range(K) if (out / f"pointnet_fold{f}.npz").exists()]
    if len(parts) < K:
        print(f"     only {len(parts)}/{K} folds finished -> merged file covers those boxes only")
    ids = np.concatenate([p["ids"] for p in parts])
    cat = lambda k: np.concatenate([p[k] for p in parts])
    yaw, pyaw = cat("yaw"), cat("prior_yaw")
    Rm = np.array([rot_z(y) for y in yaw])
    Rp = np.array([rot_z(y) for y in pyaw])
    np.savez(out / "pointnet_boxes.npz", ids=ids, center=cat("center"), dims=cat("dims"), R=Rm, yaw=yaw,
             prior_center=cat("prior_center"), prior_dims=cat("prior_dims"), prior_R=Rp,
             fold=np.array([fold_id[s] for s in ids]), dry_run=np.array(a.dry_run))
    bid = {s: i for i, s in enumerate(boxes["ids"])}
    rows = []
    for i, sid in enumerate(ids):
        b = bid[sid]
        rows.append({"sample_id": sid, "fold": int(fold_id[sid]),
                     "centre_diff_to_cv_cm": float(np.linalg.norm(cat("center")[i] - boxes["center"][b]) * 100),
                     **{f"pointnet_dim{k}_cm": float(cat("dims")[i][k] * 100) for k in range(3)},
                     **{f"cv_dim{k}_cm": float(boxes["dims"][b][k] * 100) for k in range(3)}})
    write_csv(out / "pointnet_boxes.csv", rows)
    d = np.array([r["centre_diff_to_cv_cm"] for r in rows])
    print(f"     {len(ids)} boxes -> {out / 'pointnet_boxes.npz'}  | centre difference PointNet vs CV box: median {np.median(d):.1f} cm, max {d.max():.1f} cm")
    gates.check("finite boxes", bool(np.all(np.isfinite(cat("center"))) and np.all(np.isfinite(cat("dims")))), f"{len(ids)} boxes")
    gates.check("all real boxes covered", len(ids) == len(ours), f"{len(ids)}/{len(ours)} (run all folds)", hard=False)
    gates.check("boxes are plausible", bool(np.all(cat("dims") > 0.02) and np.all(cat("dims") < 2.0)), f"dims between {cat('dims').min() * 100:.0f} and {cat('dims').max() * 100:.0f} cm")
    gates.finish()


if __name__ == "__main__":
    main()