#!/usr/bin/env python3
"""
Cluster -> 3D OBB training (revised).  Input: the folders written by augment_obb_dataset.py.

What changed compared with the first version, and why
  1. CANONICAL FRAME + RESIDUAL PREDICTION.  The cluster is rotated so that its minimum-area rectangle is axis-aligned,
     and the network only predicts CORRECTIONS (centre, size, small yaw) to that classical box. With 26 independent
     training boxes this is far easier than learning absolute yaw / size from scratch, and the starting point of the
     training is the classical estimate, so the printed 'baseline' IoU is what the network has to beat.
  2. SYMMETRY-AWARE LOSS.  A rectangle looks the same at yaw, yaw+180 deg and, with length and width swapped, at yaw+90 deg.
     Regressing cos/sin of ONE of those labellings gives contradictory targets. The corner loss takes the minimum over
     the four equivalent labellings.
  3. LOSS IN THE UNITS OF THE METRIC.  Everything is in decimetres and the loss is an L1 distance between box corners,
     so a 2 cm error is penalised as strongly as it costs IoU (the old SmoothL1 in metres gave ~38x more weight to yaw
     than to a 5 cm centre error).
  4. CAMERA-RELATIVE POSITION AS INPUT.  What the camera shows (side walls, edge smear) depends on where the cluster is
     relative to the camera axis; the old model only saw the centred points.
  5. EVALUATION YOU CAN TRUST.  Deterministic validation, results on the real un-augmented recordings only, a component
     ablation (which part of the box limits IoU?), and --cv K scene-wise cross-validation, because 6-7 validation
     boxes cannot rank epochs or models.
  6. Smaller network, AdamW + weight decay, warm-up + cosine schedule, the model of the LAST epoch is reported
     (no picking the best epoch on a tiny validation set; --select-best exists but is biased).

Usage:
    python train_obb.py --data dataset_aug                  # folder split train/ val/ as written by the augmentation
    python train_obb.py --data dataset_aug --cv 5           # scene-wise 5-fold cross-validation (recommended for the paper)
    python train_obb.py --data dataset_aug --smoke          # 2 quick epochs: checks that everything runs
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:                       # geometry, IoU and the preprocessing work without torch
    torch = None

UNIT = 0.1                                # network units: decimetres
R_C2L = np.diag([1.0, -1.0, -1.0])


# ==============================================================================
# Geometry that works for numpy arrays and torch tensors (xp = np or torch)
# ==============================================================================
def bev_corners(cx, cy, l, w, yaw, xp):
    """4 corners (clockwise, fixed order) of a rectangle with axis 0 (length l) at angle yaw -> (..., 4, 2)."""
    c, s = xp.cos(yaw), xp.sin(yaw)
    out = []
    for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        out.append(xp.stack([cx + a * l / 2 * c - b * w / 2 * s, cy + a * l / 2 * s + b * w / 2 * c], -1))
    return xp.stack(out, -2)


def corner_variants(pc, pd, py, tc, td, ty, xp):
    """Mean |corner difference| between the prediction and each of the 4 equivalent labellings of the target."""
    pcs = bev_corners(pc[:, 0], pc[:, 1], pd[:, 0], pd[:, 1], py, xp)
    out = []
    for k in range(4):
        l, w = (td[:, 0], td[:, 1]) if k % 2 == 0 else (td[:, 1], td[:, 0])
        tcs = bev_corners(tc[:, 0], tc[:, 1], l, w, ty + k * math.pi / 2, xp)
        out.append(abs(pcs - tcs).mean((-1, -2)))
    return xp.stack(out, 1)


def obb_loss(pc, pd, py, tc, td, ty, xp):
    """Per-sample loss: best-matching labelling of the footprint + L1 of the bottom and top height."""
    best = xp.amin(corner_variants(pc, pd, py, tc, td, ty, xp), 1)
    lz = (abs((pc[:, 2] - pd[:, 2] / 2) - (tc[:, 2] - td[:, 2] / 2))
          + abs((pc[:, 2] + pd[:, 2] / 2) - (tc[:, 2] + td[:, 2] / 2)))
    return best + lz


# ==============================================================================
# 3D IoU (rotated rectangle clipping, no shapely needed)
# ==============================================================================
def _ccw(poly):
    a = sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))
    return poly if a > 0 else poly[::-1]


def _clip(subject, clipper):
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def inter(s, e, a, b):
        x1, y1, x2, y2, x3, y3, x4, y4 = *s, *e, *a, *b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return e
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    out = subject
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inp, out = out, []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if inside(e, a, b):
                if not inside(s, a, b):
                    out.append(inter(s, e, a, b))
                out.append(e)
            elif inside(s, a, b):
                out.append(inter(s, e, a, b))
            s = e
    return out


def _area(p):
    return abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))) / 2 if len(p) > 2 else 0.0


def iou3d(c1, d1, y1, c2, d2, y2):
    """c = (x, y, z) centre, d = (length, width, height), y = yaw of the length axis."""
    ih = max(0.0, min(c1[2] + d1[2] / 2, c2[2] + d2[2] / 2) - max(c1[2] - d1[2] / 2, c2[2] - d2[2] / 2))
    if ih <= 0:
        return 0.0
    p1 = _ccw([tuple(p) for p in bev_corners(c1[0], c1[1], d1[0], d1[1], y1, np)])
    p2 = _ccw([tuple(p) for p in bev_corners(c2[0], c2[1], d2[0], d2[1], y2, np)])
    iv = _area(_clip(p1, p2)) * ih
    union = d1[0] * d1[1] * d1[2] + d2[0] * d2[1] * d2[2] - iv
    return float(iv / union) if union > 1e-12 else 0.0


# ==============================================================================
# Data: loading, classical box, canonical frame
# ==============================================================================
def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def cam_center(calib):
    e = calib["lidar_to_cam_extrinsic"]
    return -(rot_z(float(e["lidar_yaw"])) @ R_C2L) @ np.asarray(e["lidar_position"], dtype=np.float64)


def min_area_rect_yaw(xy):
    """Yaw in [0, pi) of the minimum-area rectangle of a 2D point set, long side first."""
    try:
        P = xy[ConvexHull(xy).vertices]
    except Exception:
        w, v = np.linalg.eigh(np.cov(xy.T))
        return float(np.arctan2(v[1, 1], v[0, 1]) % np.pi)
    best = None
    for i in range(len(P)):
        e = P[(i + 1) % len(P)] - P[i]
        ang = math.atan2(e[1], e[0])
        c, s = math.cos(ang), math.sin(ang)
        x, y = P[:, 0] * c + P[:, 1] * s, -P[:, 0] * s + P[:, 1] * c
        l, w = x.max() - x.min(), y.max() - y.min()
        if best is None or l * w < best[0]:
            best = (l * w, ang, l, w)
    _, ang, l, w = best
    if w > l:
        ang += math.pi / 2
    return float(ang % math.pi)


def canonical_frame(points, cam, psi, floor_z):
    """Rotate the cluster by -psi around its centroid. Returns origin, points, the classical box (AABB in this frame)
    and the camera position relative to the cluster (all in metres)."""
    origin = points.mean(0)
    Rm = rot_z(-psi).T
    P = (points - origin) @ Rm
    lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    top = np.percentile(P[:, 2], 99.5)
    floor_rel = floor_z - origin[2]
    prior_c = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (top + floor_rel) / 2])
    prior_d = np.array([hi[0] - lo[0], hi[1] - lo[1], top - floor_rel])
    return origin, P, prior_c, prior_d, (cam - origin) @ Rm


def prepare(points, cam, floor_z, n_points, rng, psi=None):
    """Network input for one cluster. Used for training AND for inference."""
    psi = min_area_rect_yaw(points[:, :2]) if psi is None else psi
    origin, P, prior_c, prior_d, cam_rel = canonical_frame(points, cam, psi, floor_z)
    idx = rng.choice(len(P), n_points, replace=len(P) < n_points)
    item = {"pts": (P[idx] / UNIT).astype(np.float32),
            "cond": (np.concatenate([cam_rel, prior_c, prior_d]) / UNIT).astype(np.float32),
            "prior_c": (prior_c / UNIT).astype(np.float32), "prior_d": (prior_d / UNIT).astype(np.float32)}
    return item, origin, psi


def to_world(origin, psi, c_dm, d_dm, yaw):
    """Canonical-frame box (decimetres) -> world frame (metres)."""
    return np.asarray(c_dm) * UNIT @ rot_z(-psi) + origin, np.asarray(d_dm) * UNIT, float(yaw) + psi


def load_samples(root, log=print):
    samples, skipped_tilted, low_cov = [], 0, 0
    for p in sorted(Path(root).glob("*/*/*/recording_data.json")):
        with open(p) as f:
            doc = json.load(f)
        pts = np.asarray(doc["3d_data_camera"]["object_points"], dtype=np.float64).reshape(-1, 3)
        lab = doc["obb_label"]
        R = np.asarray(lab["rotation"], dtype=np.float64).reshape(3, 3)
        d = np.asarray(lab["dimensions"], dtype=np.float64)
        vi = int(np.argmax(np.abs(R[2, :])))
        if abs(R[2, vi]) < 0.98 or len(pts) < 100:
            skipped_tilted += 1                                # BEV-yaw model: tilted boxes are not supported
            continue
        a0, a1 = [k for k in range(3) if k != vi]
        meta = doc.get("sample_metadata", {})
        aug = meta.get("augmentation", {})
        c = np.asarray(lab["center"], dtype=np.float64)
        local = np.abs((pts - c) @ R)
        if np.mean(np.all(local <= d / 2 + 0.03, axis=1)) < 0.8:
            low_cov += 1
        samples.append({
            "points": pts, "center": c, "dims": np.array([d[a0], d[a1], d[vi]]),
            "yaw": math.atan2(R[1, a0], R[0, a0]), "cam": cam_center(doc["calibration"]),
            "psi0": min_area_rect_yaw(pts[:, :2]), "scene": meta.get("scene", str(p.parent.parent)),
            "type": aug.get("type", "orig"), "split": p.relative_to(root).parts[0], "path": str(p)})
    log(f"loaded {len(samples)} samples ({skipped_tilted} skipped: tilted label or too few points; "
        f"{low_cov} with <80% of the points inside the label +3 cm -> check those labels)")
    return samples


def make_item(s, rng, train, cfg):
    jitter = rng.uniform(-cfg["jitter"], cfg["jitter"]) if train and cfg["jitter"] > 0 else 0.0
    item, origin, psi = prepare(s["points"], s["cam"], cfg["floor_z"], cfg["n_points"], rng, s["psi0"] + jitter)
    item["tc"] = (((s["center"] - origin) @ rot_z(-psi).T) / UNIT).astype(np.float32)
    item["td"] = (s["dims"] / UNIT).astype(np.float32)
    item["ty"] = np.float32(s["yaw"] - psi)
    return item


# ==============================================================================
# Evaluation (numpy)
# ==============================================================================
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def score(rec):
    """Per-sample IoU and errors, with the 'what if this component were perfect' ablation."""
    keys = ("iou", "iou_prior", "iou_gt_center", "iou_gt_dims", "iou_gt_yaw", "e_cxy", "e_cz", "e_dims", "e_yaw")
    out = {k: [] for k in keys}
    for i in range(len(rec["py"])):
        pc, pd, py = rec["pc"][i], rec["pd"][i], float(rec["py"][i])
        tc, td, ty = rec["tc"][i], rec["td"][i], float(rec["ty"][i])
        cand = corner_variants(pc[None], pd[None], np.array([py]), tc[None], td[None], np.array([ty]), np)[0]
        kb = int(np.argmin(cand))
        vd = np.array([td[0], td[1], td[2]]) if kb % 2 == 0 else np.array([td[1], td[0], td[2]])
        vy = ty + kb * math.pi / 2
        out["iou"].append(iou3d(pc, pd, py, tc, td, ty))
        out["iou_prior"].append(iou3d(rec["prior_c"][i], rec["prior_d"][i], 0.0, tc, td, ty))
        out["iou_gt_center"].append(iou3d(tc, pd, py, tc, td, ty))
        out["iou_gt_dims"].append(iou3d(pc, vd, py, tc, td, ty))
        out["iou_gt_yaw"].append(iou3d(pc, pd, vy, tc, td, ty))
        out["e_cxy"].append(np.linalg.norm((pc - tc)[:2]) * UNIT * 100)             # dm -> cm
        out["e_cz"].append(abs(pc[2] - tc[2]) * 100 * UNIT)
        out["e_dims"].append(np.mean(np.abs(pd - vd)) * 100 * UNIT)
        out["e_yaw"].append(abs(math.degrees(wrap(py - vy))))
    return {k: np.array(v) for k, v in out.items()}


def summarize(title, sc):
    if not len(sc["iou"]):
        print(f"{title}: no samples")
        return
    f = lambda k: sc[k].mean()
    print(f"{title}  (n = {len(sc['iou'])})")
    print(f"    3D IoU   model {f('iou'):.3f} (median {np.median(sc['iou']):.3f}, >0.7: {np.mean(sc['iou'] > .7):.0%}) | "
          f"classical baseline {f('iou_prior'):.3f}")
    print(f"    errors   centre xy {f('e_cxy'):.1f} cm | centre z {f('e_cz'):.1f} cm | size {f('e_dims'):.1f} cm | yaw {f('e_yaw'):.1f} deg")
    print(f"    IoU if one part were perfect:  centre {f('iou_gt_center'):.3f} | size {f('iou_gt_dims'):.3f} | yaw {f('iou_gt_yaw'):.3f}")


# ==============================================================================
# PyTorch part
# ==============================================================================
if torch is not None:

    class OBBData(Dataset):
        def __init__(self, samples, train, cfg, seed=0):
            self.s, self.train, self.cfg, self.seed = samples, train, cfg, seed

        def __len__(self):
            return len(self.s)

        def __getitem__(self, i):
            rng = np.random.default_rng() if self.train else np.random.default_rng(self.seed + i)   # val: deterministic
            return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in make_item(self.s[i], rng, self.train, self.cfg).items()}

    class OBBNet(nn.Module):
        """Shared point MLP -> max + mean pooling -> + camera/classical-box condition -> 7 corrections (zero-initialised)."""

        def __init__(self, width=64, dropout=0.2, cond_dim=9):
            super().__init__()
            c1, c2, c3 = width, 2 * width, 4 * width
            self.mlp = nn.Sequential(nn.Conv1d(3, c1, 1), nn.BatchNorm1d(c1), nn.ReLU(),
                                     nn.Conv1d(c1, c2, 1), nn.BatchNorm1d(c2), nn.ReLU(),
                                     nn.Conv1d(c2, c3, 1), nn.BatchNorm1d(c3), nn.ReLU())
            self.cond = nn.Sequential(nn.Linear(cond_dim, 64), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(2 * c3 + 64, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(256, 128), nn.ReLU())
            self.out = nn.Linear(128, 7)
            nn.init.zeros_(self.out.weight)                    # start exactly at the classical box
            nn.init.zeros_(self.out.bias)

        def forward(self, pts, cond):
            x = self.mlp(pts.transpose(1, 2))                  # (B, c3, N)
            g = torch.cat([x.max(2)[0], x.mean(2)], 1)
            o = self.out(self.head(torch.cat([g, self.cond(cond)], 1)))
            return o[:, 0:3], o[:, 3:6], (math.pi / 4) * torch.tanh(o[:, 6])      # d centre, d size (dm), yaw correction

    def decode(b, dc, dd, dy):
        return b["prior_c"] + dc, (b["prior_d"] + dd).clamp(min=0.5), dy

    @torch.no_grad()
    def predict(model, loader, device):
        model.eval()
        recs = {k: [] for k in ("pc", "pd", "py", "prior_c", "prior_d", "tc", "td", "ty")}
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            pc, pd, py = decode(b, *model(b["pts"], b["cond"]))
            for k, v in (("pc", pc), ("pd", pd), ("py", py), ("prior_c", b["prior_c"]), ("prior_d", b["prior_d"]),
                         ("tc", b["tc"]), ("td", b["td"]), ("ty", b["ty"])):
                recs[k].append(v.cpu().numpy())
        return {k: np.concatenate(v) for k, v in recs.items()}

    def fit(train_s, val_s, args, device, tag=""):
        cfg = {"jitter": math.radians(args.align_jitter_deg), "floor_z": args.floor_z, "n_points": args.n_points}
        cfg_val = dict(cfg, jitter=0.0)
        bs = max(2, min(args.batch, len(train_s) // 2))
        tl = DataLoader(OBBData(train_s, True, cfg), batch_size=bs, shuffle=True, drop_last=True, num_workers=args.workers)
        vl_all = DataLoader(OBBData(val_s, False, cfg_val), batch_size=64, shuffle=False)
        val_orig = [s for s in val_s if s["type"] == "orig"] or val_s
        vl_orig = DataLoader(OBBData(val_orig, False, cfg_val), batch_size=64, shuffle=False)

        model = OBBNet(args.width, args.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        warm = 3
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda e: min(1.0, (e + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * e / args.epochs))))
        best, best_state = -1.0, None
        t0 = time.time()
        for ep in range(args.epochs):
            model.train()
            tot, n = 0.0, 0
            for b in tl:
                b = {k: v.to(device) for k, v in b.items()}
                pc, pd, py = decode(b, *model(b["pts"], b["cond"]))
                loss = obb_loss(pc, pd, py, b["tc"], b["td"], b["ty"], torch).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += loss.item()
                n += 1
            sched.step()
            if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
                sc = score(predict(model, vl_orig, device))
                m = sc["iou"].mean() if len(sc["iou"]) else float("nan")
                print(f"{tag}epoch {ep + 1:3d}/{args.epochs} | train loss {tot / max(n, 1):.4f} (dm) | "
                      f"val IoU on {len(sc['iou'])} real recordings {m:.3f} | baseline {sc['iou_prior'].mean():.3f}", flush=True)
                if args.select_best and m > best:
                    best, best_state = m, {k: v.clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"{tag}[note] --select-best: the reported model is picked on the validation set -> optimistic")
        print(f"{tag}training time {time.time() - t0:.0f}s")
        return model, predict(model, vl_orig, device), predict(model, vl_all, device)

    def run(args):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        print(f"device: {device}")
        samples = load_samples(args.data)
        if args.smoke:
            args.epochs, args.eval_every = 2, 1
        if args.cv > 1:
            scenes = sorted({s["scene"] for s in samples})
            order = np.random.default_rng(args.seed).permutation(len(scenes))
            fold_of = {scenes[i]: r % args.cv for r, i in enumerate(order)}
            allrec = []
            for k in range(args.cv):
                tr = [s for s in samples if fold_of[s["scene"]] != k]
                te = [s for s in samples if fold_of[s["scene"]] == k]
                if args.smoke:
                    tr, te = tr[:48], te[:16]
                print(f"\n=== fold {k + 1}/{args.cv}: train {len(tr)} samples, held-out {len(te)} "
                      f"({sum(s['type'] == 'orig' for s in te)} real, scenes {sorted({s['scene'] for s in te})[:3]}...) ===")
                _, rec_orig, _ = fit(tr, te, args, device, tag=f"[fold {k + 1}] ")
                sc = score(rec_orig)
                summarize(f"fold {k + 1}", sc)
                allrec.append(sc)
            sc_all = {key: np.concatenate([a[key] for a in allrec]) for key in allrec[0]}
            print("\n" + "=" * 70)
            summarize(f"CROSS-VALIDATION over {args.cv} folds, every real recording predicted by a model that never saw its scene", sc_all)
            per = [a["iou"].mean() for a in allrec]
            print(f"    per-fold IoU mean {np.mean(per):.3f} +- {np.std(per):.3f} | model better than baseline on "
                  f"{np.mean(sc_all['iou'] > sc_all['iou_prior']):.0%} of the recordings")
            return
        tr = [s for s in samples if s["split"] == "train"]
        te = [s for s in samples if s["split"] == "val"]
        if args.smoke:
            tr, te = tr[:48], te[:16]
        n_real = sum(s["type"] == "orig" for s in te)
        print(f"train {len(tr)} samples ({len({s['scene'] for s in tr})} scenes) | val {len(te)} samples, {n_real} real recordings "
              f"({len({s['scene'] for s in te})} scenes)")
        if n_real < 15:
            print(f"[warning] only {n_real} independent validation recordings: IoU differences below ~0.05 are noise. Use --cv 5.")
        model, rec_orig, rec_all = fit(tr, te, args, device)
        print()
        summarize("validation, real recordings only (the number to report)", score(rec_orig))
        summarize("validation, all samples (augmented copies of the same recordings, optimistic)", score(rec_all))
        torch.save({"state_dict": model.state_dict(), "width": args.width, "dropout": args.dropout, "floor_z": args.floor_z,
                    "n_points": args.n_points, "unit": UNIT}, args.out)
        print(f"\nsaved {args.out}")

    @torch.no_grad()
    def predict_obb(model, points, calib, floor_z=-0.4, n_points=1024, device="cpu"):
        """Use in the pipeline: raw cluster (N, 3) in the LiDAR frame -> dict(center, dimensions, yaw) in the LiDAR frame."""
        model.eval()
        item, origin, psi = prepare(np.asarray(points, dtype=np.float64), cam_center(calib), floor_z, n_points,
                                    np.random.default_rng(0))
        b = {k: torch.as_tensor(v)[None].to(device) for k, v in item.items()}
        pc, pd, py = decode(b, *model(b["pts"], b["cond"]))
        c, d, yaw = to_world(origin, psi, pc[0].cpu().numpy(), pd[0].cpu().numpy(), py[0].item())
        return {"center": c, "dimensions": d, "yaw": yaw}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset_aug")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--floor-z", type=float, default=-0.4, help="floor height in the LiDAR frame (rig: LiDAR is 0.4 m above the floor)")
    ap.add_argument("--align-jitter-deg", type=float, default=10.0, help="random error of the classical yaw during training")
    ap.add_argument("--cv", type=int, default=0, help="K-fold cross-validation over scenes")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--select-best", action="store_true", help="report the best epoch on val (biased)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="obb_model.pth")
    args = ap.parse_args()
    if torch is None:
        raise SystemExit("PyTorch is not installed.")
    run(args)


if __name__ == "__main__":
    main()