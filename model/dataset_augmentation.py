#!/usr/bin/env python3
"""
Augmentation for training a "camera point cluster -> 3D OBB" model from the recorded json files.

Why not just translate / scale the cluster?
  The cluster is a view of a box from a camera that looks straight down. What it shows (how much of the side walls
  is visible, where the depth edge is smeared) depends on WHERE the box is relative to the camera axis, and the depth
  noise grows with the square of the distance. Moving or scaling the points therefore produces clouds that no real
  camera could have recorded, while keeping the label. So the tool only uses three kinds of changes:

  1. EXACT geometry (real cloud + rotated label)
       rotation about the vertical axis through the camera centre. For a level, downward-looking pinhole camera the
       viewing geometry is unchanged by this rotation (same incidence angles, same self-occlusion), so the real
       cluster stays physically valid. Very small shifts (default 5 cm = ~1 deg change of the viewing angle) are
       allowed as an approximation. Points that would leave the image are rejected.
  2. LABEL-PRESERVING artifacts (the label stays the true box, the cloud gets worse in the ways you want the model to see
       through): extra depth noise along the viewing ray, depth bleeding at the silhouette (the box looks bigger),
       flying pixels, one-sided dropout of the silhouette (shadowing), random voxel down-sampling / point dropout.
  3. SENSOR RE-SIMULATION (--n-synthetic): new boxes with a different position, yaw and size are ray-cast from the camera
       through the real pixel grid, so side-wall visibility and point density are right for that position, then the
       same artifacts are applied. The label is exact by construction.

Labels:
  The 'obb_3d' stored in the recordings is computed from the LiDAR points of the visible face (in your sample it is only
  0.19 m thick), so it is NOT the real box. Put the corrected box into the json as "obb_label" (same fields:
  center, dimensions, rotation = 3x3 row-major, axes = columns) or point --label-key at it.

Output (one folder per sample, no images / LiDAR / full camera clouds):
    <output>/<train|val>/<category>/<sample>/recording_data.json   (and sample.npz with --format npz|both)
    <output>/manifest.csv
The split is made per SCENE (the occluded recording and its ground-truth partner stay together), so augmented copies
of one recording never end up on both sides. Evaluate on the real, un-augmented validation samples.

Usage:
    python augment_obb_dataset.py --input dataset --output dataset_aug --n-aug 12 --n-synthetic 400
"""

import argparse
import csv
import hashlib
import itertools
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

R_C2L = np.diag([1.0, -1.0, -1.0])      # camera optical frame -> LiDAR frame (same as the pipeline)

DEFAULTS = dict(
    # depth sensor model (stereo): sigma_z = z^2 * subpixel / (fx * baseline).  D435-like
    subpixel_px=0.08, baseline_m=0.05,
    # real clouds
    max_rot_deg=180.0, max_shift=0.05, fov_margin=2, tries=40,
    extra_noise_scale=0.5,
    # artifacts
    band_px=1.0, p_bleed=0.5, bleed_frac=0.5, bleed_max_px=3.0,
    p_fly=0.1, fly_frac=0.15, fly_max_m=0.06,
    p_shadow=0.5, shadow_frac_max=0.10,
    p_voxel=0.7, voxel_m=(0.004, 0.015), keep_ratio=(0.5, 1.0),
    # synthetic boxes
    syn_noise_scale=1.0, syn_scale_xy=(0.7, 1.4), syn_scale_h=(0.7, 1.3), min_dim=0.15, max_dim=0.9,
    syn_center_frac=0.7,
)


# ----------------------------------------------------------------------------------------------
# Camera rig (calibration of one recording)
# ----------------------------------------------------------------------------------------------
class Rig:
    """q (LiDAR frame) = R (p (camera frame) - t) with R = R_yaw @ R_C2L, t = LiDAR position in the camera frame."""

    def __init__(self, calib):
        K = np.asarray(calib["camera_intrinsics"], dtype=np.float64)
        self.fx, self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        e = calib["lidar_to_cam_extrinsic"]
        self.t = np.asarray(e["lidar_position"], dtype=np.float64)
        yaw = float(e["lidar_yaw"])
        c, s = np.cos(yaw), np.sin(yaw)
        self.R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R_C2L
        self.cam_center = -self.R @ self.t            # camera origin in the LiDAR frame (straight above the LiDAR)

    def to_cam(self, q):
        return (self.R.T @ np.asarray(q, dtype=np.float64).T).T + self.t

    def project(self, q):
        p = self.to_cam(q)
        z = p[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            return p[:, 0] * self.fx / z + self.cx, p[:, 1] * self.fy / z + self.cy, z

    def unproject(self, u, v, z):
        p = np.stack([(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z], 1)
        return (p - self.t) @ self.R.T

    def pixel_dirs(self, u, v):
        """LiDAR-frame ray directions; scaled so that the ray parameter t equals the depth (z of the camera frame)."""
        d = np.stack([(np.asarray(u, float) - self.cx) / self.fx, (np.asarray(v, float) - self.cy) / self.fy,
                      np.ones(len(u))], 1)
        return d @ self.R.T

    def sigma_depth(self, z, cfg):
        return z ** 2 * cfg["subpixel_px"] / (self.fx * cfg["baseline_m"])


# ----------------------------------------------------------------------------------------------
# OBB helpers (axes = columns of R, as in Open3D)
# ----------------------------------------------------------------------------------------------
def rot_z(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def obb_corners(c, d, R):
    signs = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=np.float64)
    return c + (signs * d / 2) @ R.T


def box_sdf(points, c, d, R):
    q = np.abs((points - c) @ R) - d / 2
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)


def inside_fraction(points, c, d, R, margin):
    return float(np.mean(np.all(np.abs((points - c) @ R) <= d / 2 + margin, axis=1)))


def obb_to_json(c, d, R):
    return {"center": [round(float(x), 6) for x in c], "dimensions": [round(float(x), 6) for x in d],
            "rotation": [round(float(x), 8) for x in R.reshape(-1)], "yaw_rad": round(float(np.arctan2(R[1, 0], R[0, 0])), 6)}


def get_path(doc, dotted):
    for k in dotted.split("."):
        if not isinstance(doc, dict) or k not in doc:
            return None
        doc = doc[k]
    return doc


def parse_obb(o):
    R = np.asarray(o["rotation"], dtype=np.float64).reshape(3, 3)
    return np.asarray(o["center"], dtype=np.float64), np.asarray(o["dimensions"], dtype=np.float64), R


def in_fov_fraction(rig, pts, W, H, margin):
    u, v, z = rig.project(pts)
    ok = (z > 0.2) & (u >= margin) & (u <= W - 1 - margin) & (v >= margin) & (v <= H - 1 - margin)
    return float(ok.mean())


def voxel_mean(points, size):
    keys = np.floor(points / size).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    cnt = np.bincount(inv)
    return np.stack([np.bincount(inv, weights=points[:, k]) / cnt for k in range(3)], 1)


# ----------------------------------------------------------------------------------------------
# 1. exact geometry on a real cloud
# ----------------------------------------------------------------------------------------------
def transform_real(points, obb, rig, W, H, rng, cfg):
    c0, d0, R0 = obb
    pivot = np.array([rig.cam_center[0], rig.cam_center[1], 0.0])
    base = in_fov_fraction(rig, np.vstack([points, obb_corners(c0, d0, R0)]), W, H, cfg["fov_margin"])
    for _ in range(cfg["tries"]):
        phi = np.deg2rad(rng.uniform(-cfg["max_rot_deg"], cfg["max_rot_deg"]))
        r, th = cfg["max_shift"] * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        shift = np.array([r * np.cos(th), r * np.sin(th), 0.0])
        Rz = rot_z(phi)
        P = (points - pivot) @ Rz.T + pivot + shift
        c = (c0 - pivot) @ Rz.T + pivot + shift
        R = Rz @ R0
        if in_fov_fraction(rig, np.vstack([P, obb_corners(c, d0, R)]), W, H, cfg["fov_margin"]) >= base - 1e-9:
            return P, (c, d0, R), {"rotation_deg": round(float(np.rad2deg(phi)), 2),
                                   "shift_x_cm": round(shift[0] * 100, 2), "shift_y_cm": round(shift[1] * 100, 2)}
    return None


# ----------------------------------------------------------------------------------------------
# 2. label-preserving artifacts (shared by real and synthetic clouds)
# ----------------------------------------------------------------------------------------------
def depth_noise(points, rig, scale, cfg, rng):
    u, v, z = rig.project(points)
    sig = scale * rig.sigma_depth(z, cfg)
    delta = rng.normal(0.0, 1.0, len(points)) * sig
    return points + (points - rig.cam_center) / z[:, None] * delta[:, None], float(np.median(sig) * 100)


def edge_artifacts(points, rig, rng, cfg):
    info = {}
    if len(points) < 100:
        return points, info
    u, v, z = rig.project(points)
    uv = np.stack([u, v], 1)
    try:
        eq = ConvexHull(uv).equations
    except Exception:
        return points, info
    dist = -(uv @ eq[:, :2].T + eq[:, 2])                   # >= 0 inside, distance to every hull edge in pixels
    kmin = dist.argmin(1)
    band = dist.min(1) <= cfg["band_px"]
    normal = eq[kmin, :2]                                   # outward normal of the closest silhouette edge
    extra = None
    pts = points
    if band.any() and rng.uniform() < cfg["p_bleed"]:       # depth bleeding: the silhouette grows
        sel = band & (rng.uniform(size=len(points)) < cfg["bleed_frac"])
        if sel.any():
            uv2 = uv[sel] + normal[sel] * rng.uniform(0, cfg["bleed_max_px"], sel.sum())[:, None]
            extra = rig.unproject(uv2[:, 0], uv2[:, 1], z[sel])
            info["bleed_points"] = int(sel.sum())
    if band.any() and rng.uniform() < cfg["p_fly"]:         # flying pixels: edge points smeared towards the background
        sel = band & (rng.uniform(size=len(points)) < cfg["fly_frac"])
        if sel.any():
            pts = points.copy()
            pts[sel] += (points[sel] - rig.cam_center) / z[sel][:, None] * rng.uniform(0, cfg["fly_max_m"], sel.sum())[:, None]
            info["fly_points"] = int(sel.sum())
    if rng.uniform() < cfg["p_shadow"]:                     # one-sided loss at the silhouette (stereo shadow)
        th = rng.uniform(0, 2 * np.pi)
        s = (u - u.mean()) * np.cos(th) + (v - v.mean()) * np.sin(th)
        frac = rng.uniform(0.02, cfg["shadow_frac_max"])
        keep = s <= s.max() - frac * (s.max() - s.min())
        pts = pts[keep]
        info["shadow_removed"] = int((~keep).sum())
    if extra is not None:
        pts = np.vstack([pts, extra])
    return pts, info


def density_aug(points, rng, cfg):
    info = {}
    if rng.uniform() < cfg["p_voxel"]:
        size = rng.uniform(*cfg["voxel_m"])
        points = voxel_mean(points, size)
        info["voxel_cm"] = round(size * 100, 2)
    kr = rng.uniform(*cfg["keep_ratio"])
    if kr < 0.999:
        points = points[rng.uniform(size=len(points)) < kr]
        info["keep_ratio"] = round(float(kr), 3)
    return points, info


def augment_cloud(points, rig, rng, cfg, noise_scale):
    pts, sig_cm = depth_noise(points, rig, noise_scale, cfg, rng)
    pts, info = edge_artifacts(pts, rig, rng, cfg)
    pts, info2 = density_aug(pts, rng, cfg)
    return pts, {"extra_depth_sigma_cm": round(sig_cm, 3), **info, **info2}


# ----------------------------------------------------------------------------------------------
# 3. sensor re-simulation of a box
# ----------------------------------------------------------------------------------------------
def render_box(rig, c, d, R, W, H, margin=2):
    """Clean ray casting of the box through the real pixel grid (top and visible side walls). None if the box is not
    completely inside the image. The returned points are exact surface points; noise is added by augment_cloud."""
    u, v, z = rig.project(obb_corners(c, d, R))
    if np.any(z <= 0.2):
        return None
    u0, u1, v0, v1 = int(np.floor(u.min())) - 1, int(np.ceil(u.max())) + 1, int(np.floor(v.min())) - 1, int(np.ceil(v.max())) + 1
    if u0 < margin or v0 < margin or u1 > W - 1 - margin or v1 > H - 1 - margin:
        return None
    uu, vv = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
    uu, vv = uu.ravel().astype(np.float64), vv.ravel().astype(np.float64)
    dirs = rig.pixel_dirs(uu, vv)
    o = (rig.cam_center - c) @ R
    dl = dirs @ R
    hh = d / 2
    par = np.abs(dl) < 1e-12
    ds = np.where(par, 1.0, dl)
    t1, t2 = (-hh - o) / ds, (hh - o) / ds
    inside = np.abs(o) <= hh
    tlo = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
    thi = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
    tmin, tmax = tlo.max(1), thi.min(1)
    hit = (tmax >= tmin) & (tmin > 0)
    if hit.sum() < 50:
        return None
    return rig.cam_center + tmin[hit][:, None] * dirs[hit]


def sample_synthetic(src_obb, rig, W, H, rng, cfg):
    c0, d0, R0 = src_obb
    vi = int(np.argmax(np.abs(R0[2, :])))                   # the box axis that points up
    d = d0.copy()
    for k in range(3):
        lo, hi = cfg["syn_scale_h"] if k == vi else cfg["syn_scale_xy"]
        d[k] = np.clip(d0[k] * rng.uniform(lo, hi), cfg["min_dim"], cfg["max_dim"])
    zc = (c0[2] - d0[vi] / 2) + d[vi] / 2                   # keep the bottom of the box where it was
    R = rot_z(rng.uniform(0, 2 * np.pi)) @ R0
    f = cfg["syn_center_frac"]
    u = rng.uniform(W / 2 * (1 - f), W / 2 * (1 + f))
    v = rng.uniform(H / 2 * (1 - f), H / 2 * (1 + f))
    dr = rig.pixel_dirs(np.array([u]), np.array([v]))[0]
    t = (zc - rig.cam_center[2]) / dr[2]
    xy = rig.cam_center[:2] + t * dr[:2]
    return np.array([xy[0], xy[1], zc]), d, R


# ----------------------------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------------------------
class Source:
    pass


def load_sources(root, label_key, warn):
    sources, fallback_used = [], 0
    for cat in sorted(p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith("_")):
        for rec in sorted(p for p in cat.iterdir() if p.is_dir() and re.fullmatch(r"recording_\d+", p.name)):
            jp = rec / "recording_data.json"
            if not jp.exists():
                continue
            with open(jp) as f:
                doc = json.load(f)
            raw = get_path(doc, "3d_data_camera.object_points")
            lab = get_path(doc, label_key)
            if lab is None and label_key == "obb_label":
                lab = get_path(doc, "3d_data_lidar.obb_3d")
                fallback_used += lab is not None
            if raw is None or lab is None or len(raw) == 0:
                warn(f"{cat.name}/{rec.name}: no camera points or no label, skipped")
                continue
            s = Source()
            s.category, s.name, s.doc = cat.name, rec.name, doc
            s.calib = doc["calibration"]
            s.rig = Rig(s.calib)
            s.size = tuple(int(x) for x in doc["2d_data"]["image_dimensions"])        # (H, W)
            pts = np.asarray(raw, dtype=np.float64).reshape(len(raw), -1)[:, :3]
            if np.mean(pts[:, 2] < 0) <= 0.005:                                          # camera frame -> LiDAR frame
                pts = (pts - s.rig.t) @ s.rig.R.T
            s.points, s.obb = pts, parse_obb(lab)
            n = int(re.search(r"(\d+)", rec.name).group(1))
            s.scene = f"{cat.name}/{(n + 1) // 2:02d}"
            sources.append(s)
    return sources, fallback_used


def write_sample(folder, points, obb, calib, rig, meta, fmt):
    folder.mkdir(parents=True, exist_ok=True)
    c, d, R = obb
    if fmt in ("json", "both"):
        doc = {"sample_metadata": meta,
               "3d_data_camera": {"total_point_count": int(len(points)),
                                  "object_points": np.round(points, 5).tolist()},
               "obb_label": obb_to_json(c, d, R),
               "calibration": calib}
        with open(folder / "recording_data.json", "w") as f:
            json.dump(doc, f, indent=4)
    if fmt in ("npz", "both"):
        np.savez_compressed(folder / "sample.npz", points=points.astype(np.float32), obb_center=c, obb_dimensions=d,
                            obb_rotation=R, camera_center=rig.cam_center)


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def residual_summary(name, arrays):
    if not arrays:
        return f"  {name}: -"
    r = np.abs(np.concatenate(arrays)) * 100
    return (f"  {name:<22} median {np.median(r):5.2f} cm | p90 {np.percentile(r, 90):5.2f} cm | "
            f"p99 {np.percentile(r, 99):5.2f} cm | outside the box by > 2 cm: "
            f"{np.mean(np.concatenate(arrays) > 0.02):.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="dataset root (category/recording_XX/recording_data.json)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--label-key", default="obb_label", help="dotted json path of the true OBB (default obb_label; "
                    "falls back to 3d_data_lidar.obb_3d with a warning)")
    ap.add_argument("--n-aug", type=int, default=10, help="augmented copies per real recording")
    ap.add_argument("--n-synthetic", type=int, default=0, help="re-simulated clusters (new position / yaw / size)")
    ap.add_argument("--no-originals", action="store_true", help="do not copy the un-augmented recordings")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", choices=["json", "npz", "both"], default="json")
    ap.add_argument("--max-points", type=int, default=0, help="random cap on the points per sample (0 = no cap)")
    for k, v in DEFAULTS.items():
        if isinstance(v, (int, float)):
            ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    args = ap.parse_args()
    cfg = {k: getattr(args, k) for k, v in DEFAULTS.items() if isinstance(v, (int, float))}
    cfg.update({k: v for k, v in DEFAULTS.items() if not isinstance(v, (int, float))})
    rng = np.random.default_rng(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    warnings = []
    sources, fb = load_sources(args.input, args.label_key, warnings.append)
    if not sources:
        raise SystemExit("No usable recordings found.")
    if fb:
        print(f"[WARNING] {fb} recording(s) have no '{args.label_key}': using 3d_data_lidar.obb_3d, which is computed from "
              f"the LiDAR points of the visible face and is usually NOT the real box. Label the real boxes first.")
    for w in warnings:
        print("  [warn]", w)
    print(f"{len(sources)} source recordings in {len({s.scene for s in sources})} scenes")

    scenes = sorted({s.scene for s in sources})
    perm = np.random.default_rng(args.split_seed).permutation(len(scenes))
    n_val = int(round(args.val_fraction * len(scenes)))
    val_scenes = {scenes[i] for i in perm[:n_val]}
    split_of = lambda s: "val" if s.scene in val_scenes else "train"

    manifest, real_res, syn_res, aug_res = [], [], [], []
    low_cov = []

    def emit(src, points, obb, kind, idx, info):
        if len(points) < 50:
            return False
        if args.max_points and len(points) > args.max_points:
            points = points[rng.choice(len(points), args.max_points, replace=False)]
        sid = f"{src.name}__{kind}{idx:04d}"
        meta = {"label": src.doc.get("sample_metadata", {}).get("label", "occluded boxes"), "source": f"{src.category}/{src.name}",
                "scene": src.scene, "split": split_of(src), "sample_id": sid,
                "augmentation": {"type": kind, **info}}
        write_sample(out / split_of(src) / src.category / sid, points, obb, src.calib, src.rig, meta, args.format)
        manifest.append({"sample_id": sid, "split": split_of(src), "category": src.category, "source": src.name,
                         "scene": src.scene, "type": kind, "n_points": len(points),
                         "path": str(Path(split_of(src)) / src.category / sid), **info})
        return True

    t0 = time.time()
    for s in sources:
        H, W = s.size
        c0, d0, R0 = s.obb
        frac = inside_fraction(s.points, c0, d0, R0, 0.03)
        if frac < 0.85:
            low_cov.append((f"{s.category}/{s.name}", frac))
        real_res.append(box_sdf(s.points, c0, d0, R0))
        if not args.no_originals:
            emit(s, s.points, s.obb, "orig", 0, {"inside_frac_3cm": round(frac, 3)})
        made, attempts = 0, 0
        while made < args.n_aug and attempts < 6 * max(args.n_aug, 1):
            attempts += 1
            tr = transform_real(s.points, s.obb, s.rig, W, H, rng, cfg)
            if tr is None:
                continue
            P, obb, ginfo = tr
            P, ainfo = augment_cloud(P, s.rig, rng, cfg, cfg["extra_noise_scale"])
            if emit(s, P, obb, "aug", made, {**ginfo, **ainfo}):
                aug_res.append(box_sdf(P, *obb))
                made += 1
        if made < args.n_aug:
            print(f"  [warn] {s.category}/{s.name}: only {made}/{args.n_aug} augmented copies (cluster too close to the image border)")

    if args.n_synthetic:
        pool = [s for s in sources if abs(s.obb[2][2, :]).max() > 0.99]        # gravity-aligned boxes only
        print(f"synthetic re-simulation from {len(pool)}/{len(sources)} gravity-aligned labels "
              f"(tilted boxes are skipped: the generator assumes a box standing on the floor)")
        made, attempts = 0, 0
        while pool and made < args.n_synthetic and attempts < 30 * args.n_synthetic:
            attempts += 1
            s = pool[rng.integers(len(pool))]
            H, W = s.size
            c, d, R = sample_synthetic(s.obb, s.rig, W, H, rng, cfg)
            P = render_box(s.rig, c, d, R, W, H)
            if P is None:
                continue
            P, ainfo = augment_cloud(P, s.rig, rng, cfg, cfg["syn_noise_scale"])
            if emit(s, P, (c, d, R), "syn", made, {"dims_cm": [round(x * 100, 1) for x in d],
                                                   "center_xy_m": [round(c[0], 3), round(c[1], 3)], **ainfo}):
                syn_res.append(box_sdf(P, c, d, R))
                made += 1
        print(f"synthetic samples: {made}/{args.n_synthetic} ({attempts} attempts)")

    if manifest:
        keys = list(dict.fromkeys(k for row in manifest for k in row))
        with open(out / "manifest.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(manifest)

    n_by = lambda sp, ty: sum(1 for r in manifest if r["split"] == sp and r["type"] == ty)
    print(f"\nwritten to {out.resolve()} in {time.time() - t0:.0f}s")
    print(f"{'':>8} {'orig':>6} {'aug':>6} {'syn':>6}")
    for sp in ("train", "val"):
        print(f"{sp:>8} {n_by(sp, 'orig'):>6} {n_by(sp, 'aug'):>6} {n_by(sp, 'syn'):>6}")
    print(f"validation scenes: {sorted(val_scenes)}")
    if low_cov:
        print(f"\n[label check] {len(low_cov)} recording(s) with < 85% of the camera points inside the labelled box "
              f"(+3 cm): wrong or missing label?")
        for name, fr in low_cov[:15]:
            print(f"   {name}: {fr:.0%}")
    print("\ndistance of the camera points to the labelled box surface (does the simulation look like the sensor?):")
    print(residual_summary("real recordings", real_res))
    print(residual_summary("real + augmentations", aug_res))
    print(residual_summary("synthetic", syn_res))


if __name__ == "__main__":
    main()