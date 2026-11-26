#!/usr/bin/env python3
"""Shared helpers for the occlusion-completion pipeline (stages 0-4).

Coordinate conventions (identical to the live script):
  * Everything is stored in the LiDAR frame (x forward, z up, floor at z = -LIDAR_HEIGHT).
  * The overhead camera looks straight down. Camera frame -> LiDAR frame:
        q = R (p - t),   R = R_yaw @ diag(1,-1,-1),   t = LiDAR position in the camera frame
    so the camera optical axis maps to -z of the LiDAR frame and the camera centre sits at -R t.
  * Box axes are the COLUMNS of R (Open3D convention). Faces are moved along their outward normal.
"""
from __future__ import annotations

import csv
import io
import itertools
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # minimal fallback so the scripts also run without tqdm
    _tqdm = None

R_C2L = np.diag([1.0, -1.0, -1.0])


# ----------------------------------------------------------------------------------------------
# progress / logging / gates
# ----------------------------------------------------------------------------------------------
class _SimpleProgress:
    def __init__(self, it, total, desc):
        self.it = it
        self.total = total if total is not None else (len(it) if hasattr(it, "__len__") else None)
        self.desc = desc
        self.t0 = time.time()

    def __iter__(self):
        for i, x in enumerate(self.it, 1):
            yield x
            if self.total and (i == self.total or i % max(1, self.total // 20) == 0):
                el = time.time() - self.t0
                print(f"  {self.desc}: {i}/{self.total} ({100 * i / self.total:.0f}%)  "
                      f"elapsed {el:.0f}s  eta {el / i * (self.total - i):.0f}s", flush=True)

    def set_postfix_str(self, s, refresh=True):
        pass


def progress(iterable, total=None, desc=""):
    if _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    return _SimpleProgress(iterable, total, desc)


def banner(stage, title):
    print(f"\n{'=' * 78}\nSTAGE {stage}: {title}\n{'=' * 78}", flush=True)


def step(msg):
    print(f"  -> {msg}", flush=True)


class Gates:
    """Sanity gates: a failed HARD gate stops the pipeline (exit code 2), a failed soft gate only warns."""

    def __init__(self, stage):
        self.stage, self.results = stage, []

    def check(self, name, ok, detail="", hard=True):
        tag = "PASS" if ok else ("FAIL" if hard else "WARN")
        print(f"  [GATE {tag}] {name}: {detail}", flush=True)
        self.results.append((name, bool(ok), hard))
        return ok

    def finish(self):
        failed = [n for n, ok, hard in self.results if not ok and hard]
        if failed:
            print(f"\n[STAGE {self.stage}] FAILED hard gates: {failed}\n"
                  f"Fix this before running the next stage.", flush=True)
            sys.exit(2)
        warns = [n for n, ok, hard in self.results if not ok and not hard]
        print(f"\n[STAGE {self.stage}] hard gates passed" + (f" (warnings: {warns})" if warns else "") + ".",
              flush=True)


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------------------------
# geometry
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


def region_mask(points, box, half):
    """Points inside the box frame region |local| <= half (half: (3,) half extents about the box centre)."""
    c, _, R = box
    return np.all(np.abs((points - c) @ R) <= half, axis=1)


def ray_box_interval(o, d, lo, hi):
    """Slab test of rays o + t d (o: (3,), d: (N,3)) against the axis-aligned box [lo, hi]. -> tin, tout (N,)."""
    tin = np.full(len(d), -np.inf)
    tout = np.full(len(d), np.inf)
    for k in range(3):
        dk = d[:, k]
        par = np.abs(dk) < 1e-12
        ds = np.where(par, 1.0, dk)
        t1, t2 = (lo[k] - o[k]) / ds, (hi[k] - o[k]) / ds
        a, b = np.minimum(t1, t2), np.maximum(t1, t2)
        inside = bool(lo[k] <= o[k] <= hi[k])
        a = np.where(par, -np.inf if inside else np.inf, a)
        b = np.where(par, np.inf if inside else -np.inf, b)
        tin, tout = np.maximum(tin, a), np.minimum(tout, b)
    return tin, tout


def min_area_rect(xy):
    """Minimum-area bounding rectangle of 2D points (same result as cv2.minAreaRect). -> corners (4,2) in loop order."""
    h = xy[ConvexHull(xy).vertices]
    edges = np.roll(h, -1, 0) - h
    best, best_area = None, np.inf
    for a in np.arctan2(edges[:, 1], edges[:, 0]):
        u, w = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        pu, pw = h @ u, h @ w
        area = np.ptp(pu) * np.ptp(pw)
        if area < best_area:
            best_area, best = area, (u, w, pu.min(), pu.max(), pw.min(), pw.max())
    u, w, u0, u1, w0, w1 = best
    return np.array([u * u0 + w * w0, u * u1 + w * w0, u * u1 + w * w1, u * u0 + w * w1])


# ----------------------------------------------------------------------------------------------
# camera rig
# ----------------------------------------------------------------------------------------------
class Rig:
    """q (LiDAR frame) = R (p (camera frame) - t);  R = R_yaw @ R_C2L;  t = LiDAR position in the camera frame."""

    def __init__(self, K, R, t, hw):
        self.K, self.R, self.t = np.asarray(K, float), np.asarray(R, float), np.asarray(t, float)
        self.hw = tuple(int(x) for x in hw)                           # (H, W)
        self.fx, self.fy, self.cx, self.cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        self.cam_center = -self.R @ self.t                            # camera origin in the LiDAR frame

    @classmethod
    def from_calib(cls, calib, hw):
        e = calib["lidar_to_cam_extrinsic"]
        R = rot_z(float(e["lidar_yaw"])) @ R_C2L
        return cls(calib["camera_intrinsics"], R, e["lidar_position"], hw)

    def to_cam(self, q):
        return (self.R.T @ np.asarray(q, float).T).T + self.t

    def project(self, q):
        p = self.to_cam(q)
        z = p[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            return p[:, 0] * self.fx / z + self.cx, p[:, 1] * self.fy / z + self.cy, z

    def unproject(self, u, v, z):
        p = np.stack([(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z], 1)
        return (p - self.t) @ self.R.T

    def pixel_dirs(self, u, v):
        d = np.stack([(np.asarray(u, float) - self.cx) / self.fx, (np.asarray(v, float) - self.cy) / self.fy,
                      np.ones(len(u))], 1)
        return d @ self.R.T

    def sigma_depth(self, z, cfg):
        return z ** 2 * cfg["subpixel_px"] / (self.fx * cfg["baseline_m"])


# ----------------------------------------------------------------------------------------------
# camera-cluster artifacts (taken over unchanged from augment_obb_dataset.py: they only touch the CAMERA cloud)
# ----------------------------------------------------------------------------------------------
ART_DEFAULTS = dict(
    subpixel_px=0.08, baseline_m=0.05,
    extra_noise_scale=0.5,
    band_px=1.0, p_bleed=0.5, bleed_frac=0.5, bleed_max_px=3.0,
    p_fly=0.1, fly_frac=0.15, fly_max_m=0.06,
    p_shadow=0.5, shadow_frac_max=0.10,
    p_voxel=0.7, voxel_m=(0.004, 0.015), keep_ratio=(0.5, 1.0),
)


def voxel_mean(points, size):
    keys = np.floor(points / size).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    cnt = np.bincount(inv)
    return np.stack([np.bincount(inv, weights=points[:, k]) / cnt for k in range(3)], 1)


def in_fov_fraction(rig, pts, margin):
    H, W = rig.hw
    u, v, z = rig.project(pts)
    ok = (z > 0.2) & (u >= margin) & (u <= W - 1 - margin) & (v >= margin) & (v <= H - 1 - margin)
    return float(ok.mean())


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
    dist = -(uv @ eq[:, :2].T + eq[:, 2])
    kmin = dist.argmin(1)
    band = dist.min(1) <= cfg["band_px"]
    normal = eq[kmin, :2]
    extra, pts = None, points
    if band.any() and rng.uniform() < cfg["p_bleed"]:
        sel = band & (rng.uniform(size=len(points)) < cfg["bleed_frac"])
        if sel.any():
            uv2 = uv[sel] + normal[sel] * rng.uniform(0, cfg["bleed_max_px"], sel.sum())[:, None]
            extra = rig.unproject(uv2[:, 0], uv2[:, 1], z[sel])
            info["bleed_points"] = int(sel.sum())
    if band.any() and rng.uniform() < cfg["p_fly"]:
        sel = band & (rng.uniform(size=len(points)) < cfg["fly_frac"])
        if sel.any():
            pts = points.copy()
            pts[sel] += (points[sel] - rig.cam_center) / z[sel][:, None] * rng.uniform(0, cfg["fly_max_m"], sel.sum())[:, None]
            info["fly_points"] = int(sel.sum())
    if rng.uniform() < cfg["p_shadow"]:
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
# dataset discovery and file IO
# ----------------------------------------------------------------------------------------------
@dataclass
class Pair:
    category: str
    pair_no: int            # 1 for recordings (01, 02), 2 for (03, 04), ...
    odd_dir: Path           # occluded scene
    even_dir: Path          # ground truth (occluder removed)

    @property
    def scene(self):
        return f"{self.category}/{self.pair_no:02d}"

    @property
    def slug(self):
        return f"{slugify(self.category)}_{self.pair_no:02d}"


def slugify(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def find_pairs(root):
    pairs = []
    for cat in sorted(p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
        recs = {}
        for p in cat.iterdir():
            m = re.fullmatch(r"recording_(\d+)", p.name)
            if p.is_dir() and m:
                recs[int(m.group(1))] = p
        for n in sorted(recs):
            if n % 2 == 1 and (n + 1) in recs:
                pairs.append(Pair(cat.name, (n + 1) // 2, recs[n], recs[n + 1]))
            elif n % 2 == 1:
                print(f"  [warn] {cat.name}/{recs[n].name}: no even partner recording_{n + 1:02d}, skipped")
    return pairs


def split_scenes(scenes, val_fraction, seed):
    scenes = sorted(scenes)
    perm = np.random.default_rng(seed).permutation(len(scenes))
    n_val = int(round(val_fraction * len(scenes)))
    return {scenes[i] for i in perm[:n_val]}


_PCD_CODE = {"F": "f", "U": "u", "I": "i"}


def read_pcd_xyz(path):
    """x,y,z of a .pcd file (ascii, binary or binary_compressed) with a pure-python parser.
    open3d is only the fallback: on Windows it cannot open paths with non-ASCII characters (e.g. 'Davi\u0301d') and its
    python binding then raises UnicodeDecodeError. NaN / inf points are dropped."""
    try:
        pts = _read_pcd_py(path)
    except Exception as e_py:
        try:
            import open3d as o3d
            pts = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float64)
            if len(pts) == 0:
                raise RuntimeError("open3d returned 0 points")
        except Exception as e_o3d:
            raise RuntimeError(f"cannot read {path}: own parser: {e_py!r}; open3d: {e_o3d!r}") from None
    return pts[np.isfinite(pts).all(1)]


def _lzf_decompress(data, out_len):
    """liblzf decompression (PCD 'binary_compressed')."""
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        ctrl = data[i]
        i += 1
        if ctrl < 32:                                            # literal run
            out += data[i:i + ctrl + 1]
            i += ctrl + 1
        else:                                                    # back reference (may overlap itself)
            length = ctrl >> 5
            if length == 7:
                length += data[i]
                i += 1
            ref = len(out) - ((ctrl & 0x1F) << 8) - data[i] - 1
            i += 1
            for _ in range(length + 2):
                out.append(out[ref])
                ref += 1
    if len(out) != out_len:
        raise RuntimeError(f"LZF: got {len(out)} bytes, expected {out_len}")
    return bytes(out)


def _read_pcd_py(path):
    raw = Path(path).read_bytes()                                # python handles unicode paths on every OS
    header, pos = {}, 0
    while True:
        nl = raw.index(b"\n", pos)
        line = raw[pos:nl].decode("ascii", "ignore").strip()
        pos = nl + 1
        if not line or line.startswith("#"):
            continue
        key, *val = line.split()
        header[key.upper()] = val
        if key.upper() == "DATA":
            break
    fields, n = header["FIELDS"], int(header["POINTS"][0])
    size = [int(x) for x in header["SIZE"]]
    typ = header["TYPE"]
    count = [int(x) for x in header.get("COUNT", ["1"] * len(fields))]
    mode = header["DATA"][0]
    for f in ("x", "y", "z"):
        if f not in fields:
            raise RuntimeError(f"{path}: no '{f}' field (FIELDS = {fields})")
    ix = [fields.index(f) for f in ("x", "y", "z")]
    if mode == "ascii":
        arr = np.loadtxt(io.BytesIO(raw[pos:]), ndmin=2)
        col, cols = 0, []
        for c in count:
            cols.append(col)
            col += c
        return arr[:n, [cols[i] for i in ix]].astype(np.float64)
    if mode == "binary":
        dt = np.dtype([(f"f{k}", f"<{_PCD_CODE[t]}{s}", (c,) if c > 1 else ())
                       for k, (t, s, c) in enumerate(zip(typ, size, count))])
        arr = np.frombuffer(raw, dtype=dt, count=n, offset=pos)
        return np.stack([arr[f"f{i}"].reshape(n, -1)[:, 0] for i in ix], 1).astype(np.float64)
    if mode == "binary_compressed":                              # field-major layout after LZF decompression
        comp, unc = np.frombuffer(raw, dtype="<u4", count=2, offset=pos)
        buf = _lzf_decompress(raw[pos + 8:pos + 8 + int(comp)], int(unc))
        off, cols = 0, {}
        for k, (t, s, c) in enumerate(zip(typ, size, count)):
            cols[k] = np.frombuffer(buf, dtype=f"<{_PCD_CODE[t]}{s}", count=n * c, offset=off).reshape(n, c)[:, 0]
            off += n * s * c
        return np.stack([cols[i] for i in ix], 1).astype(np.float64)
    raise RuntimeError(f"{path}: unknown PCD DATA mode {mode}")


def write_pcd_ascii(path, pts, extra=None):
    pts = np.asarray(pts, float)
    cols = pts if extra is None else np.hstack([pts, np.asarray(extra, float).reshape(len(pts), -1)])
    fields = "x y z" + (" intensity" if extra is not None else "")
    n = len(pts)
    with open(path, "w") as f:
        f.write(f"# .PCD v0.7\nVERSION 0.7\nFIELDS {fields}\nSIZE {' '.join(['4'] * cols.shape[1])}\n"
                f"TYPE {' '.join(['F'] * cols.shape[1])}\nCOUNT {' '.join(['1'] * cols.shape[1])}\n"
                f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA ascii\n")
        np.savetxt(f, cols, fmt="%.6f")