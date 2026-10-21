#!/usr/bin/env python3
"""
Offline evaluation of the LiDAR occlusion-completion pipeline on pre-recorded scenes.

The pipeline functions (get_boxes_3d, ray_intersect_triangles_batch, project_3d_to_2d) and the
constant VOXEL_SIZE are NOT copied: they are pulled straight
out of your live script with `ast`, so the algorithm stays exactly the one you run live and this
file never opens the socket / ROS node of the original script.

Dataset layout expected:
    dataset/<category>/recording_01/{recording_data.json, image_overhead.png, pointcloud_lidar.pcd, pointcloud_camera.pcd}
    dataset/<category>/recording_02/...      (odd = occluded, even = ground truth of the previous odd one)

Metrics (for every voxel size in TOLERANCES_CM, each with its own distance tolerance):
  precision  fraction of predicted voxels that have a GT voxel within the tolerance
  recall     fraction of GT voxels that have a predicted voxel within the tolerance
  hits / misses / false positives   the same decision in counts
  distances  mean / max distance prediction -> GT and GT -> prediction (no threshold involved)
The old exact-voxel ("strict") numbers are still written to the CSV (columns strict_*) but not charted.

Every scene can have its own calibration offset: SCENE_OFFSETS below (dial them in visualize_dataset.py,
press P there and paste the printed dictionary).

Usage:
    python evaluate_dataset.py --dataset dataset --pipeline your_live_script.py --output evaluation_results
    python evaluate_dataset.py --voxel-sizes 1 3 --tolerances 2 3      # one tolerance per voxel size

Extra dependency (besides the ones your live script already needs):  pip install matplotlib
"""

import argparse
import ast
import csv
import json
import re
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")  # headless: charts are only saved, never shown
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------------------------
DATASET_DIR = "../dataset"
PIPELINE_SOURCE = "lidar-ray-extension-voxelized.py"   # <-- file name of your live script (the pasted code)
OUTPUT_DIR = "evaluation_results"

# Frame of pointcloud_camera.pcd:
#   "auto"   -> camera frame has only positive z (depth); a LiDAR-frame cloud has many z < 0 (boxes/floor
#               lie below the sensor). The detected frame is printed for every scene.
#   "lidar"  -> already in the LiDAR frame (what the live script shows in blue); converted back to the
#               camera frame, because the live pipeline works on camera-frame points.
#   "camera" -> raw camera-frame points, used as they are.
CAMERA_PCD_FRAME = "auto"

# Recording rig geometry -- taken from the RECORDING script (not from the live script, whose 0.65 m /
# marker-offset values belong to another setup).  The json lidar_position is already the LiDAR centre
# (marker + rotated offset), so the floor lies LIDAR_HEIGHT further from the camera than the LiDAR:
#     floor depth (camera z) = lidar_pos[2] + LIDAR_HEIGHT        (= ceiling_height = 2.65 m)
#     ground threshold       = lidar_pos[2] + LIDAR_HEIGHT - GROUND_MARGIN
LIDAR_HEIGHT = 0.4      # LiDAR centre above the floor (m)
GROUND_MARGIN = 0.08    # camera points deeper than (floor - margin) are ground

# Manual calibration offsets (metres, LiDAR frame: x forward, y left, z up). Positive values move the
# prediction towards +x / +y / +z.
#   SCENE_OFFSETS  one offset per scene, keyed "<category folder>/<occluded recording folder>".
#                  Dial them in visualize_dataset.py, press P there and paste the printed dictionary here.
#   CALIB_OFFSET   used for every scene that has no entry in SCENE_OFFSETS.
#   mode "geometry"   -> added to the camera-derived box walls BEFORE the ray casting. This is what a camera->LiDAR
#                        extrinsic error really is, so the hit points move exactly as they would after a re-calibration.
#   mode "prediction" -> added to the predicted points AFTER the ray casting (simple translation of the result).
CALIB_OFFSET = (0.07, 0.0, -0.03)
SCENE_OFFSETS = {
    "edgecase occlusion boxes/recording_01": (0.050, 0.000, -0.030),
    "edgecase occlusion boxes/recording_03": (0.030, 0.000, -0.030),
    "edgecase occlusion boxes/recording_05": (0.040, -0.010, -0.010),
    "edgecase occlusion boxes/recording_07": (0.040, 0.000, -0.030),
    "edgecase occlusion boxes/recording_09": (0.030, 0.000, -0.030),
    "edgecase occlusion boxes/recording_11": (0.050, 0.000, -0.030),
    "full occlusion boxes/recording_01": (0.040, -0.010, -0.030),
    "full occlusion boxes/recording_03": (0.040, -0.010, -0.030),
    "full occlusion boxes/recording_05": (0.010, -0.020, -0.030),
    "full occlusion boxes/recording_07": (0.040, 0.000, -0.040),
    "full occlusion boxes/recording_09": (0.000, -0.010, -0.030),
    "full occlusion boxes/recording_11": (-0.020, -0.030, -0.110),
    "multi occlusion boxes/recording_01": (0.030, -0.010, -0.030),
    "multi occlusion boxes/recording_03": (0.040, 0.000, -0.030),
    "multi occlusion boxes/recording_05": (0.000, -0.030, -0.060),
    "multi occlusion boxes/recording_07": (0.010, 0.000, -0.030),
    "multi occlusion boxes/recording_09": (0.030, 0.000, -0.030),
    # "multi occlusion boxes/recording_11": (0.070, 0.000, -0.030),
    # "multi occlusion boxes/recording_13": (0.070, 0.000, -0.030),
    "partial occlusion boxes/recording_01": (0.020, -0.040, -0.030),
    "partial occlusion boxes/recording_03": (0.010, -0.060, -0.030),
    "partial occlusion boxes/recording_05": (0.030, -0.040, -0.030),
    "partial occlusion boxes/recording_07": (0.060, -0.020, -0.040),
    # "partial occlusion boxes/recording_09": (0.070, 0.000, -0.030),
    "partial occlusion boxes/recording_11": (0.040, 0.000, -0.060),
}
CALIB_OFFSET_MODE = "geometry"

MIN_POINTS_PER_VOXEL = 2   # minimum number of ray-hit points in a voxel to keep it
NOISE_MAGNITUDE = 0.01     # random noise along the triangle normals
NOISE_SEED = 0             # seed of that noise so that runs / voxel sizes are reproducible (None = new noise every run)
MAX_LIDAR_RANGE = 5.0      # same 5 m cut as in pointcloud_callback of the live script
RAY_CHUNK = 20000          # rays are independent -> processed in chunks to limit RAM (no effect on results)
SAVE_DEBUG_PLOTS = True    # one overlay + top-down figure per scene (for the FIRST voxel size)

# Voxel size (cm) -> distance tolerance (cm) used for that voxel size. The keys also define which voxel sizes
# are evaluated. Rule used here: one voxel of slack, but never less than 2 cm (about the ring spacing of the LiDAR
# at ~1.4 m, below which two scans of the same surface cannot be expected to agree).
TOLERANCES_CM = {1: 1.0, 2: 2.0, 3: 3.0, 5: 5.0, 10: 10.0}

PIPELINE_FUNCS = ["get_boxes_3d", "ray_intersect_triangles_batch", "project_3d_to_2d"]
PIPELINE_CONSTS = ["VOXEL_SIZE"]


def scene_key(category, scene_name):
    """Key of a scene in SCENE_OFFSETS, e.g. 'full occlusion boxes/recording_01'."""
    return f"{category}/{scene_name}"


def get_scene_offset(key):
    return tuple(SCENE_OFFSETS.get(key, CALIB_OFFSET))


# ----------------------------------------------------------------------------------------------
# Load the pipeline functions / constants from the live script without executing it
# ----------------------------------------------------------------------------------------------
def load_pipeline(source_path):
    path = Path(source_path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in PIPELINE_FUNCS]
    missing = set(PIPELINE_FUNCS) - {f.name for f in funcs}
    if missing:
        raise RuntimeError(f"Could not find {sorted(missing)} in {path}")

    consts = {}
    for n in tree.body:
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in PIPELINE_CONSTS):
            consts[n.targets[0].id] = ast.literal_eval(n.value)
    missing = set(PIPELINE_CONSTS) - set(consts)
    if missing:
        raise RuntimeError(f"Could not find constants {sorted(missing)} in {path}")

    namespace = {"np": np, "cv2": cv2, "o3d": o3d}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in PIPELINE_FUNCS}, consts


# ----------------------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------------------
def as_points(arr):
    """(N,3+) list -> (N,3) float64, tolerant to empty input."""
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return np.zeros((0, 3))
    return a.reshape(len(a), -1)[:, :3]


def filter_dense_voxels(points, voxel_size):
    """Keep only the points that lie in a voxel with >= MIN_POINTS_PER_VOXEL points."""
    if len(points) == 0:
        return points
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)
    _, inverse_indices, counts = np.unique(voxel_coords, axis=0, return_inverse=True, return_counts=True)
    inverse_indices = inverse_indices.reshape(-1)
    return points[counts[inverse_indices] >= MIN_POINTS_PER_VOXEL]


def load_camera_pcd(rec_dir, lidar_pos, lidar_yaw_rad):
    """pointcloud_camera.pcd -> (Nx3 points in the CAMERA frame, detected frame name)."""
    path = Path(rec_dir) / "pointcloud_camera.pcd"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    pts = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return pts, "empty"

    frame = CAMERA_PCD_FRAME
    if frame == "auto":
        frame = "lidar" if np.mean(pts[:, 2] < 0) > 0.005 else "camera"

    if frame == "lidar":
        R_c2l = np.diag([1.0, -1.0, -1.0])
        R_yaw = np.array([[np.cos(lidar_yaw_rad), -np.sin(lidar_yaw_rad), 0],
                          [np.sin(lidar_yaw_rad),  np.cos(lidar_yaw_rad), 0],
                          [0, 0, 1]], dtype=np.float64)
        # inverse of: q = R_yaw @ R_c2l @ (p - lidar_pos)
        pts = (R_c2l.T @ (R_yaw.T @ pts.T)).T + lidar_pos
    return pts, frame


def load_scene(rec_dir):
    """Everything the pipeline needs for one OCCLUDED recording."""
    rec_dir = Path(rec_dir)
    with open(rec_dir / "recording_data.json", "r") as f:
        meta = json.load(f)

    pcd = o3d.io.read_point_cloud(str(rec_dir / "pointcloud_lidar.pcd"))
    lidar = np.asarray(pcd.points, dtype=np.float32)          # float32 like the live callback
    lidar = lidar[np.isfinite(lidar).all(axis=1)]

    image = cv2.imread(str(rec_dir / "image_overhead.png"))   # None if missing
    if image is not None:
        h, w = image.shape[:2]
    else:
        h, w = meta["2d_data"]["image_dimensions"]

    K = meta["calibration"]["camera_intrinsics"]
    extr = meta["calibration"]["lidar_to_cam_extrinsic"]
    lidar_pos = np.asarray(extr["lidar_position"], dtype=np.float64)   # in CAMERA frame (ArUco result)
    lidar_yaw = float(extr["lidar_yaw"])

    camera_points, camera_frame = load_camera_pcd(rec_dir, lidar_pos, lidar_yaw)

    return {
        "lidar_points": lidar,
        "camera_points": camera_points,
        "camera_frame": camera_frame,
        "image": image,
        "h": int(h),
        "w": int(w),
        "intrinsics": {"fx": K[0][0], "fy": K[1][1], "ppx": K[0][2], "ppy": K[1][2]},
        "lidar_pos": lidar_pos,
        "lidar_yaw_rad": lidar_yaw,
    }


def load_gt_points(rec_dir, voxel_size=None):
    """Ground truth = 3d lidar object points (x, y, z, [intensity]) of the even recording.
    With a voxel_size, voxels holding fewer than MIN_POINTS_PER_VOXEL points are dropped (as before);
    without it the raw points are returned (the sweep filters per voxel size itself)."""
    with open(Path(rec_dir) / "recording_data.json", "r") as f:
        meta = json.load(f)

    gt_points = as_points(meta["3d_data_lidar"]["object_points"])
    return filter_dense_voxels(gt_points, voxel_size) if voxel_size else gt_points


def find_pairs(category_dir):
    recs = {}
    for d in category_dir.iterdir():
        m = re.fullmatch(r"recording_(\d+)", d.name)
        if m and d.is_dir():
            recs[int(m.group(1))] = d
    pairs = []
    for n in sorted(recs):
        if n % 2 == 1:
            if n + 1 in recs:
                pairs.append((recs[n], recs[n + 1]))
            else:
                print(f"  [warn] {recs[n].name} has no ground-truth partner (recording_{n + 1:02d}), skipped")
    return pairs


# ----------------------------------------------------------------------------------------------
# The pipeline (mirrors the body of your MAIN LOOP step by step)
#   compute_hits()    everything up to the noisy ray hits -> does NOT depend on the voxel size
#   voxelize_hits()   density filter + voxel down-sampling  -> depends on the voxel size
#   run_pipeline()    both stages in one call (used by the viewer)
# ----------------------------------------------------------------------------------------------
def _resolve_offset(offset, offset_mode):
    offset = np.asarray(CALIB_OFFSET if offset is None else offset, dtype=np.float64)
    offset_mode = offset_mode or CALIB_OFFSET_MODE
    if offset_mode not in ("geometry", "prediction"):
        raise ValueError(f"offset_mode must be 'geometry' or 'prediction', got {offset_mode!r}")
    return offset, offset_mode


def compute_hits(scene, funcs, offset=None, offset_mode=None, cached_boxes=None):
    """offset = the calibration offset of THIS scene (see SCENE_OFFSETS).
    cached_boxes=(box_tops, sidewall_tris) skips the box detection (it does not depend on the offset);
    the viewer uses this to re-run only the ray casting while you nudge the offset."""
    offset, offset_mode = _resolve_offset(offset, offset_mode)

    get_boxes_3d = funcs["get_boxes_3d"]
    ray_intersect = funcs["ray_intersect_triangles_batch"]

    lidar_pos = scene["lidar_pos"]
    lidar_yaw_rad = scene["lidar_yaw_rad"]
    intrinsics = scene["intrinsics"]
    h, w = scene["h"], scene["w"]

    # Coordinate frame rotation (camera OpenCV -> LiDAR ROS)  [same as live]
    R_cam_to_lidar = np.array([[1, 0, 0],
                               [0, -1, 0],
                               [0, 0, -1]], dtype=np.float64)
    # Yaw rotation in LiDAR/world frame (from ArUco marker)   [same as live]
    R_yaw = np.array([[np.cos(lidar_yaw_rad), -np.sin(lidar_yaw_rad), 0],
                      [np.sin(lidar_yaw_rad),  np.cos(lidar_yaw_rad), 0],
                      [0, 0, 1]], dtype=np.float64)

    result = {"pred_points": np.zeros((0, 3)), "box_tops": [], "sidewall_tris": [],
              "camera_points": 0, "raw_hits": 0,
              "camera_points_used": np.zeros((0, 3)), "raw_hit_points": np.zeros((0, 3)),
              "hit_points": np.zeros((0, 3))}

    # --- camera points in CAMERA frame (already converted in load_scene) ---
    points = scene["camera_points"]

    # Same filters as live: valid depth, < 5 m, remove ground plane
    points = points[(points[:, 2] > 0) & (points[:, 2] < 5.0)]
    if lidar_pos[2] > 0:
        ground_threshold = lidar_pos[2] + LIDAR_HEIGHT - GROUND_MARGIN
        points = points[points[:, 2] < ground_threshold]
    if len(points) == 0:
        return result

    result["camera_points"] = len(points)
    result["camera_points_used"] = points      # camera frame, after the same filters as live

    # --- LIDAR RAY EXTENSION LOGIC ---
    if cached_boxes is None:
        raw_camera_pcd = o3d.geometry.PointCloud()
        raw_camera_pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        box_tops, sidewall_tris = get_boxes_3d(
            raw_camera_pcd, intrinsics, h, w,
            floor_z=lidar_pos[2] + LIDAR_HEIGHT
        )
    else:
        box_tops, sidewall_tris = cached_boxes

    # Filter out triangles / box tops that are too close to the LiDAR
    sidewall_tris = [t for t in sidewall_tris
                     if np.linalg.norm(np.mean(t, axis=0) - np.array(lidar_pos)) > 0.5]
    box_tops = [b for b in box_tops
                if np.linalg.norm(np.mean(b, axis=0) - np.array(lidar_pos)) > 0.5]
    result["box_tops"], result["sidewall_tris"] = box_tops, sidewall_tris

    if len(sidewall_tris) == 0:
        return result

    triangles_cam = np.array(sidewall_tris, dtype=np.float64)  # (M, 3, 3)

    # Transform triangles from camera space to world (LiDAR) space
    M_count = triangles_cam.shape[0]
    tris_flat = triangles_cam.reshape(-1, 3)
    tris_flat = (R_cam_to_lidar @ tris_flat.T).T
    lidar_pos_transformed = R_cam_to_lidar @ np.array(lidar_pos)
    tris_flat = tris_flat - lidar_pos_transformed
    tris_flat = (R_yaw @ tris_flat.T).T
    triangles_world = tris_flat.reshape(M_count, 3, 3)
    if offset_mode == "geometry":
        triangles_world = triangles_world + offset      # manual calibration correction of the box walls

    # LiDAR scene: same >5 m cut as pointcloud_callback
    pts = scene["lidar_points"]
    pts = pts[np.linalg.norm(pts, axis=1) < MAX_LIDAR_RANGE]
    if len(pts) == 0:
        return result

    origins = np.zeros_like(pts)
    vecs = pts - origins
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    valid_rays = norms[:, 0] > 0
    directions = np.where(valid_rays[:, None], vecs / np.where(norms > 0, norms, 1), 0)

    valid_idx = np.where(valid_rays)[0]
    hit_chunks = []
    tri_idx_chunks = []

    for s in range(0, len(valid_idx), RAY_CHUNK):
        sel = valid_idx[s:s + RAY_CHUNK]
        hp, _, tri_idx = ray_intersect(origins[sel], directions[sel], np.array(triangles_world), ignore_first=True)
        
        if len(hp) > 0:
            hit_chunks.append(hp)
            tri_idx_chunks.append(tri_idx)

    if not hit_chunks:
        return result

    hit_points = np.concatenate(hit_chunks, axis=0)
    hit_tri_indices = np.concatenate(tri_idx_chunks, axis=0)

    # Add random noise along the triangle normals
    tri_normals = np.cross(triangles_world[:, 1] - triangles_world[:, 0],
                           triangles_world[:, 2] - triangles_world[:, 0])

    norms = np.linalg.norm(tri_normals, axis=1, keepdims=True)
    tri_normals = np.divide(tri_normals, norms, out=np.zeros_like(tri_normals), where=norms > 0)
    normals_at_hits = tri_normals[hit_tri_indices]

    rng = np.random.default_rng(NOISE_SEED)
    random_offsets = rng.uniform(-NOISE_MAGNITUDE, NOISE_MAGNITUDE, size=(len(hit_points), 1))

    hit_points = hit_points + normals_at_hits * random_offsets

    result["raw_hits"] = len(hit_points)
    result["raw_hit_points"] = hit_points      # LiDAR frame, before voxel down-sampling
    result["hit_points"] = hit_points
    return result


def voxelize_hits(hit_points, voxel_size, offset=None, offset_mode=None):
    """Density filter + voxel down-sample (like the 'green' extended cloud the live 'o' key records)."""
    offset, offset_mode = _resolve_offset(offset, offset_mode)

    filtered_hit_points = filter_dense_voxels(hit_points, voxel_size)

    if len(filtered_hit_points) == 0:
        pred = np.empty((0, 3))
    else:
        extended_pcd = o3d.geometry.PointCloud()
        extended_pcd.points = o3d.utility.Vector3dVector(filtered_hit_points.astype(np.float64))
        voxelized_ext = extended_pcd.voxel_down_sample(voxel_size=voxel_size)
        pred = np.asarray(voxelized_ext.points)

    if offset_mode == "prediction":
        pred = pred + offset                            # manual calibration correction of the result
    return pred


def run_pipeline(scene, funcs, consts, offset=None, offset_mode=None, cached_boxes=None, voxel_size=None):
    """Both stages in one call. voxel_size defaults to VOXEL_SIZE of the live script."""
    offset, offset_mode = _resolve_offset(offset, offset_mode)
    voxel_size = voxel_size or consts["VOXEL_SIZE"]

    result = compute_hits(scene, funcs, offset, offset_mode, cached_boxes)
    if len(result["hit_points"]) > 0:
        result["pred_points"] = voxelize_hits(result["hit_points"], voxel_size, offset, offset_mode)
        if offset_mode == "prediction":
            result["raw_hit_points"] = result["hit_points"] + offset
    return result


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def voxel_centroids(points, voxel_size):
    """One point per occupied voxel: the centroid of the points inside it."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros((0, 3))
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    counts = np.bincount(inv)
    return np.stack([np.bincount(inv, weights=points[:, k]) / counts for k in range(3)], axis=1)


def evaluate(pred_pts, gt_pts, voxel_size, tolerance_cm):
    """Returns (metrics dict, (hit_keys, false_positive_keys, missed_keys)).
    The key sets are voxel indices: predicted voxels with a GT voxel within the tolerance (hits), predicted voxels
    without one (false positives) and GT voxels without a predicted voxel within the tolerance (misses)."""

    # ---------- strict: exact voxel match (only kept as a reference in the CSV) ----------
    def to_keys(p):
        if len(p) == 0:
            return set()
        return set(map(tuple, np.floor(np.asarray(p) / voxel_size).astype(int)))

    pred_keys, gt_keys = to_keys(pred_pts), to_keys(gt_pts)
    s_tp, s_fp, s_fn = len(pred_keys & gt_keys), len(pred_keys - gt_keys), len(gt_keys - pred_keys)
    s_prec = s_tp / len(pred_keys) if pred_keys else 0.0
    s_rec = s_tp / len(gt_keys) if gt_keys else 0.0
    s_f1 = 2 * s_prec * s_rec / (s_prec + s_rec) if (s_prec + s_rec) > 0 else 0.0
    if not (pred_keys - gt_keys):
        s_avg = s_max = 0.0
    elif gt_keys:
        gt_c = (np.array(list(gt_keys)) + 0.5) * voxel_size
        fp_c = (np.array(list(pred_keys - gt_keys)) + 0.5) * voxel_size
        d = cKDTree(gt_c).query(fp_c, k=1)[0]
        s_avg, s_max = float(d.mean()), float(d.max())
    else:
        s_avg = s_max = float("nan")

    # ---------- tolerant: distance-based precision / recall on the voxel centroids ----------
    pc = voxel_centroids(pred_pts, voxel_size)     # predicted voxels
    gc = voxel_centroids(gt_pts, voxel_size)       # GT voxels
    if len(pc) and len(gc):
        d_pg = cKDTree(gc).query(pc, k=1)[0]       # every predicted voxel -> nearest GT voxel
        d_gp = cKDTree(pc).query(gc, k=1)[0]       # every GT voxel -> nearest predicted voxel
        dist = {"p2g_mean_cm": float(d_pg.mean()) * 100, "p2g_max_cm": float(d_pg.max()) * 100,
                "g2p_mean_cm": float(d_gp.mean()) * 100, "g2p_max_cm": float(d_gp.max()) * 100}
    else:
        d_pg, d_gp = np.full(len(pc), np.inf), np.full(len(gc), np.inf)
        dist = {"p2g_mean_cm": float("nan"), "p2g_max_cm": float("nan"),
                "g2p_mean_cm": float("nan"), "g2p_max_cm": float("nan")}

    tol = tolerance_cm / 100.0 + 1e-9
    near_gt = d_pg <= tol                          # predicted voxel has a GT voxel within the tolerance
    covered = d_gp <= tol                          # GT voxel has a predicted voxel within the tolerance
    hits, fps, misses = int(near_gt.sum()), int((~near_gt).sum()), int((~covered).sum())
    precision = hits / len(pc) if len(pc) else 0.0
    recall = int(covered.sum()) / len(gc) if len(gc) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def keys_of(c):
        return set(map(tuple, np.floor(c / voxel_size).astype(int))) if len(c) else set()

    key_sets = (keys_of(pc[near_gt]), keys_of(pc[~near_gt]), keys_of(gc[~covered]))

    metrics = {
        "tolerance_cm": float(tolerance_cm),
        "pred_voxels": len(pc), "gt_voxels": len(gc),
        "hits": hits, "misses": misses, "false_positives": fps,
        "precision": precision, "recall": recall, "f1": f1,
        **dist,
        "strict_hits": s_tp, "strict_misses": s_fn, "strict_false_positives": s_fp,
        "strict_precision": s_prec, "strict_recall": s_rec, "strict_f1": s_f1,
        "strict_avg_error_cm": s_avg * 100.0, "strict_max_error_cm": s_max * 100.0,
    }
    return metrics, key_sets


def summarize(rows):
    def vals(k):
        return [r[k] for r in rows if not np.isnan(r[k])]

    def mean(k):
        v = vals(k)
        return float(np.mean(v)) if v else float("nan")

    def vmax(k):
        v = vals(k)
        return float(np.max(v)) if v else float("nan")

    def ssum(k):
        return int(sum(r[k] for r in rows))

    s = {"scenes": len(rows), "tolerance_cm": rows[0]["tolerance_cm"]}
    for k in ("precision", "recall", "f1", "p2g_mean_cm", "g2p_mean_cm", "strict_precision", "strict_recall", "strict_f1"):
        s[f"mean_{k}"] = mean(k)
    s["p2g_max_cm"], s["g2p_max_cm"] = vmax("p2g_max_cm"), vmax("g2p_max_cm")

    hits, misses, fps, n_gt = ssum("hits"), ssum("misses"), ssum("false_positives"), ssum("gt_voxels")
    s.update({"total_hits": hits, "total_misses": misses, "total_false_positives": fps,
              "pooled_precision": hits / (hits + fps) if (hits + fps) else 0.0,
              "pooled_recall": (n_gt - misses) / n_gt if n_gt else 0.0})
    return s


# ----------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------
def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def plot_category(category, rows, out_path, voxel_cm, tol_cm):
    """Per-recording results of one category at one voxel size."""
    n = len(rows)
    x = np.arange(n)
    labels = [r["scene"].replace("recording_", "#") for r in rows]

    def g(k):
        return np.array([r[k] for r in rows], dtype=float)

    P, R, F = g("precision") * 100, g("recall") * 100, g("f1") * 100
    hits, miss, fps = g("hits"), g("misses"), g("false_positives")
    p2g, g2p, p2g_max, g2p_max = g("p2g_mean_cm"), g("g2p_mean_cm"), g("p2g_max_cm"), g("g2p_max_cm")

    fig, axes = plt.subplots(2, 2, figsize=(max(14, 0.6 * n + 6), 10))
    bw = 0.27

    ax = axes[0, 0]
    ax.bar(x - bw, P, bw, label="Precision", color="C0")
    ax.bar(x, R, bw, label="Recall", color="C1")
    ax.bar(x + bw, F, bw, label="F1", color="C2")
    ax.set_ylim(0, 135)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("%")
    ax.set_title(f"Precision / Recall / F1  (mean P={np.nanmean(P):.1f}%, R={np.nanmean(R):.1f}%, "
                 f"F1={np.nanmean(F):.1f}%)")

    ax = axes[0, 1]
    ax.bar(x - bw, hits, bw, label="Hits (pred. voxel near GT)", color="tab:green")
    ax.bar(x, miss, bw, label="Misses (GT voxel not covered)", color="tab:blue")
    ax.bar(x + bw, fps, bw, label="False positives", color="tab:red")
    ax.set_ylabel("voxels")
    ax.set_title("Hits / Misses / False positives")

    ax = axes[1, 0]
    ax.bar(x - bw / 2, np.nan_to_num(p2g), bw, label="mean: prediction -> GT", color="C4")
    ax.bar(x + bw / 2, np.nan_to_num(g2p), bw, label="mean: GT -> prediction", color="C5")
    ax.plot(x - bw / 2, p2g_max, "kx", label="max (same two directions)")
    ax.plot(x + bw / 2, g2p_max, "kx")
    ax.axhline(tol_cm, color="k", ls="--", alpha=.5, label=f"tolerance = {tol_cm:g} cm")
    ax.axhline(voxel_cm, color="gray", ls=":", label=f"1 voxel = {voxel_cm:g} cm")
    ax.set_ylabel("cm")
    ax.set_title("Distance between prediction and ground truth")

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        if ax is not axes[0, 0]:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.35)      # head-room so the legend does not cover the bars
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("occluded recording")
        ax.legend(fontsize=8, loc="upper center", ncol=2)
        ax.grid(axis="y", alpha=.3)

    ax = axes[1, 1]
    ax.scatter(R, P, s=60, c="C0", edgecolor="k", zorder=3)
    for xi, yi, lb in zip(R, P, labels):
        ax.annotate(lb, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlim(-2, 105)
    ax.set_ylim(-2, 105)
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Precision vs. recall per scene")
    ax.grid(alpha=.3)

    fig.suptitle(f"{category}  -  voxel {voxel_cm:g} cm, tolerance {tol_cm:g} cm  -  {n} scene(s)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overall(summaries, out_path, voxel_cm, tol_cm):
    """Category comparison at one voxel size."""
    cats = list(summaries.keys())
    names = [textwrap.fill(c, 14) for c in cats]
    x = np.arange(len(cats))
    bw = 0.27

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    a1.bar(x - bw, [summaries[c]["mean_precision"] * 100 for c in cats], bw, label="Precision", color="C0")
    a1.bar(x, [summaries[c]["mean_recall"] * 100 for c in cats], bw, label="Recall", color="C1")
    a1.bar(x + bw, [summaries[c]["mean_f1"] * 100 for c in cats], bw, label="F1", color="C2")
    a1.set_ylim(0, 135)
    a1.set_yticks(range(0, 101, 20))
    a1.set_ylabel("%")
    a1.set_title("Mean precision / recall / F1 per category")

    a2.bar(x - bw / 2, [np.nan_to_num(summaries[c]["mean_p2g_mean_cm"]) for c in cats], bw,
           label="mean: prediction -> GT", color="C4")
    a2.bar(x + bw / 2, [np.nan_to_num(summaries[c]["mean_g2p_mean_cm"]) for c in cats], bw,
           label="mean: GT -> prediction", color="C5")
    a2.plot(x - bw / 2, [summaries[c]["p2g_max_cm"] for c in cats], "kx", label="max")
    a2.plot(x + bw / 2, [summaries[c]["g2p_max_cm"] for c in cats], "kx")
    a2.axhline(tol_cm, color="k", ls="--", alpha=.5, label=f"tolerance = {tol_cm:g} cm")
    a2.axhline(voxel_cm, color="gray", ls=":", label=f"1 voxel = {voxel_cm:g} cm")
    a2.set_ylabel("cm")
    a2.set_title("Distance between prediction and ground truth")
    a2.set_ylim(0, a2.get_ylim()[1] * 1.35)

    for ax in (a1, a2):
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend(fontsize=8, loc="upper center", ncol=2)
        ax.grid(axis="y", alpha=.3)
    fig.suptitle(f"Categories  -  voxel {voxel_cm:g} cm, tolerance {tol_cm:g} cm", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _voxel_axis(by_voxel):
    vs = sorted(by_voxel)
    labels = [f"{v:g} cm\n(tol. {by_voxel[v]['tolerance_cm']:g} cm)" for v in vs]
    return vs, np.arange(len(vs)), labels


def _label_points(ax, x, y, color, dx=0, dy=7):
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points", xytext=(dx, dy), ha="center",
                    fontsize=8, color=color)


def plot_sweep(title, by_voxel, out_path):
    """Metrics vs. voxel size for one category (or all). Every voxel size uses its own tolerance (shown below it)."""
    vs, x, labels = _voxel_axis(by_voxel)

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(18, 5.8))
    for key, name, color, marker, dx, dy in (("precision", "Precision", "C0", "o", -11, 7),
                                             ("recall", "Recall", "C1", "s", 0, -15),
                                             ("f1", "F1", "C2", "D", 11, 7)):
        y = [by_voxel[v][f"mean_{key}"] * 100 for v in vs]
        a0.plot(x, y, "-" + marker, color=color, lw=2, label=name)
        _label_points(a0, x, y, color, dx, dy)
    a0.set_ylim(0, 115)
    a0.set_yticks(range(0, 101, 20))
    a0.set_ylabel("%")
    a0.set_title("Precision / Recall / F1 vs. voxel size")

    a1.plot(x, [by_voxel[v]["mean_p2g_mean_cm"] for v in vs], "-o", color="C4", lw=2, label="mean: prediction -> GT")
    a1.plot(x, [by_voxel[v]["mean_g2p_mean_cm"] for v in vs], "-o", color="C5", lw=2, label="mean: GT -> prediction")
    a1.plot(x, [by_voxel[v]["p2g_max_cm"] for v in vs], "--^", color="C4", alpha=.7, label="max: prediction -> GT")
    a1.plot(x, [by_voxel[v]["g2p_max_cm"] for v in vs], "--^", color="C5", alpha=.7, label="max: GT -> prediction")
    a1.plot(x, [by_voxel[v]["tolerance_cm"] for v in vs], "k--", marker="_", ms=14, alpha=.6, label="tolerance used")
    a1.plot(x, vs, ":", color="gray", label="1 voxel")
    a1.set_ylabel("distance (cm)")
    a1.set_title("Distance between prediction and ground truth")

    bw = 0.27
    a2.bar(x - bw, [by_voxel[v]["total_hits"] for v in vs], bw, label="Hits", color="tab:green")
    a2.bar(x, [by_voxel[v]["total_misses"] for v in vs], bw, label="Misses", color="tab:blue")
    a2.bar(x + bw, [by_voxel[v]["total_false_positives"] for v in vs], bw, label="False positives", color="tab:red")
    a2.set_ylabel("voxels (summed over all scenes)")
    a2.set_title("Hits / Misses / False positives")

    for ax in (a0, a1, a2):
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("voxel size")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sweep_categories(summaries, categories, voxel_sizes_cm, out_path):
    """Category comparison across voxel sizes (+ all categories pooled)."""
    by_voxel_all = {v: summaries[("ALL", v)] for v in voxel_sizes_cm if ("ALL", v) in summaries}
    vs, x, labels = _voxel_axis(by_voxel_all)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, key, name in zip(axes, ("f1", "precision", "recall"), ("F1", "Precision", "Recall")):
        for i, cat in enumerate(categories):
            ax.plot(x, [summaries[(cat, v)][f"mean_{key}"] * 100 for v in vs], "-o", color=f"C{i}", lw=2, label=cat)
        ax.plot(x, [summaries[("ALL", v)][f"mean_{key}"] * 100 for v in vs], "-s", color="k", lw=3,
                label="all categories")
        ax.set_ylim(0, 105)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("voxel size")
        ax.set_ylabel(f"{name} (%)")
        ax.set_title(f"{name} vs. voxel size")
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Categories vs. voxel size", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_debug_plot(path, scene, result, keys, voxel_size, project_3d_to_2d, title):
    tp_k, fp_k, fn_k = keys

    def centers(k):
        return (np.array(list(k)) + 0.5) * voxel_size if k else np.zeros((0, 3))

    fig, (ax_img, ax_map) = plt.subplots(1, 2, figsize=(15, 6))

    if scene["image"] is not None:
        overlay = scene["image"].copy()
        for tri in result["sidewall_tris"]:
            cv2.polylines(overlay, [project_3d_to_2d(np.array(tri), scene["intrinsics"])], True, (255, 0, 0), 1)
        for box in result["box_tops"]:
            cv2.polylines(overlay, [project_3d_to_2d(box, scene["intrinsics"])], True, (0, 0, 255), 3)
        ax_img.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax_img.set_title(f"Detected boxes: {len(result['box_tops'])}")
    ax_img.axis("off")

    for k, col, lb in ((tp_k, "tab:green", "hit"),
                       (fp_k, "tab:red", "false positive"),
                       (fn_k, "tab:blue", "miss")):
        c = centers(k)
        if len(c):
            ax_map.scatter(c[:, 0], c[:, 1], s=30, c=col, marker="s", label=f"{lb}: {len(c)}")
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.set_xlabel("x (m, LiDAR frame)")
    ax_map.set_ylabel("y (m, LiDAR frame)")
    ax_map.set_title("Top-down voxel comparison")
    ax_map.grid(alpha=.3)
    ax_map.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def build_tolerance_map(voxel_sizes_cli, tolerances_cli):
    """voxel size (cm) -> tolerance (cm), from TOLERANCES_CM or from the command line."""
    if voxel_sizes_cli is None:
        if tolerances_cli is not None:
            raise SystemExit("--tolerances needs --voxel-sizes")
        return dict(sorted(TOLERANCES_CM.items()))
    vs = list(voxel_sizes_cli)
    if tolerances_cli is None:
        missing = [v for v in vs if v not in TOLERANCES_CM]
        if missing:
            raise SystemExit(f"No tolerance configured in TOLERANCES_CM for voxel size(s) {missing} cm; "
                             f"add them there or pass --tolerances")
        return {v: TOLERANCES_CM[v] for v in vs}
    if len(tolerances_cli) == 1:
        tolerances_cli = list(tolerances_cli) * len(vs)
    if len(tolerances_cli) != len(vs):
        raise SystemExit("--tolerances needs one value per voxel size (or a single value for all)")
    return dict(zip(vs, tolerances_cli))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET_DIR)
    ap.add_argument("--pipeline", default=PIPELINE_SOURCE, help="path of the live script containing the pipeline")
    ap.add_argument("--output", default=OUTPUT_DIR)
    ap.add_argument("--offset-mode", choices=["geometry", "prediction"], default=None,
                    help="apply the offsets to the box walls before ray casting (default) or to the result")
    ap.add_argument("--voxel-sizes", nargs="+", type=float, default=None, metavar="CM",
                    help="voxel sizes in cm (default: the keys of TOLERANCES_CM)")
    ap.add_argument("--tolerances", nargs="+", type=float, default=None, metavar="CM",
                    help="one tolerance (cm) per voxel size given in --voxel-sizes (or a single value for all)")
    args = ap.parse_args()

    funcs, consts = load_pipeline(args.pipeline)
    offset_mode = args.offset_mode or CALIB_OFFSET_MODE
    tol_map = build_tolerance_map(args.voxel_sizes, args.tolerances)
    voxel_sizes_cm = list(tol_map.keys())

    print(f"Pipeline loaded from {args.pipeline}")
    print(f"Offset mode: {offset_mode} | default offset [cm]: x={CALIB_OFFSET[0] * 100:+.1f} "
          f"y={CALIB_OFFSET[1] * 100:+.1f} z={CALIB_OFFSET[2] * 100:+.1f} | scenes with their own offset: {len(SCENE_OFFSETS)}")
    print("Voxel size -> tolerance: " + ", ".join(f"{v:g} cm -> {t:g} cm" for v, t in tol_map.items()))

    dataset = Path(args.dataset)
    out = Path(args.output)
    chart_dir = out / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    categories = sorted(p for p in dataset.iterdir() if p.is_dir() and not p.name.startswith("_"))
    cat_pairs = [(c.name, find_pairs(c)) for c in categories]

    known = {scene_key(cat, occ.name) for cat, pairs in cat_pairs for occ, _ in pairs}
    for k in SCENE_OFFSETS:
        if k not in known:
            print(f"  [warn] SCENE_OFFSETS has an entry for '{k}', but no such scene exists (typo?)")

    all_rows = []
    rows_by = {}          # (category, voxel_cm) -> rows
    cat_names = []

    for category, pairs in cat_pairs:
        print(f"\n=== {category}: {len(pairs)} occluded/GT pair(s) ===")
        got_any = False

        for occ_dir, gt_dir in pairs:
            t0 = time.time()
            key = scene_key(category, occ_dir.name)
            offset = get_scene_offset(key)
            try:
                scene = load_scene(occ_dir)
                gt_raw = load_gt_points(gt_dir)
                base = compute_hits(scene, funcs, offset, offset_mode)   # independent of the voxel size
            except Exception as e:
                print(f"  [error] {occ_dir.name}: {e}")
                continue

            own = "" if key in SCENE_OFFSETS else " (default)"
            print(f"  {occ_dir.name} vs {gt_dir.name} | offset [cm] x={offset[0] * 100:+.1f} y={offset[1] * 100:+.1f} "
                  f"z={offset[2] * 100:+.1f}{own} | cam_pts={base['camera_points']} ({scene['camera_frame']} frame) "
                  f"boxes={len(base['box_tops'])} raw_hits={base['raw_hits']} | {time.time() - t0:.1f}s")
            if len(base["box_tops"]) < 2:
                print("    [hint] <2 boxes detected: the extension needs the occluder AND the hidden box. "
                      "If pointcloud_camera.pcd is already voxel-downsampled, the min_points=200 DBSCAN "
                      "in get_boxes_3d may be rejecting the boxes.")

            for vi, (v_cm, tol_cm) in enumerate(tol_map.items()):
                v = v_cm / 100.0
                try:
                    pred = (voxelize_hits(base["hit_points"], v, offset, offset_mode)
                            if len(base["hit_points"]) else np.empty((0, 3)))
                    gt_pts = filter_dense_voxels(gt_raw, v)
                    metrics, keys = evaluate(pred, gt_pts, v, tol_cm)
                except Exception as e:
                    print(f"    [error] voxel {v_cm:g} cm: {e}")
                    continue

                row = {"category": category, "scene": occ_dir.name, "gt_scene": gt_dir.name, "voxel_cm": v_cm,
                       "camera_points": base["camera_points"], "boxes_detected": len(base["box_tops"]),
                       "raw_hits": base["raw_hits"], **metrics,
                       "offset_x_cm": offset[0] * 100, "offset_y_cm": offset[1] * 100,
                       "offset_z_cm": offset[2] * 100, "offset_mode": offset_mode}
                rows_by.setdefault((category, v_cm), []).append(row)
                all_rows.append(row)
                got_any = True

                print(f"    v={v_cm:g}cm tol={tol_cm:g}cm | P={metrics['precision']:.1%} R={metrics['recall']:.1%} "
                      f"F1={metrics['f1']:.1%} [hits {metrics['hits']} misses {metrics['misses']} "
                      f"FP {metrics['false_positives']}] | pred->GT {metrics['p2g_mean_cm']:.1f}/{metrics['p2g_max_cm']:.1f} cm, "
                      f"GT->pred {metrics['g2p_mean_cm']:.1f}/{metrics['g2p_max_cm']:.1f} cm (mean/max)")

                if SAVE_DEBUG_PLOTS and vi == 0:
                    dbg_dir = out / "debug" / safe_name(category)
                    dbg_dir.mkdir(parents=True, exist_ok=True)
                    save_debug_plot(dbg_dir / f"{occ_dir.name}.png", scene, base, keys, v,
                                    funcs["project_3d_to_2d"],
                                    f"{category} / {occ_dir.name}  voxel {v_cm:g} cm, tolerance {tol_cm:g} cm  "
                                    f"(P={metrics['precision']:.1%}, R={metrics['recall']:.1%})")

        if got_any:
            cat_names.append(category)

    if not all_rows:
        print("No scenes evaluated.")
        return

    # ---------------- summaries ----------------
    summaries = {}
    for (cat, v_cm), rows in rows_by.items():
        summaries[(cat, v_cm)] = summarize(rows)
    for v_cm in voxel_sizes_cm:
        rows = [r for r in all_rows if r["voxel_cm"] == v_cm]
        if rows:
            summaries[("ALL", v_cm)] = summarize(rows)

    print("\n" + "=" * 100)
    for cat in cat_names + ["ALL"]:
        for v_cm in voxel_sizes_cm:
            s = summaries.get((cat, v_cm))
            if s is None:
                continue
            print(f"{cat:<28} v={v_cm:>4g}cm tol={s['tolerance_cm']:g}cm | P={s['mean_precision']:.1%} "
                  f"R={s['mean_recall']:.1%} F1={s['mean_f1']:.1%} | hits {s['total_hits']} misses {s['total_misses']} "
                  f"FP {s['total_false_positives']} | pred->GT mean {s['mean_p2g_mean_cm']:.1f} max {s['p2g_max_cm']:.1f} cm")

    # ---------------- CSV ----------------
    write_csv(out / "per_recording_metrics.csv", all_rows)
    write_csv(out / "category_summary.csv",
              [{"category": cat, "voxel_cm": v_cm, **s} for (cat, v_cm), s in summaries.items()])

    # ---------------- charts ----------------
    for v_cm, tol_cm in tol_map.items():
        vdir = chart_dir / f"voxel_{v_cm:g}cm"
        vdir.mkdir(parents=True, exist_ok=True)
        for cat in cat_names:
            if (cat, v_cm) in rows_by:
                plot_category(cat, rows_by[(cat, v_cm)], vdir / f"{safe_name(cat)}_metrics.png", v_cm, tol_cm)
        plot_overall({c: summaries[(c, v_cm)] for c in cat_names if (c, v_cm) in summaries},
                     vdir / "overall_comparison.png", v_cm, tol_cm)

    for cat in cat_names + ["ALL"]:
        by_voxel = {v: summaries[(cat, v)] for v in voxel_sizes_cm if (cat, v) in summaries}
        title = "All categories" if cat == "ALL" else cat
        plot_sweep(f"{title}  -  metrics vs. voxel size", by_voxel,
                   chart_dir / f"voxel_sweep_{safe_name(cat if cat != 'ALL' else 'all_categories')}.png")
    plot_sweep_categories(summaries, cat_names, voxel_sizes_cm, chart_dir / "voxel_sweep_categories.png")

    print(f"\nDone. Results in: {out.resolve()}")


if __name__ == "__main__":
    main()