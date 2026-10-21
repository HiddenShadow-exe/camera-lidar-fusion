#!/usr/bin/env python3
"""Generate conference-ready top-down navigation visuals before/after LiDAR completion.

Save this next to your existing evaluator module (dataset_evaluation.py or evaluate_dataset.py) and run, for example:
    python visualize_navigation_before_after.py \
        --dataset dataset \
        --pipeline lidar-ray-extension-voxelized.py \
        --output navigation_visualizations

For each odd/even recording pair, this script creates up to two images:
    <category>/recording_01_before.png
    <category>/recording_01_after.png

The evaluator loads pipeline functions from the live script's AST; the live ROS/socket loop is
never imported or executed. Box outlines come exclusively from the original get_boxes_3d output.
JSON OBB/object geometry is never read. Only detected boxes not already visible in the LiDAR cloud
and with pipeline-generated points are shown as hidden-box candidates.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import math
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon as MplPolygon, Patch
import numpy as np
import open3d as o3d  # required by evaluate_dataset.py and its cloud loader
from scipy.ndimage import binary_dilation

try:
    import dataset_evaluation as ev
except ImportError:
    try:
        import evaluate_dataset as ev
    except ImportError as exc:
        raise SystemExit(
            "Could not import dataset_evaluation.py or evaluate_dataset.py. Save this script "
            "beside your existing evaluation script, or add that folder to PYTHONPATH."
        ) from exc


# ---------------------------- geometry and occupancy map ----------------------------
def finite_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def obstacle_height_filter(points: np.ndarray, z_min: float, z_max: float,
                           max_range: float) -> np.ndarray:
    pts = finite_points(points)
    if not len(pts):
        return pts
    keep = ((pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
            & (np.linalg.norm(pts[:, :3], axis=1) <= max_range))
    return pts[keep]


def camera_points_to_lidar(points: np.ndarray, scene: dict) -> np.ndarray:
    """Transform camera-frame box corners into the same LiDAR frame used by the evaluator."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    r_c2l = np.diag([1.0, -1.0, -1.0])
    yaw = float(scene["lidar_yaw_rad"])
    r_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                       [np.sin(yaw),  np.cos(yaw), 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
    lidar_pos = np.asarray(scene["lidar_pos"], dtype=np.float64)
    return (r_yaw @ (r_c2l @ pts.T - (r_c2l @ lidar_pos)[:, None])).T


def polygon_from_detected_box(box_top: np.ndarray, scene: dict,
                              offset: np.ndarray, offset_mode: str) -> np.ndarray:
    """Convert one get_boxes_3d top-face rectangle to an ordered XY footprint.

    This deliberately uses the original pipeline's detected 3-D box corners rather than any
    recording_data.json OBB/object geometry.
    """
    top = np.asarray(box_top, dtype=np.float64).reshape(-1, 3)
    if len(top) < 3 or not np.isfinite(top).all():
        raise ValueError("Invalid detected box corners")
    corners_lidar = camera_points_to_lidar(top, scene)
    if offset_mode == "geometry":
        corners_lidar = corners_lidar + np.asarray(offset, dtype=np.float64)[None, :]
    xy = corners_lidar[:, :2]
    centre = xy.mean(axis=0)
    angle = np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0])
    poly = xy[np.argsort(angle)]
    if len(poly) < 3 or not np.isfinite(poly).all():
        raise ValueError("Detected box did not produce a valid XY footprint")
    return poly


def detected_box_z_range(box_top: np.ndarray, scene: dict,
                         offset: np.ndarray, offset_mode: str) -> tuple[float, float]:
    """Estimate a detected box's vertical span using get_boxes_3d's own floor plane."""
    top = np.asarray(box_top, dtype=np.float64).reshape(-1, 3)
    bottom = top.copy()
    # This is the same floor_z supplied to get_boxes_3d by evaluator.compute_hits.
    lidar_pos = np.asarray(scene["lidar_pos"], dtype=np.float64)
    bottom[:, 2] = lidar_pos[2] + float(getattr(ev, "LIDAR_HEIGHT", 0.4))
    both_lidar = camera_points_to_lidar(np.vstack((top, bottom)), scene)
    if offset_mode == "geometry":
        both_lidar += np.asarray(offset, dtype=np.float64)[None, :]
    z_values = both_lidar[:, 2]
    return float(np.min(z_values)), float(np.max(z_values))


def points_near_box_count(points: np.ndarray, poly: np.ndarray, tolerance: float,
                          z_range: Optional[tuple[float, float]] = None) -> int:
    """Count returns on/near a detected footprint, limited to the box's vertical span."""
    if len(points) == 0:
        return 0
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    tol = max(0.0, float(tolerance))
    if z_range is not None:
        z_lo, z_hi = z_range
        pts = pts[(pts[:, 2] >= z_lo - tol) & (pts[:, 2] <= z_hi + tol)]
        if len(pts) == 0:
            return 0
    distances = distance_points_to_polygon(pts[:, :2], poly)
    return int(np.count_nonzero(distances <= tol))


def find_completed_hidden_boxes(scene: dict, pred_result: dict,
                                scene_obstacle_points: np.ndarray,
                                predicted_obstacle_points: np.ndarray,
                                offset: np.ndarray, offset_mode: str,
                                resolution: float,
                                visible_min_points: int = 3,
                                generated_min_points: int = 3
                                ) -> tuple[list[dict], int, int]:
    """Use get_boxes_3d detections, dropping visible boxes and boxes without completion hits.

    Returns (eligible boxes, visible-box count, no-generated-points count). Each eligible box
    contains its original detection index, ordered XY polygon, and generated-point support.
    """
    eligible: list[dict] = []
    visible_dropped = 0
    no_prediction_dropped = 0
    # The output box_tops are produced by the original get_boxes_3d function through ev.run_pipeline.
    for det_index, box_top in enumerate(pred_result.get("box_tops", []), start=1):
        try:
            poly = polygon_from_detected_box(box_top, scene, offset, offset_mode)
        except (ValueError, TypeError):
            continue
        # The forward-facing LiDAR cannot usefully plan to a target wholly behind its origin.
        if float(np.max(poly[:, 0])) <= 0.0 or float(np.mean(poly[:, 0])) <= 0.0:
            continue
        tolerance = max(0.06, 1.5 * resolution)
        z_range = detected_box_z_range(box_top, scene, offset, offset_mode)
        visible_count = points_near_box_count(scene_obstacle_points, poly, tolerance, z_range)
        if visible_count >= visible_min_points:
            visible_dropped += 1
            continue
        generated_count = points_near_box_count(predicted_obstacle_points, poly, tolerance, z_range)
        if generated_count < generated_min_points:
            no_prediction_dropped += 1
            continue
        eligible.append({
            "detection_index": det_index,
            "poly": poly,
            "generated_points": generated_count,
            "original_lidar_support": visible_count,
            "z_range": z_range,
        })
    return eligible, visible_dropped, no_prediction_dropped


def world_to_cell(xy: np.ndarray, extent: float, resolution: float,
                  size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    cols = np.rint((xy[:, 0] + extent) / resolution).astype(np.int32)
    rows = np.rint((xy[:, 1] + extent) / resolution).astype(np.int32)
    valid = ((cols >= 0) & (cols < size) & (rows >= 0) & (rows < size)
             & (np.linalg.norm(xy, axis=1) <= extent))
    return rows, cols, valid


def make_disk(radius_cells: int) -> np.ndarray:
    radius_cells = max(0, int(radius_cells))
    yy, xx = np.ogrid[-radius_cells:radius_cells + 1,
                      -radius_cells:radius_cells + 1]
    return (xx * xx + yy * yy) <= radius_cells * radius_cells


def build_occupancy(points: np.ndarray, extent: float, resolution: float,
                    inflation_radius: float, start_clear_radius: float) -> np.ndarray:
    """Build an XY occupancy grid from observed 3-D point endpoints."""
    size = int(round((2.0 * extent) / resolution)) + 1
    occ = np.zeros((size, size), dtype=bool)
    if len(points):
        rows, cols, valid = world_to_cell(points[:, :2], extent, resolution, size)
        occ[rows[valid], cols[valid]] = True

    radius_cells = int(math.ceil(inflation_radius / resolution))
    if radius_cells > 0 and occ.any():
        occ = binary_dilation(occ, structure=make_disk(radius_cells))

    # Keep planning inside the requested sensor-range disc (rather than the square corners).
    yy, xx = np.mgrid[0:size, 0:size]
    world_x = xx * resolution - extent
    world_y = yy * resolution - extent
    occ[(world_x * world_x + world_y * world_y) > extent * extent] = True

    # The robot begins at the LiDAR origin and the sensor's usable field of view is forward.
    # Clear a small start footprint, then make x < 0 impassable so the planner cannot route
    # behind the sensor even though the internal grid is kept symmetric for simple indexing.
    occ[(world_x * world_x + world_y * world_y) <= start_clear_radius * start_clear_radius] = False
    occ[world_x < -0.5 * resolution] = True
    return occ


def path_astar(occ: np.ndarray, start_xy: tuple[float, float], goal_xy: tuple[float, float],
               extent: float, resolution: float) -> Optional[np.ndarray]:
    """8-connected A* with no diagonal corner cutting; returns world XY cell centres."""
    size = occ.shape[0]
    sr, sc, sv = world_to_cell(np.array([start_xy]), extent, resolution, size)
    gr, gc, gv = world_to_cell(np.array([goal_xy]), extent, resolution, size)
    if not bool(sv[0] and gv[0]):
        return None
    sr, sc, gr, gc = int(sr[0]), int(sc[0]), int(gr[0]), int(gc[0])
    if occ[sr, sc] or occ[gr, gc]:
        return None

    inf = np.inf
    best = np.full(occ.shape, inf, dtype=np.float64)
    parent = np.full(occ.shape, -1, dtype=np.int64)
    best[sr, sc] = 0.0
    start_flat = sr * size + sc
    goal_flat = gr * size + gc
    heap = [(math.hypot(gr - sr, gc - sc), 0.0, sr, sc)]
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
             (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]

    found = False
    while heap:
        _, g, r, c = heapq.heappop(heap)
        if g > best[r, c] + 1e-10:
            continue
        if r == gr and c == gc:
            found = True
            break
        for dr, dc, move_cost in moves:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= size or nc < 0 or nc >= size or occ[nr, nc]:
                continue
            if dr != 0 and dc != 0 and (occ[r, nc] or occ[nr, c]):
                continue
            ng = g + move_cost
            if ng + 1e-10 < best[nr, nc]:
                best[nr, nc] = ng
                parent[nr, nc] = r * size + c
                h = math.hypot(gr - nr, gc - nc)
                heapq.heappush(heap, (ng + h, ng, nr, nc))

    if not found:
        return None

    flat = goal_flat
    cells = []
    while flat != start_flat:
        r, c = divmod(flat, size)
        cells.append((r, c))
        flat = int(parent[r, c])
        if flat < 0:
            return None
    cells.append((sr, sc))
    cells.reverse()
    cells = np.asarray(cells, dtype=np.float64)
    path = np.column_stack((cells[:, 1] * resolution - extent,
                            cells[:, 0] * resolution - extent))
    path[0] = np.asarray(start_xy, dtype=np.float64)
    path[-1] = np.asarray(goal_xy, dtype=np.float64)
    return smooth_path(path, occ, extent, resolution)


def line_is_free(a: np.ndarray, b: np.ndarray, occ: np.ndarray,
                 extent: float, resolution: float) -> bool:
    dist = float(np.linalg.norm(b - a))
    count = max(2, int(math.ceil(dist / (resolution * 0.35))) + 1)
    sample = a[None, :] + np.linspace(0.0, 1.0, count)[:, None] * (b - a)[None, :]
    rows, cols, valid = world_to_cell(sample, extent, resolution, occ.shape[0])
    return bool(valid.all() and not occ[rows, cols].any())


def smooth_path(path: np.ndarray, occ: np.ndarray, extent: float,
                resolution: float) -> np.ndarray:
    if path is None or len(path) <= 2:
        return path
    result = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_is_free(path[i], path[j], occ, extent, resolution):
            j -= 1
        result.append(path[j])
        i = j
    return np.asarray(result, dtype=np.float64)


def sample_path(path: Optional[np.ndarray], step: float) -> np.ndarray:
    if path is None or len(path) == 0:
        return np.empty((0, 2), dtype=np.float64)
    samples = []
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / max(step, 1e-4))))
        samples.extend(a[None, :] + (np.arange(n) / n)[:, None] * (b - a)[None, :])
    samples.append(path[-1])
    return np.asarray(samples, dtype=np.float64)


def points_inside_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crossing = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= crossing
        j = i
    return inside


def distance_points_to_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Euclidean distance to polygon (zero for points inside)."""
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    inside = points_inside_polygon(points, poly)
    min_d2 = np.full(len(points), np.inf, dtype=np.float64)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            t = np.zeros(len(points))
        else:
            t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
        nearest = a[None, :] + t[:, None] * ab[None, :]
        d2 = np.sum((points - nearest) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
    d = np.sqrt(min_d2)
    d[inside] = 0.0
    return d


def path_hits_detected_box(path: Optional[np.ndarray], poly: np.ndarray, robot_radius: float,
                     resolution: float) -> tuple[Optional[bool], float]:
    if path is None:
        return None, float("nan")
    samples = sample_path(path, resolution * 0.3)
    distances = distance_points_to_polygon(samples, poly)
    min_distance = float(distances.min()) if len(distances) else float("inf")
    return bool(np.any(distances <= robot_radius)), min_distance


def path_length(path: Optional[np.ndarray]) -> float:
    if path is None or len(path) < 2:
        return float("nan") if path is None else 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def path_prefix_length(path: Optional[np.ndarray], point: Optional[np.ndarray]) -> float:
    """Length along path from its start to the closest projection of point onto a segment."""
    if path is None or len(path) < 2 or point is None:
        return float("nan")
    p = np.asarray(point, dtype=np.float64).reshape(2)
    cumulative = 0.0
    best_distance = float("inf")
    best_length = float("nan")
    for a, b in zip(path[:-1], path[1:]):
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom <= 1e-12 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        projection = a + t * ab
        distance = float(np.linalg.norm(p - projection))
        segment_length = float(np.linalg.norm(ab))
        if distance < best_distance:
            best_distance = distance
            best_length = cumulative + t * segment_length
        cumulative += segment_length
    return best_length


def make_goal_candidates(poly: np.ndarray, robot_radius: float, extent: float) -> list[np.ndarray]:
    """Candidate goals lie beyond the hidden box, along the origin-to-box direction."""
    centre = poly.mean(axis=0)
    norm = float(np.linalg.norm(centre))
    direction = centre / norm if norm > 1e-3 else np.array([1.0, 0.0])
    perpendicular = np.array([-direction[1], direction[0]])
    front_extent = float(np.max((poly - centre[None, :]) @ direction))
    goals: list[np.ndarray] = []
    extras = [0.28, 0.45, 0.65, 0.9, 1.2, 1.5]
    laterals = [0.0, -0.2, 0.2, -0.4, 0.4, -0.65, 0.65, -0.9, 0.9, -1.2, 1.2]
    for extra in extras:
        for lateral in laterals:
            goal = centre + direction * (front_extent + robot_radius + extra) + perpendicular * lateral
            if np.linalg.norm(goal) < extent - 0.15:
                goals.append(goal)
    return goals


def select_shared_goal(base_occ: np.ndarray, completed_occ: np.ndarray, poly: np.ndarray,
                       extent: float, resolution: float, robot_radius: float
                       ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], bool, bool]:
    """Find a shared goal where the raw-LiDAR route hits the detected box and completion avoids it.

    If no such example exists in the data/prediction, choose the best reachable candidate
    without forcing a result. This keeps the figures faithful to the actual pipeline output.
    """
    start = (0.0, 0.0)
    candidates = make_goal_candidates(poly, robot_radius, extent)
    evaluated = []
    grid_size = base_occ.shape[0]
    for goal in candidates:
        # A goal must be outside the hidden-box footprint plus the robot clearance radius.
        if float(distance_points_to_polygon(goal[None, :], poly)[0]) <= robot_radius + 0.08:
            continue
        br, bc, bv = world_to_cell(goal[None, :], extent, resolution, grid_size)
        if not bv[0] or base_occ[br[0], bc[0]] or completed_occ[br[0], bc[0]]:
            continue
        before = path_astar(base_occ, start, tuple(goal), extent, resolution)
        if before is None:
            continue
        after = path_astar(completed_occ, start, tuple(goal), extent, resolution)
        before_hit, before_min = path_hits_detected_box(before, poly, robot_radius, resolution)
        after_hit, after_min = path_hits_detected_box(after, poly, robot_radius, resolution)
        if after is not None:
            # Prefer a genuine before/after contrast, then shorter paths and central crossings.
            centre_distance = float(distance_points_to_polygon(before[len(before)//2:len(before)//2+1], poly)[0])
            score = (0.0 if (before_hit and not after_hit) else 100.0)
            score += path_length(after) + 0.12 * path_length(before) + 0.15 * centre_distance
        else:
            score = 1000.0 + path_length(before)
        evaluated.append((score, goal, before, after, bool(before_hit), bool(after_hit) if after_hit is not None else False))

    if not evaluated:
        # Last resort: one goal directly behind the target. We still produce the two PNGs,
        # with an explicit no-route annotation if the occupancy maps make it unreachable.
        centre = poly.mean(axis=0)
        direction = centre / max(float(np.linalg.norm(centre)), 1e-3)
        front_extent = float(np.max((poly - centre[None, :]) @ direction))
        goal = centre + direction * (front_extent + robot_radius + 0.8)
        before = path_astar(base_occ, start, tuple(goal), extent, resolution)
        after = path_astar(completed_occ, start, tuple(goal), extent, resolution)
        before_hit, _ = path_hits_detected_box(before, poly, robot_radius, resolution)
        after_hit, _ = path_hits_detected_box(after, poly, robot_radius, resolution)
        return goal, before, after, bool(before_hit), bool(after_hit) if after_hit is not None else False

    evaluated.sort(key=lambda x: x[0])
    _, goal, before, after, before_hit, after_hit = evaluated[0]
    return goal, before, after, before_hit, after_hit


def add_discovered_box(occ: np.ndarray, poly: np.ndarray, extent: float,
                       resolution: float, clearance: float) -> np.ndarray:
    """Return a new map after a previously hidden box is detected.

    The detected box footprint stands in for the newly observed obstacle at the replan event.
    It is not present in the initial LiDAR-only map.
    """
    updated = occ.copy()
    size = occ.shape[0]
    axis = np.arange(size, dtype=np.float64) * resolution - extent
    grid_x, grid_y = np.meshgrid(axis, axis)
    grid_xy = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    near_box = distance_points_to_polygon(grid_xy, poly) <= clearance
    updated.ravel()[near_box] = True
    # Maintain the forward-only traversable region.
    updated[grid_x < -0.5 * resolution] = True
    return updated


def plan_online_replan(initial_path: Optional[np.ndarray], base_occ: np.ndarray,
                       poly: np.ndarray, goal: np.ndarray, extent: float,
                       resolution: float, clearance: float
                       ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[bool]]:
    """Simulate a mid-route obstacle discovery and plan a safe continuation.

    Returns (replan point, route from that point to goal, collision flag). The replan
    point is placed on the initially planned route, just before its first approach to the
    hidden obstacle. If a grid cell is already occupied, the point is moved backward along
    the original path until a free cell is found.
    """
    if initial_path is None or len(initial_path) == 0:
        return None, None, None

    step = max(resolution * 0.45, 0.01)
    samples = sample_path(initial_path, step)
    if len(samples) == 0:
        return None, None, None
    distances = distance_points_to_polygon(samples, poly)
    colliding = np.flatnonzero(distances <= clearance)
    if len(colliding) == 0:
        return None, None, None

    # Trigger when the robot first comes within a short detection lead distance, with a
    # hard limit to ensure this is before the initial path enters the box clearance area.
    detection_lead = max(0.18, 4.0 * resolution)
    trigger_dist = clearance + detection_lead
    trigger_candidates = np.flatnonzero(distances <= trigger_dist)
    first_collision = int(colliding[0])
    start_index = int(trigger_candidates[0]) if len(trigger_candidates) else first_collision - 1
    start_index = min(start_index, first_collision - 1)
    start_index = max(0, start_index)

    updated_occ = add_discovered_box(base_occ, poly, extent, resolution, clearance)
    size = base_occ.shape[0]

    # Prefer the first detection point, stepping backward only when discretization or an
    # already-observed return occupies that exact cell.
    for idx in range(start_index, -1, -1):
        point = samples[idx].copy()
        rows, cols, valid = world_to_cell(point[None, :], extent, resolution, size)
        if not bool(valid[0]):
            continue
        r, c = int(rows[0]), int(cols[0])
        if base_occ[r, c] or updated_occ[r, c]:
            continue
        route = path_astar(updated_occ, tuple(point), tuple(goal), extent, resolution)
        if route is None:
            continue
        hit, _ = path_hits_detected_box(route, poly, clearance, resolution)
        if hit is False:
            return point, route, False

    # Preserve a meaningful marker even if the map leaves no feasible safe continuation.
    point = samples[start_index].copy()
    rows, cols, valid = world_to_cell(point[None, :], extent, resolution, size)
    if bool(valid[0]):
        route = path_astar(updated_occ, tuple(point), tuple(goal), extent, resolution)
        hit, _ = path_hits_detected_box(route, poly, clearance, resolution)
        return point, route, hit
    return point, None, None


def view_limits(target_poly: np.ndarray, goal: np.ndarray, paths: list[Optional[np.ndarray]],
                extent: float, robot_radius: float = 0.18,
                other_polys: Optional[list[np.ndarray]] = None) -> tuple[float, float, float, float]:
    """Crop the top-down view to the forward route, predicted boxes and destination."""
    chunks = [np.array([[0.0, 0.0], [goal[0], goal[1]]], dtype=np.float64), target_poly[:, :2]]
    if other_polys:
        chunks.extend(np.asarray(poly, dtype=np.float64)[:, :2] for poly in other_polys)
    for path in paths:
        if path is not None and len(path):
            chunks.append(np.asarray(path)[:, :2])
    xy = np.vstack([c for c in chunks if len(c)])
    xmax_focus = float(np.max(xy[:, 0]))
    xmax = min(extent, max(1.2, xmax_focus + max(0.38, 0.10 * xmax_focus)))
    # Keep only a small silhouette margin behind the origin so the robot marker is whole;
    # point returns and x-axis tick labels still start at x=0.
    xmin = -min(0.25, max(0.12, robot_radius + 0.02))
    # Leave a forward-side callout bay for the GOAL label so it never sits on the route.
    xmax = max(xmax, min(extent, float(goal[0]) + 0.75))

    ymin_focus = float(np.min(xy[:, 1]))
    ymax_focus = float(np.max(xy[:, 1]))
    yspan = max(0.0, ymax_focus - ymin_focus)
    # The bottom pad also reserves a quiet band for the START/LIDAR callout, away from routes.
    pad = max(0.48, 0.10 * yspan)
    ymin = max(-extent, ymin_focus - pad)
    ymax = min(extent, ymax_focus + pad)
    if ymax - ymin < 1.0:
        centre = 0.5 * (ymin + ymax)
        ymin = max(-extent, centre - 0.5)
        ymax = min(extent, centre + 0.5)
    return xmin, xmax, ymin, ymax


def clip_cloud_to_view(points: np.ndarray, limits: tuple[float, float, float, float]) -> np.ndarray:
    """Drop off-screen point-cloud data instead of rendering irrelevant regions."""
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    xmin, xmax, ymin, ymax = limits
    keep = ((points[:, 0] >= max(0.0, xmin)) & (points[:, 0] <= xmax) &
            (points[:, 1] >= ymin) & (points[:, 1] <= ymax))
    return points[keep]


# ---------------------------- rendering ----------------------------
def thin_xy(points: np.ndarray, max_points: int = 30000) -> np.ndarray:
    if len(points) <= max_points:
        return points
    # Deterministic evenly spaced subsampling keeps rendering fast and repeatable.
    idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[idx]


def route_label(hit: Optional[bool]) -> str:
    if hit is None:
        return "NO FEASIBLE ROUTE"
    return "COLLIDES WITH HIDDEN BOX" if hit else "AVOIDS HIDDEN BOX"


def render_visual(out_path: Path, *, phase: str, category: str, occ_dir: Path, gt_dir: Path,
                  scene_cloud: np.ndarray, predicted_cloud: np.ndarray, target_poly: np.ndarray,
                  completed_box_polys: Optional[list[np.ndarray]] = None,
                  path: Optional[np.ndarray], goal: np.ndarray, path_hit: Optional[bool],
                  replanned_path: Optional[np.ndarray] = None,
                  replan_point: Optional[np.ndarray] = None,
                  replanned_hit: Optional[bool] = None,
                  robot_radius: float = 0.18, extent: float = 5.0,
                  resolution: float = 0.04, dpi: int = 300) -> None:
    """Export a large, forward-only, conference-ready overhead view."""
    is_after = phase == "after"
    initial_color = "#E59A23"
    safe_color = "#13A88A"
    bad_color = "#E85D4A"
    path_color = safe_color if path_hit is False else (bad_color if path_hit is True else "#7C8798")
    observed_color = "#667487"
    prediction_color = "#2584F4"
    gt_color = "#D94358"
    goal_color = "#F2B632"
    replan_color = "#7C3AED"

    limits = view_limits(target_poly, goal, [path, replanned_path], extent, robot_radius,
                         other_polys=completed_box_polys)
    xmin, xmax, ymin, ymax = limits
    visible_scene = clip_cloud_to_view(scene_cloud, limits)
    visible_pred = clip_cloud_to_view(predicted_cloud, limits) if is_after else np.empty((0, 3))

    # Keep a 16:9 conference canvas. The map gets most of the left side, while the
    # legend occupies its own vertical panel on the right instead of a footer over/under the map.
    fig = plt.figure(figsize=(14, 7.5), facecolor="#FAFBFD")
    ax = fig.add_axes([0.085, 0.1, 0.625, 0.715])
    title = "AFTER COMPLETION" if is_after else "BEFORE COMPLETION · ONLINE REPLAN"
    if is_after:
        status = "ROUTE AVOIDS BOX" if path_hit is False else (
            "ROUTE INTERSECTS BOX" if path_hit is True else "NO FEASIBLE ROUTE")
        status_color = safe_color if path_hit is False else (bad_color if path_hit is True else "#657181")
    else:
        if path_hit is True and replanned_hit is False:
            status, status_color = "DISCOVERY → DETOUR", bad_color
        elif path_hit is True:
            status, status_color = "INITIAL ROUTE COLLIDES", bad_color
        else:
            status, status_color = "NO COLLISION-TRIGGERED REPLAN", "#657181"
    fig.text(0.02, 0.965, title, fontsize=25, fontweight="bold", color="#16202C", va="top")
    fig.text(0.955, 0.963, status, fontsize=15.5, fontweight="bold", color=status_color,
             ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                       edgecolor=status_color, linewidth=1.5, alpha=0.98))

    def length_text(route: Optional[np.ndarray]) -> str:
        length = path_length(route)
        return f"{length:.2f} m" if np.isfinite(length) else "N/A"

    if is_after:
        metric_text = f"COMPLETED ROUTE  {length_text(path)}"
        metric_color = path_color
    else:
        prefix_len = path_prefix_length(path, replan_point)
        remaining_len = path_length(replanned_path)
        replanned_total = prefix_len + remaining_len if np.isfinite(prefix_len) and np.isfinite(remaining_len) else float("nan")
        replanned_total_text = f"{replanned_total:.2f} m" if np.isfinite(replanned_total) else "N/A"
        metric_text = (f"INITIAL PLAN  {length_text(path)}     ·     "
                       f"REPLAN TOTAL  {replanned_total_text}")
        metric_color = safe_color if replanned_hit is False else initial_color
    fig.text(0.02, 0.875, metric_text, fontsize=19, fontweight="bold", color=metric_color,
             ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.38", facecolor="white",
                       edgecolor="#D5DCE6", linewidth=1.2, alpha=0.99))

    # Plot all visible returns (no deterministic thinning); rasterization preserves crisp
    # points while keeping the 300-dpi export manageable.
    if len(visible_scene):
        ax.scatter(visible_scene[:, 0], visible_scene[:, 1], s=1.8, c=observed_color,
                   alpha=0.62, linewidths=0, rasterized=True, zorder=2)
    if is_after and len(visible_pred):
        ax.scatter(visible_pred[:, 0], visible_pred[:, 1], s=5.2, c=prediction_color,
                   alpha=0.95, linewidths=0, rasterized=True, zorder=3)

    # Only display camera-detected boxes that passed BOTH filters: not already visible in
    # the original LiDAR cloud, and with pipeline-generated points on their footprints.
    # The selected target is emphasized; other qualifying boxes get thinner outlines.
    all_completed_polys = completed_box_polys or [target_poly]
    for other_poly in all_completed_polys:
        if np.allclose(other_poly, target_poly, atol=1e-6):
            continue
        ax.add_patch(MplPolygon(other_poly, closed=True, facecolor="none", edgecolor="#E8A0AB",
                                linewidth=1.8, linestyle=(0, (4, 3)), alpha=0.9, zorder=7.5))
    # The selected detected box is not in the initial occupancy map. In BEFORE it is revealed
    # only at the simulated discovery/replan event; AFTER shows completion around that box.
    ax.add_patch(MplPolygon(target_poly, closed=True, facecolor="#E85D75", edgecolor="none",
                            alpha=0.18, zorder=4))
    ax.add_patch(MplPolygon(target_poly, closed=True, facecolor="none", edgecolor=gt_color,
                            linewidth=3.0, linestyle=(0, (6, 3)), zorder=8))

    if not is_after and path is not None and len(path):
        # The entire original plan is dashed so the intended collision remains visible.
        ax.plot(path[:, 0], path[:, 1], color="white", linewidth=7.5, alpha=0.95,
                linestyle=(0, (7, 5)), solid_capstyle="round", zorder=5)
        ax.plot(path[:, 0], path[:, 1], color=initial_color, linewidth=3.2, alpha=0.95,
                linestyle=(0, (7, 5)), solid_capstyle="round", zorder=6)
        if replan_point is not None:
            # Highlight the segment the robot actually followed before discovering the box.
            dense = sample_path(path, max(resolution * 0.35, 0.01))
            idx = int(np.argmin(np.linalg.norm(dense - replan_point[None, :], axis=1))) if len(dense) else 0
            prefix = dense[:idx + 1]
            if len(prefix) >= 2:
                ax.plot(prefix[:, 0], prefix[:, 1], color="white", linewidth=8.5,
                        solid_capstyle="round", zorder=6.5)
                ax.plot(prefix[:, 0], prefix[:, 1], color=initial_color, linewidth=4.6,
                        solid_capstyle="round", zorder=7)

            if replanned_path is not None and len(replanned_path):
                replanned_color = safe_color if replanned_hit is False else bad_color
                ax.plot(replanned_path[:, 0], replanned_path[:, 1], color="white", linewidth=9.0,
                        solid_capstyle="round", zorder=7.2)
                ax.plot(replanned_path[:, 0], replanned_path[:, 1], color=replanned_color,
                        linewidth=5.0, solid_capstyle="round", zorder=7.5)
            # Recalculation icon: high-contrast purple marker at the route switch point.
            ax.scatter([replan_point[0]], [replan_point[1]], marker="D", s=300,
                       facecolor=replan_color, edgecolor="white", linewidth=2.2, zorder=12)
            ax.scatter([replan_point[0]], [replan_point[1]], marker="P", s=105,
                       facecolor="white", edgecolor="white", linewidth=0.5, zorder=13)
    elif is_after and path is not None and len(path):
        ax.plot(path[:, 0], path[:, 1], color="white", linewidth=9.0, alpha=0.98,
                solid_capstyle="round", zorder=5)
        ax.plot(path[:, 0], path[:, 1], color=path_color, linewidth=5.0,
                solid_capstyle="round", zorder=6)

    # START marker; text is deliberately placed in the lower-left whitespace, not on the route.
    ax.add_patch(Circle((0.0, 0.0), robot_radius, facecolor="#159D8A", edgecolor="white",
                        linewidth=2.0, alpha=0.98, zorder=10))
    ax.scatter([0.0], [0.0], marker="o", s=36, c="#103C33", zorder=11)
    # Keep the start label down in the reserved lower map margin, away from route geometry.
    start_text_y = ymin + 0.035 * (ymax - ymin)
    start_text_xy = (min(xmax - 0.08, max(0.12, 0.025 * xmax)), start_text_y)
    ax.annotate("START / LIDAR", xy=(0.0, 0.0), xytext=start_text_xy,
                textcoords="data", fontsize=16, fontweight="bold", color="#17352E",
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.36", facecolor="white",
                          edgecolor="#B6D8D0", linewidth=1.4, alpha=0.98),
                arrowprops=dict(arrowstyle="-", color="#66867F", linewidth=1.4,
                                connectionstyle="arc3,rad=0.12"), zorder=14)

    # Put GOAL in the forward-side callout bay, away from the final path segment.
    ax.scatter([goal[0]], [goal[1]], marker="*", s=720, c=goal_color,
               edgecolors="white", linewidths=1.8, zorder=11)
    goal_label_x = min(xmax - 0.58, goal[0] + 0.18)
    goal_label_y = float(np.clip(goal[1] + 0.12, ymin + 0.16, ymax - 0.16))
    ax.annotate("GOAL", xy=(goal[0], goal[1]), xytext=(goal_label_x, goal_label_y),
                textcoords="data", fontsize=17, fontweight="bold", color="#6B4A00",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.36", facecolor="white",
                          edgecolor="#EBD8A5", linewidth=1.4, alpha=0.98),
                arrowprops=dict(arrowstyle="-", color="#B58B2E", linewidth=1.4,
                                connectionstyle="arc3,rad=0.08"), zorder=14)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # Forward-only map: display and label only x >= 0.
    ax.set_xlim(max(-0.5, xmin), xmax)
    xtick_step = 0.5 if xmax <= 4.5 else 1.0
    ax.set_xticks(np.arange(0.0, xmax + 1e-6, xtick_step))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("FORWARD FROM LIDAR (m)", fontsize=16, labelpad=10,
                   fontweight="bold", color="#384454")
    ax.set_ylabel("LATERAL (+LEFT / −RIGHT, m)", fontsize=16, labelpad=10,
                   fontweight="bold", color="#384454")
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, color="#E6EAF0", linewidth=0.75, zorder=0)
    ax.tick_params(colors="#526071", labelsize=14, length=6, width=1.2)
    for spine in ax.spines.values():
        spine.set_color("#CFD6E0")
        spine.set_linewidth(1.1)

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=observed_color,
                      markeredgecolor="none", markersize=7, label="Observed LiDAR")]
    if is_after:
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=prediction_color,
                              markeredgecolor="none", markersize=7, label="Completed geometry"))
        handles.append(Line2D([0], [0], color=path_color, linewidth=4.2, label="Planned route"))
    else:
        handles.extend([
            Line2D([0], [0], color=initial_color, linewidth=3.2, linestyle=(0, (7, 5)),
                   label="Original plan"),
            Line2D([0], [0], color=safe_color if replanned_hit is False else bad_color,
                   linewidth=4.2, label="Replanned route"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=replan_color,
                   markeredgecolor="white", markeredgewidth=1.2, markersize=9, label="Replan point"),
        ])
    handles.extend([
        Patch(facecolor="#E85D75", alpha=0.16, edgecolor=gt_color, linestyle="--",
              label="Detected hidden box"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=goal_color,
               markeredgecolor="white", markersize=16, label="Goal"),
    ])
    # Dedicated right-side legend panel in one column. A divider visually separates it
    # from the map; the panel stays clear of every route, marker, and callout.
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.725, 0.795),
                     bbox_transform=fig.transFigure, ncol=1, frameon=True,
                     fontsize=16, labelcolor="#263344", handlelength=2.0,
                     labelspacing=1.05, handletextpad=0.85, borderpad=1.0,
                     borderaxespad=0.0, fancybox=True, framealpha=0.99)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#D6DDE6")
    leg.get_frame().set_linewidth(1.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------- orchestration ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=ev.DATASET_DIR, help="dataset root")
    ap.add_argument("--pipeline", default=ev.PIPELINE_SOURCE, help="existing live pipeline .py file")
    ap.add_argument("--output", default="navigation_visualizations", help="output folder for PNGs and CSV")
    ap.add_argument("--category", default=None, help="only categories whose names contain this text")
    ap.add_argument("--voxel-cm", type=float, default=None,
                    help="pipeline voxel size in cm (default: VOXEL_SIZE parsed from live pipeline)")
    ap.add_argument("--offset", nargs=3, type=float, default=None, metavar=("DX", "DY", "DZ"),
                    help="same LiDAR-frame calibration offset used by evaluator, in metres")
    ap.add_argument("--offset-mode", choices=["geometry", "prediction"], default=None)
    ap.add_argument("--resolution", type=float, default=0.04, help="XY occupancy-grid cell size in metres")
    ap.add_argument("--robot-radius", type=float, default=0.18, help="robot collision radius in metres")
    ap.add_argument("--safety-margin", type=float, default=0.04, help="extra obstacle inflation in metres")
    ap.add_argument("--z-min", type=float, default=-0.36,
                    help="minimum LiDAR-frame z of obstacle returns; excludes most floor returns")
    ap.add_argument("--z-max", type=float, default=0.80,
                    help="maximum LiDAR-frame z of obstacle returns")
    ap.add_argument("--max-range", type=float, default=5.0, help="planning/point-cloud radius in metres")
    ap.add_argument("--visible-min-points", type=int, default=3,
                    help="observed LiDAR points on a box footprint needed to treat it as already visible")
    ap.add_argument("--generated-min-points", type=int, default=3,
                    help="pipeline-generated points on a box footprint required to visualize it")
    ap.add_argument("--dpi", type=int, default=300, help="PNG export resolution (default 300 dpi)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset)
    output = Path(args.output).expanduser().resolve()
    pipeline = Path(args.pipeline).expanduser().resolve()
    if not dataset.is_dir():
        raise SystemExit(f"Dataset folder does not exist: {dataset}")
    if not pipeline.is_file():
        raise SystemExit(f"Pipeline source file does not exist: {pipeline}")
    if args.resolution <= 0 or args.robot_radius < 0 or args.max_range <= 0:
        raise SystemExit("--resolution and --max-range must be positive; --robot-radius cannot be negative")
    if args.z_min >= args.z_max:
        raise SystemExit("--z-min must be less than --z-max")

    funcs, consts = ev.load_pipeline(pipeline)
    if args.voxel_cm is not None:
        consts["VOXEL_SIZE"] = args.voxel_cm / 100.0
    offset = args.offset if args.offset is not None else ev.CALIB_OFFSET
    offset_mode = args.offset_mode or ev.CALIB_OFFSET_MODE

    pairs = []
    for cat_dir in sorted(p for p in dataset.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if args.category and args.category.lower() not in cat_dir.name.lower():
            continue
        try:
            for occ_dir, gt_dir in ev.find_pairs(cat_dir):
                pairs.append((cat_dir.name, occ_dir, gt_dir))
        except OSError as exc:
            print(f"[warn] Cannot inspect {cat_dir}: {exc}", file=sys.stderr)
    if not pairs:
        raise SystemExit(f"No odd/even recording pairs found under {dataset}")

    output.mkdir(parents=True, exist_ok=True)
    report_rows = []
    good_examples = 0
    for index, (category, occ_dir, gt_dir) in enumerate(pairs, 1):
        print(f"[{index}/{len(pairs)}] {category}/{occ_dir.name} -> {gt_dir.name}")
        try:
            scene = ev.load_scene(occ_dir)
            pred_result = ev.run_pipeline(scene, funcs, consts, offset=offset,
                                          offset_mode=offset_mode, voxel_size=consts["VOXEL_SIZE"])
            prediction_all = finite_points(pred_result.get("pred_points", np.empty((0, 3))))
            scene_points = obstacle_height_filter(scene["lidar_points"], args.z_min,
                                                  args.z_max, args.max_range)
            prediction_points = obstacle_height_filter(prediction_all, args.z_min,
                                                       args.z_max, args.max_range)
            # Box detections are generated by get_boxes_3d inside the unchanged evaluator pipeline.
            # No JSON OBB (or JSON object geometry) is read to derive a footprint.
            hidden_boxes, visible_dropped, uncompleted_dropped = find_completed_hidden_boxes(
                scene, pred_result, scene_points, prediction_points, offset, offset_mode,
                args.resolution, args.visible_min_points, args.generated_min_points)
            if not hidden_boxes:
                raise ValueError(
                    "No camera-detected boxes passed both filters: not already visible in LiDAR "
                    "and enough pipeline-generated points on the box footprint. "
                    f"detected={len(pred_result.get('box_tops', []))}, "
                    f"visible_dropped={visible_dropped}, "
                    f"without_completion_points_dropped={uncompleted_dropped}"
                )

            inflate = args.robot_radius + args.safety_margin
            start_clear = args.robot_radius + args.safety_margin + 0.06
            base_occ = build_occupancy(scene_points, args.max_range, args.resolution, inflate, start_clear)
            completed_cloud = np.vstack((scene_points, prediction_points)) if len(prediction_points) else scene_points
            completed_occ = build_occupancy(completed_cloud, args.max_range, args.resolution, inflate, start_clear)

            # Evaluate each eligible camera-detected hidden box as the route target. Prefer a
            # real raw-LiDAR collision plus completed-route avoidance, and then a safe online replan.
            candidates = []
            for box in hidden_boxes:
                poly = box["poly"]
                goal_i, before_i, after_i, _, _ = select_shared_goal(
                    base_occ, completed_occ, poly, args.max_range, args.resolution, inflate)
                before_hit_i, before_min_i = path_hits_detected_box(before_i, poly, inflate, args.resolution)
                after_hit_i, after_min_i = path_hits_detected_box(after_i, poly, inflate, args.resolution)
                replan_i, replanned_i, replanned_hit_i = (None, None, None)
                if before_hit_i is True:
                    replan_i, replanned_i, replanned_hit_i = plan_online_replan(
                        before_i, base_occ, poly, goal_i, args.max_range, args.resolution, inflate)
                contrast_i = (before_hit_i is True and after_hit_i is False)
                online_success_i = (replanned_i is not None and replanned_hit_i is False)
                length_after_i = path_length(after_i)
                score = (
                    0 if contrast_i else 1,
                    0 if online_success_i else 1,
                    length_after_i if np.isfinite(length_after_i) else 1e6,
                    path_length(before_i) if np.isfinite(path_length(before_i)) else 1e6,
                    -int(box["generated_points"]),
                )
                candidates.append((score, box, goal_i, before_i, after_i, before_hit_i,
                                   after_hit_i, before_min_i, after_min_i, replan_i,
                                   replanned_i, replanned_hit_i, online_success_i))
            candidates.sort(key=lambda row: row[0])
            (_, target_box, goal, before_path, after_path, before_hit, after_hit,
             before_min, after_min, replan_point, replanned_path, replanned_hit,
             online_replan_success) = candidates[0]
            target_poly = target_box["poly"]
            completed_box_polys = [box["poly"] for box in hidden_boxes]
            successful_contrast = (before_hit is True and after_hit is False)
            if successful_contrast:
                good_examples += 1
            before_len = path_length(before_path)
            after_len = path_length(after_path)
            replanned_len = path_length(replanned_path)  # remaining distance from the replan point
            replan_prefix_len = path_prefix_length(before_path, replan_point)
            replanned_total_len = (replan_prefix_len + replanned_len
                                   if np.isfinite(replan_prefix_len) and np.isfinite(replanned_len)
                                   else float("nan"))

            cat_out = output / re.sub(r"[^A-Za-z0-9_.-]+", "_", category)
            stem = occ_dir.name
            before_file = cat_out / f"{stem}_before.png"
            after_file = cat_out / f"{stem}_after.png"
            render_visual(before_file, phase="before", category=category, occ_dir=occ_dir, gt_dir=gt_dir,
                          scene_cloud=scene_points, predicted_cloud=prediction_points, target_poly=target_poly,
                          completed_box_polys=completed_box_polys,
                          path=before_path, goal=goal, path_hit=before_hit,
                          replanned_path=replanned_path, replan_point=replan_point,
                          replanned_hit=replanned_hit, robot_radius=args.robot_radius,
                          extent=args.max_range, resolution=args.resolution, dpi=args.dpi)
            render_visual(after_file, phase="after", category=category, occ_dir=occ_dir, gt_dir=gt_dir,
                          scene_cloud=scene_points, predicted_cloud=prediction_points, target_poly=target_poly,
                          completed_box_polys=completed_box_polys,
                          path=after_path, goal=goal, path_hit=after_hit,
                          robot_radius=args.robot_radius, extent=args.max_range,
                          resolution=args.resolution, dpi=args.dpi)

            report_rows.append({
                "category": category,
                "occluded_recording": occ_dir.name,
                "ground_truth_pair_recording": gt_dir.name,
                "detected_boxes_total": len(pred_result.get("box_tops", [])),
                "visible_boxes_dropped": visible_dropped,
                "boxes_without_completion_points_dropped": uncompleted_dropped,
                "completed_hidden_boxes_visualized": len(hidden_boxes),
                "target_detection_index": target_box["detection_index"],
                "target_generated_points": target_box["generated_points"],
                "goal_x_m": round(float(goal[0]), 4),
                "goal_y_m": round(float(goal[1]), 4),
                "before_path_found": before_path is not None,
                "before_path_length_m": round(before_len, 4) if np.isfinite(before_len) else "",
                "before_collides_with_detected_box": before_hit if before_hit is not None else "",
                "before_min_distance_to_detected_box_m": round(before_min, 4) if np.isfinite(before_min) else "",
                "before_replan_point_x_m": round(float(replan_point[0]), 4) if replan_point is not None else "",
                "before_replan_point_y_m": round(float(replan_point[1]), 4) if replan_point is not None else "",
                "before_replanned_path_found": replanned_path is not None,
                "before_replanned_remaining_path_length_m": round(replanned_len, 4) if np.isfinite(replanned_len) else "",
                "before_replanned_total_path_length_m": round(replanned_total_len, 4) if np.isfinite(replanned_total_len) else "",
                "before_replanned_collides_with_detected_box": replanned_hit if replanned_hit is not None else "",
                "successful_online_replan": online_replan_success,
                "after_path_found": after_path is not None,
                "after_path_length_m": round(after_len, 4) if np.isfinite(after_len) else "",
                "after_collides_with_detected_box": after_hit if after_hit is not None else "",
                "after_min_distance_to_detected_box_m": round(after_min, 4) if np.isfinite(after_min) else "",
                "pipeline_predicted_points_total": len(prediction_all),
                "pipeline_predicted_obstacle_points": len(prediction_points),
                "successful_before_after_contrast": successful_contrast,
                "before_png": str(before_file.relative_to(output)),
                "after_png": str(after_file.relative_to(output)),
            })
            verdict = "DEMONSTRATION SUCCESS" if successful_contrast else "no valid before/after contrast"
            replan_verdict = "online replan avoids box" if online_replan_success else "online replan not verified"
            print(f"    detections={len(pred_result.get('box_tops', []))}; visible dropped={visible_dropped}; "
                  f"no-completion dropped={uncompleted_dropped}; qualifying hidden boxes={len(hidden_boxes)}; "
                  f"target generated points={target_box['generated_points']}; {verdict}; {replan_verdict}")
        except Exception as exc:
            print(f"    [error] {type(exc).__name__}: {exc}", file=sys.stderr)
            report_rows.append({
                "category": category, "occluded_recording": occ_dir.name,
                "ground_truth_pair_recording": gt_dir.name, "successful_before_after_contrast": False,
                "error": f"{type(exc).__name__}: {exc}",
            })

    report_file = output / "navigation_summary.csv"
    if report_rows:
        columns = list(dict.fromkeys(k for row in report_rows for k in row.keys()))
        with open(report_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report_rows)

    rendered = sum(1 for r in report_rows if "before_png" in r)
    print("\nFinished.")
    print(f"  Scene pairs: {len(pairs)}")
    print(f"  Pairs rendered: {rendered} (two PNGs per rendered pair)")
    print(f"  Before/after contrasts where the raw route hits the selected detected box and completion avoids it: {good_examples}")
    print(f"  Output: {output}")
    print(f"  Report: {report_file}")
    print("Note: target footprints come only from get_boxes_3d pipeline detections; JSON OBB metadata is never read.")
    print("Note: a successful contrast requires the raw LiDAR route to hit the selected detected box and the completed route to avoid it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
