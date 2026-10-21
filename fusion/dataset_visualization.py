#!/usr/bin/env python3
"""
Interactive 3D inspector for the pre-recorded dataset (companion of evaluate_dataset.py).

It runs the very same pipeline (through evaluate_dataset.run_pipeline) on one occluded/GT pair at a
time and shows everything in the LiDAR frame, so you can see WHERE the prediction goes wrong.
Every scene has its OWN calibration offset: dial it here, press P, paste the printed SCENE_OFFSETS
dictionary into evaluate_dataset.py.

Usage (keep it in the same folder as evaluate_dataset.py):
    python visualize_dataset.py --dataset dataset --pipeline your_live_script.py
    python visualize_dataset.py --category "full" --start 5       # only categories containing "full", start at recording_05
    python visualize_dataset.py --voxel-cm 3 --tol-cm 3           # voxel size / tolerance of the view

Keys (click into the 3D window first):
    Right / Left   next / previous scene
    Up / Down      first scene of the next / previous category
    1-9            toggle layers (see legend printed at start)
    V              voxels: solid cubes <-> wireframe (wireframe lets you see the points inside)
    A / D          offset of THIS scene   x - / +      (x = forward)
    S / W          offset of THIS scene   y - / +      (y = left)
    Z / X          offset of THIS scene   z - / +      (z = up)
    T              cycle the nudge step (0.5 / 1 / 2 / 5 cm)
    0              reset this scene's offset to the value configured in evaluate_dataset.py
    C              copy the offset of the previous scene into this one (handy as a starting point)
    P              print the offsets of ALL scenes as a SCENE_OFFSETS dictionary to paste into evaluate_dataset.py
    R              reset camera view (Open3D default);  - / =  point size;  Q  quit
Offsets are kept for the whole session: you can go back and forth between scenes without losing them.
Colours use the same decision as the evaluation: a predicted voxel is a hit if a GT voxel lies within the
tolerance of the voxel size, a GT voxel is missed if no predicted voxel lies within it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import dataset_evaluation as ev

# GLFW key codes
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265

R_C2L = np.diag([1.0, -1.0, -1.0])   # camera OpenCV -> LiDAR ROS (same as the pipeline)

# key -> (label, colour, visible at start)
LAYERS = {
    "1": ("LiDAR scene of the occluded recording (<5 m)", (0.55, 0.55, 0.55), True),
    "2": ("LiDAR object points of the occluded recording (json)", (1.0, 0.55, 0.0), False),
    "3": ("Camera cloud actually fed to the pipeline (in LiDAR frame)", (0.2, 0.8, 0.8), False),
    "4": ("Detected boxes: tops = magenta, sidewall triangles = yellow", None, True),
    "5": ("Raw extended hits, before voxel down-sampling", (0.7, 0.2, 0.95), False),
    "6": ("HIT voxels (GT voxel within tolerance)  = green", (0.1, 0.8, 0.1), True),
    "7": ("FALSE-POSITIVE voxels (no GT within tolerance) = red", (0.9, 0.1, 0.1), True),
    "8": ("MISSED GT voxels (no prediction within tolerance) = blue", (0.1, 0.2, 0.9), True),
    "9": ("GT LiDAR object points (even recording, raw)", (0.0, 0.5, 0.0), False),
}

CUBE_CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64)
CUBE_EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                       [0, 4], [1, 5], [2, 6], [3, 7]])
CUBE_TRIS = np.array([[0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                      [2, 3, 7], [2, 7, 6], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])


# ----------------------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------------------
def cam_to_lidar(pts, lidar_pos, yaw):
    """Same transform the pipeline applies to the triangles (and the live script to the blue cloud)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw),  np.cos(yaw), 0],
                      [0, 0, 1]], dtype=np.float64)
    return (R_yaw @ (R_C2L @ pts.T - (R_C2L @ lidar_pos)[:, None])).T


def make_points(pts, color):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color(color)
    return pcd


def make_lineset(points, lines, color):
    if len(lines) == 0:
        return None
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.paint_uniform_color(color)
    return ls


def voxel_solid(keys, color, v):
    if not keys:
        return None
    mins = np.array(list(keys), dtype=np.float64) * v
    n = len(mins)
    verts = (mins[:, None, :] + CUBE_CORNERS[None] * v).reshape(-1, 3)
    tris = (CUBE_TRIS[None] + (np.arange(n) * 8)[:, None, None]).reshape(-1, 3).astype(np.int32)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(tris))
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def voxel_wire(keys, color, v):
    if not keys:
        return None
    mins = np.array(list(keys), dtype=np.float64) * v
    n = len(mins)
    verts = (mins[:, None, :] + CUBE_CORNERS[None] * v).reshape(-1, 3)
    lines = (CUBE_EDGES[None] + (np.arange(n) * 8)[:, None, None]).reshape(-1, 2)
    return make_lineset(verts, lines, color)


def boxes_lineset(box_tops, floor_z_cam, lidar_pos, yaw, color, shift=(0.0, 0.0, 0.0)):
    pts, lines = [], []
    for b in box_tops:
        top = np.asarray(b, dtype=np.float64)
        bot = top.copy()
        bot[:, 2] = floor_z_cam
        base = len(pts)
        pts.extend(cam_to_lidar(np.vstack([top, bot]), lidar_pos, yaw) + shift)
        for i in range(4):
            lines += [[base + i, base + (i + 1) % 4],
                      [base + 4 + i, base + 4 + (i + 1) % 4],
                      [base + i, base + 4 + i]]
    return make_lineset(pts, lines, color)


def triangles_lineset(tris, lidar_pos, yaw, color, shift=(0.0, 0.0, 0.0)):
    if len(tris) == 0:
        return None
    pts = cam_to_lidar(np.array(tris, dtype=np.float64).reshape(-1, 3), lidar_pos, yaw) + shift
    base = np.arange(len(tris))[:, None] * 3
    lines = (base[:, :, None] + np.array([[0, 1], [1, 2], [2, 0]])[None]).reshape(-1, 2)
    return make_lineset(pts, lines, color)


def read_json(rec_dir):
    with open(Path(rec_dir) / "recording_data.json", "r") as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------
# Inspector
# ----------------------------------------------------------------------------------------------
NUDGE_STEPS = [0.005, 0.01, 0.02, 0.05]                      # metres
NUDGE_KEYS = {"A": (0, -1), "D": (0, +1), "S": (1, -1), "W": (1, +1), "Z": (2, -1), "X": (2, +1)}


class Inspector:
    def __init__(self, entries, funcs, consts, start_idx, mode, tol_cm):
        self.entries, self.funcs, self.consts = entries, funcs, consts
        self.voxel = consts["VOXEL_SIZE"]
        self.tol_cm = tol_cm
        self.idx = start_idx
        self.mode = mode
        self.step_i = 1
        # one calibration offset per scene (metres), initialised from evaluate_dataset.SCENE_OFFSETS / CALIB_OFFSET
        self.keys = [ev.scene_key(cat, occ.name) for cat, occ, _ in entries]
        self.start_offsets = {k: np.array(ev.get_scene_offset(k), dtype=np.float64) for k in self.keys}
        self.offsets = {k: v.copy() for k, v in self.start_offsets.items()}
        self.static = {}      # per scene: everything that does not depend on the offset
        self.dyn = {}         # per scene: (offset key, data)  -> recomputed when THAT scene's offset changes
        self.visible = {k: v[2] for k, v in LAYERS.items()}
        self.wire = False
        self.initialized = False
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

    @property
    def offset(self):
        """Offset of the scene that is currently shown (the array itself, so it can be edited in place)."""
        return self.offsets[self.keys[self.idx]]

    def edited(self, i):
        k = self.keys[i]
        return not np.allclose(self.offsets[k], self.start_offsets[k])

    # ---- data -------------------------------------------------------------------------------
    def get_static(self, i):
        if i in self.static:
            return self.static[i]
        category, occ_dir, gt_dir = self.entries[i]
        print(f"\nLoading {category} / {occ_dir.name} ...")
        scene = ev.load_scene(occ_dir)
        gt_pts = ev.load_gt_points(gt_dir, self.voxel)     # same density filter as the evaluation
        occ_meta, gt_meta = read_json(occ_dir), read_json(gt_dir)
        occ_obj = ev.as_points(occ_meta["3d_data_lidar"]["object_points"])
        lidar = scene["lidar_points"]
        lidar = lidar[np.linalg.norm(lidar, axis=1) < ev.MAX_LIDAR_RANGE]
        self.static[i] = {
            "scene": scene, "gt_pts": gt_pts, "lidar": lidar, "occ_obj": occ_obj,
            "occ_extr": occ_meta["calibration"]["lidar_to_cam_extrinsic"],
            "gt_extr": gt_meta["calibration"]["lidar_to_cam_extrinsic"],
            "boxes": None,    # (box_tops, sidewall_tris), filled by the first pipeline run
            "layers": {"1": make_points(lidar, LAYERS["1"][1]),
                       "2": make_points(occ_obj, LAYERS["2"][1]),
                       "9": make_points(gt_pts, LAYERS["9"][1])},
        }
        return self.static[i]

    def get_dynamic(self, i):
        off = self.offsets[self.keys[i]]
        key = (tuple(np.round(off, 6)), self.mode)
        if i in self.dyn and self.dyn[i][0] == key:
            return self.dyn[i][1]

        st = self.get_static(i)
        scene, v = st["scene"], self.voxel
        lp, yaw = scene["lidar_pos"], scene["lidar_yaw_rad"]

        res = ev.run_pipeline(scene, self.funcs, self.consts, offset=off, offset_mode=self.mode,
                              cached_boxes=st["boxes"], voxel_size=v)
        if st["boxes"] is None:
            st["boxes"] = (res["box_tops"], res["sidewall_tris"])
        if "3" not in st["layers"]:       # camera cloud as the pipeline used it (independent of the offset)
            cam = make_points(cam_to_lidar(res["camera_points_used"], lp, yaw), LAYERS["3"][1])
            st["layers"]["3"] = cam.voxel_down_sample(0.02) if cam is not None else None   # display only

        metrics, (tp, fp, fn) = ev.evaluate(res["pred_points"], st["gt_pts"], v, self.tol_cm)
        floor_z_cam = lp[2] + ev.LIDAR_HEIGHT
        shift = off if self.mode == "geometry" else np.zeros(3)   # walls move with a geometry offset

        layers = {
            "4": [g for g in (boxes_lineset(res["box_tops"], floor_z_cam, lp, yaw, (1.0, 0.0, 1.0), shift),
                              triangles_lineset(res["sidewall_tris"], lp, yaw, (1.0, 0.9, 0.0), shift))
                  if g is not None],
            "5": make_points(res["raw_hit_points"], LAYERS["5"][1]),
            "6": {"solid": voxel_solid(tp, LAYERS["6"][1], v), "wire": voxel_wire(tp, LAYERS["6"][1], v)},
            "7": {"solid": voxel_solid(fp, LAYERS["7"][1], v), "wire": voxel_wire(fp, LAYERS["7"][1], v)},
            "8": {"solid": voxel_solid(fn, LAYERS["8"][1], v), "wire": voxel_wire(fn, LAYERS["8"][1], v)},
        }
        data = {"layers": layers, "metrics": metrics, "result": res}
        self.dyn[i] = (key, data)
        return data

    # ---- console report ---------------------------------------------------------------------
    def offset_str(self):
        o = self.offset * 100
        return f"x={o[0]:+.1f} y={o[1]:+.1f} z={o[2]:+.1f} cm ({self.mode})"

    def metrics_line(self, m):
        return (f"tolerance {m['tolerance_cm']:g} cm: P={m['precision']:.1%} R={m['recall']:.1%} F1={m['f1']:.1%} | "
                f"HITS {m['hits']} | MISSES {m['misses']} | FALSE POS {m['false_positives']} | "
                f"pred->GT mean {m['p2g_mean_cm']:.1f} / max {m['p2g_max_cm']:.1f} cm, "
                f"GT->pred mean {m['g2p_mean_cm']:.1f} / max {m['g2p_max_cm']:.1f} cm")

    def print_info(self, i):
        st, d = self.get_static(i), self.get_dynamic(i)
        category, occ_dir, gt_dir = self.entries[i]
        m, r, sc = d["metrics"], d["result"], st["scene"]
        pred, gt = r["pred_points"], st["gt_pts"]
        n_edit = sum(self.edited(j) for j in range(len(self.entries)))

        print("=" * 100)
        print(f"[{i + 1}/{len(self.entries)}] {category}: {occ_dir.name} (occluded)  vs  {gt_dir.name} (GT)")
        print(f"  calibration offset of this scene: {self.offset_str()}"
              f"{'   [edited]' if self.edited(i) else ''}   | scenes edited so far: {n_edit}/{len(self.entries)}")
        print(f"  voxel {self.voxel * 100:g} cm | camera cloud: {r['camera_points']} pts ({sc['camera_frame']} frame) -> "
              f"boxes detected: {len(r['box_tops'])}, sidewall triangles: {len(r['sidewall_tris'])}")
        print(f"  LiDAR: scene {len(st['lidar'])} pts (<5 m) | occluded object pts {len(st['occ_obj'])} | GT pts {len(gt)}")
        print(f"  raw extended hits: {r['raw_hits']} -> {m['pred_voxels']} predicted voxels | GT voxels: {m['gt_voxels']}")
        print(f"  {self.metrics_line(m)}")

        oe, ge = st["occ_extr"], st["gt_extr"]
        same = (np.allclose(oe["lidar_position"], ge["lidar_position"], atol=1e-3)
                and abs(oe["lidar_yaw"] - ge["lidar_yaw"]) < 1e-3)
        print(f"  lidar pose occluded {np.round(oe['lidar_position'], 3).tolist()} yaw {oe['lidar_yaw']:.3f} | "
              f"GT {np.round(ge['lidar_position'], 3).tolist()} yaw {ge['lidar_yaw']:.3f}"
              f"{'' if same else '   <-- DIFFERENT, GT is not in the same frame!'}")

        if len(pred) and len(gt):
            off = (pred.mean(0) - gt.mean(0)) * 100
            nn, _ = cKDTree(gt).query(pred, k=1)
            print(f"  centroid pred {np.round(pred.mean(0), 3).tolist()} | GT {np.round(gt.mean(0), 3).tolist()} "
                  f"| offset pred-GT [cm]: {np.round(off, 1).tolist()}  (a starting point for the nudge: move by minus this)")
            print(f"  extent pred min {np.round(pred.min(0), 3).tolist()} max {np.round(pred.max(0), 3).tolist()}")
            print(f"  extent GT   min {np.round(gt.min(0), 3).tolist()} max {np.round(gt.max(0), 3).tolist()}")
            print(f"  point-to-nearest-GT distance of raw predicted points: mean {nn.mean() * 100:.1f} cm, "
                  f"median {np.median(nn) * 100:.1f} cm, max {nn.max() * 100:.1f} cm")
        else:
            print("  (no predicted points)" if len(gt) else "  (no GT points)")

    # ---- rendering --------------------------------------------------------------------------
    def refresh(self):
        st, d = self.get_static(self.idx), self.get_dynamic(self.idx)
        layers = {**st["layers"], **d["layers"]}
        vis = self.vis
        vis.clear_geometries()
        reset = not self.initialized

        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3), reset_bounding_box=reset)
        for k in LAYERS:
            if not self.visible[k]:
                continue
            g = layers.get(k)
            if isinstance(g, dict):
                g = g["wire" if self.wire else "solid"]
            for geom in (g if isinstance(g, list) else [g]):
                if geom is not None:
                    vis.add_geometry(geom, reset_bounding_box=reset)
        self.initialized = True
        vis.update_renderer()

    def go(self, delta):
        return self._goto((self.idx + delta) % len(self.entries))

    def go_category(self, delta):
        cats = [e[0] for e in self.entries]
        order = list(dict.fromkeys(cats))
        target = order[(order.index(cats[self.idx]) + delta) % len(order)]
        return self._goto(cats.index(target))

    def _goto(self, new):
        old = self.idx
        self.idx = new
        try:
            self.print_info(new)
        except Exception as e:                                  # keep the viewer alive on a broken scene
            print(f"  [error] could not load scene: {e}")
            self.static.pop(new, None)
            self.dyn.pop(new, None)
            self.idx = old
            return False
        self.refresh()
        return False

    def toggle(self, key):
        self.visible[key] = not self.visible[key]
        print(f"  layer {key} ({LAYERS[key][0]}): {'ON' if self.visible[key] else 'off'}")
        self.refresh()
        return False

    def toggle_style(self):
        self.wire = not self.wire
        print(f"  voxel style: {'wireframe' if self.wire else 'solid'}")
        self.refresh()
        return False

    # ---- offset nudging (always for the scene that is shown) ---------------------------------
    def rescore(self):
        try:
            d = self.get_dynamic(self.idx)
        except Exception as e:
            print(f"  [error] {e}")
            return False
        print(f"  offset {self.offset_str()}  ->  {self.metrics_line(d['metrics'])}")
        self.refresh()
        return False

    def nudge(self, axis, sign):
        off = self.offset
        off[axis] = round(off[axis] + sign * NUDGE_STEPS[self.step_i], 6)
        return self.rescore()

    def cycle_step(self):
        self.step_i = (self.step_i + 1) % len(NUDGE_STEPS)
        print(f"  nudge step: {NUDGE_STEPS[self.step_i] * 100:.1f} cm")
        return False

    def reset_offset(self):
        self.offsets[self.keys[self.idx]] = self.start_offsets[self.keys[self.idx]].copy()
        return self.rescore()

    def copy_previous(self):
        if self.idx == 0:
            print("  this is the first scene, there is no previous scene to copy from")
            return False
        self.offsets[self.keys[self.idx]] = self.offsets[self.keys[self.idx - 1]].copy()
        print(f"  copied the offset of {self.keys[self.idx - 1]}")
        return self.rescore()

    def print_offsets(self):
        """Prints ALL scenes so the dictionary can replace SCENE_OFFSETS in evaluate_dataset.py as it is."""
        print("\n# ---- paste into evaluate_dataset.py (replaces SCENE_OFFSETS) " + "-" * 40)
        print("SCENE_OFFSETS = {")
        for i, k in enumerate(self.keys):
            o = np.round(self.offsets[k], 3) + 0.0                  # + 0.0 turns -0.0 into 0.0
            print(f"    {json.dumps(k)}: ({o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f}),{'  # edited' if self.edited(i) else ''}")
        print("}")
        print(f'CALIB_OFFSET_MODE = "{self.mode}"')
        print("# " + "-" * 98)
        return False

    # ---- main loop --------------------------------------------------------------------------
    def run(self):
        vis = self.vis
        vis.create_window(window_name="Dataset inspector", width=1400, height=850)
        vis.register_key_callback(KEY_RIGHT, lambda _: self.go(+1))
        vis.register_key_callback(KEY_LEFT, lambda _: self.go(-1))
        vis.register_key_callback(KEY_UP, lambda _: self.go_category(+1))
        vis.register_key_callback(KEY_DOWN, lambda _: self.go_category(-1))
        for k in LAYERS:
            vis.register_key_callback(ord(k), lambda _, k=k: self.toggle(k))
        vis.register_key_callback(ord("V"), lambda _: self.toggle_style())
        for k, (axis, sign) in NUDGE_KEYS.items():
            vis.register_key_callback(ord(k), lambda _, a=axis, s=sign: self.nudge(a, s))
        vis.register_key_callback(ord("T"), lambda _: self.cycle_step())
        vis.register_key_callback(ord("0"), lambda _: self.reset_offset())
        vis.register_key_callback(ord("C"), lambda _: self.copy_previous())
        vis.register_key_callback(ord("P"), lambda _: self.print_offsets())

        print("\nKeys: Right/Left = next/prev scene | Up/Down = next/prev category | V = voxel solid/wire")
        print("      offset of the shown scene: A/D = x -/+ | S/W = y -/+ | Z/X = z -/+ | T = step | 0 = reset | "
              "C = copy previous scene | P = print ALL offsets")
        for k, (label, _, on) in LAYERS.items():
            print(f"  [{k}] {label}{'  (on at start)' if on else ''}")
        print(f"Voxel {self.voxel * 100:g} cm, tolerance {self.tol_cm:g} cm, offset mode {self.mode}, "
              f"nudge step {NUDGE_STEPS[self.step_i] * 100:.1f} cm")

        self._goto(self.idx)
        opt = vis.get_render_option()
        if opt is not None:
            opt.point_size = 3.0
            opt.background_color = np.array([0.05, 0.05, 0.08])
        vis.run()
        vis.destroy_window()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=ev.DATASET_DIR)
    ap.add_argument("--pipeline", default=ev.PIPELINE_SOURCE)
    ap.add_argument("--category", default=None, help="only categories whose folder name contains this text")
    ap.add_argument("--start", type=int, default=None, help="occluded recording number to start at (e.g. 5)")
    ap.add_argument("--offset-mode", choices=["geometry", "prediction"], default=None)
    ap.add_argument("--voxel-cm", type=float, default=None,
                    help="voxel size in cm for the view (default: VOXEL_SIZE of the live script)")
    ap.add_argument("--tol-cm", type=float, default=None,
                    help="tolerance in cm (default: TOLERANCES_CM[voxel size] of evaluate_dataset.py, else the voxel size)")
    args = ap.parse_args()

    funcs, consts = ev.load_pipeline(args.pipeline)
    if args.voxel_cm is not None:
        consts["VOXEL_SIZE"] = args.voxel_cm / 100.0
    voxel_cm = round(consts["VOXEL_SIZE"] * 100, 6)
    tol_cm = args.tol_cm if args.tol_cm is not None else ev.TOLERANCES_CM.get(voxel_cm, voxel_cm)
    mode = args.offset_mode or ev.CALIB_OFFSET_MODE

    entries = []
    for cat_dir in sorted(p for p in Path(args.dataset).iterdir() if p.is_dir() and not p.name.startswith("_")):
        if args.category and args.category.lower() not in cat_dir.name.lower():
            continue
        for occ_dir, gt_dir in ev.find_pairs(cat_dir):
            entries.append((cat_dir.name, occ_dir, gt_dir))
    if not entries:
        print("No occluded/GT pairs found.")
        return

    start_idx = 0
    if args.start is not None:
        for i, (_, occ, _) in enumerate(entries):
            if int(occ.name.split("_")[1]) == args.start:
                start_idx = i
                break

    print(f"{len(entries)} scene pair(s) in {len(dict.fromkeys(e[0] for e in entries))} categor(ies)")
    Inspector(entries, funcs, consts, start_idx, mode, tol_cm).run()


if __name__ == "__main__":
    main()