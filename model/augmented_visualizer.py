#!/usr/bin/env python3
"""
Interactive 3D inspector for augmented training data.
Groups augmented/synthetic samples by category and base recording name.
Colors: Original = Green, Augmented = Blue, Synthetic = Orange.

Usage:
    python simple_inspector.py
    
Keys:
    Right / Left : next / previous scene
    Q            : quit
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

# GLFW key codes
KEY_RIGHT, KEY_LEFT = 262, 263
DEFAULT_DATASET = "dataset_aug"

COLORS = {
    "orig": np.array([0.1, 0.8, 0.1]),  # Green
    "aug":  np.array([0.2, 0.6, 1.0]),  # Blue
    "syn":  np.array([1.0, 0.6, 0.0])   # Orange
}


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


class AugmentedInspector:
    def __init__(self, scenes_dict):
        """scenes_dict: { 'train/category/recording_01': [path1, path2, ...], ... }"""
        self.scenes = scenes_dict
        self.scene_names = sorted(list(scenes_dict.keys()))
        self.idx = 0
        
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.initialized = False

    def load_scene(self, step=0):
        # Update index with wrap-around
        self.idx = (self.idx + step) % len(self.scene_names)
        current_scene = self.scene_names[self.idx]
        json_paths = self.scenes[current_scene]

        self.vis.clear_geometries()

        # Add a coordinate frame for reference (Red=X, Green=Y, Blue=Z)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self.vis.add_geometry(frame, reset_bounding_box=not self.initialized)

        counts = {"orig": 0, "aug": 0, "syn": 0, "unknown": 0}

        for jpath in json_paths:
            data = load_json(jpath)
            
            # Determine type from folder suffix
            folder_name = jpath.parent.name
            suffix = folder_name.split("__")[-1]
            
            if suffix.startswith("orig"):
                kind = "orig"
            elif suffix.startswith("aug"):
                kind = "aug"
            elif suffix.startswith("syn"):
                kind = "syn"
            else:
                kind = "unknown"
                
            counts[kind] += 1
            color = COLORS.get(kind, np.array([0.5, 0.5, 0.5]))

            # 1. Load Camera Point Cloud
            points = data.get("3d_data_camera", {}).get("object_points", [])
            if points:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
                pcd.paint_uniform_color(color)
                self.vis.add_geometry(pcd, reset_bounding_box=not self.initialized)

            # 2. Load OBB
            obb_data = data.get("obb_label")
            if obb_data:
                c = np.array(obb_data["center"], dtype=np.float64)
                d = np.array(obb_data["dimensions"], dtype=np.float64)
                R = np.array(obb_data["rotation"], dtype=np.float64).reshape(3, 3)
                
                obb = o3d.geometry.OrientedBoundingBox(c, R, d)
                obb.color = color
                self.vis.add_geometry(obb, reset_bounding_box=not self.initialized)

        print("-" * 60)
        print(f"[{self.idx + 1}/{len(self.scene_names)}] Scene: {current_scene}")
        print(f"  Contents: {counts['orig']} orig (Green) | {counts['aug']} aug (Blue) | {counts['syn']} syn (Orange)")

        self.initialized = True
        self.vis.update_renderer()

    def run(self):
        self.vis.create_window(window_name="Augmented Data Inspector", width=1400, height=850)
        
        # Register key callbacks
        self.vis.register_key_callback(KEY_RIGHT, lambda _: self.load_scene(step=+1) or False)
        self.vis.register_key_callback(KEY_LEFT, lambda _: self.load_scene(step=-1) or False)

        print("\nKeys: Right/Left = next/prev scene | Q = quit | Scroll = zoom | Click+Drag = rotate")
        
        self.load_scene(step=0)
        
        # Make it look nice
        opt = self.vis.get_render_option()
        if opt is not None:
            opt.point_size = 3.0
            opt.background_color = np.array([0.05, 0.05, 0.08])
            
        self.vis.run()
        self.vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D inspector for augmented data.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Root folder of augmented data.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Folder '{dataset_path}' not found.")
        return

    # Group JSONs by category and base recording name
    # e.g., "train/full occlusion boxes/recording_01"
    scenes_dict = {}
    for json_path in dataset_path.rglob("recording_data.json"):
        folder_name = json_path.parent.name
        
        # Extract base name (recording_XX)
        parts = folder_name.split("__")
        if len(parts) != 2:
            continue
        base_name = parts[0]
        
        # Include split and category in the grouping key to ensure uniqueness
        category_dir = json_path.parent.parent
        split_dir = category_dir.parent
        scene_key = f"{split_dir.name}/{category_dir.name}/{base_name}"
        
        if scene_key not in scenes_dict:
            scenes_dict[scene_key] = []
        scenes_dict[scene_key].append(json_path)

    if not scenes_dict:
        print(f"No valid recording_data.json files found in {dataset_path}")
        return

    print(f"Found {len(scenes_dict)} unique scenes.")
    
    # Run the viewer
    inspector = AugmentedInspector(scenes_dict)
    inspector.run()


if __name__ == "__main__":
    main()