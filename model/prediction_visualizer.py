#!/usr/bin/env python3
"""
Evaluation Inspector for Trained OBBNet Model (matching train_obb.py).
Visualizes the object point cloud, Ground Truth OBB (Green), and Predicted OBB (Red).

Usage:
    python evaluate_model.py --dataset dataset_aug --weights obb_model.pth

Keys:
    Right / Left : next / previous validation sample
    Q            : quit
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

# Allow importing helpers from train_obb.py if in the same folder
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_obb import (  # noqa: E402
    OBBNet,
    UNIT,
    cam_center,
    decode,
    iou3d,
    prepare,
    to_world,
)

# GLFW key codes
KEY_RIGHT, KEY_LEFT = 262, 263


class EvaluationInspector:
    def __init__(self, dataset_path, weights_path, floor_z_override=None, orig_only=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            width = ckpt.get("width", 64)
            dropout = ckpt.get("dropout", 0.2)
            self.n_points = ckpt.get("n_points", 1024)
            self.floor_z = floor_z_override if floor_z_override is not None else ckpt.get("floor_z", -0.4)
        else:
            # Fallback if a plain state_dict was saved
            state_dict = ckpt
            width, dropout = 64, 0.2
            self.n_points = 1024
            self.floor_z = floor_z_override if floor_z_override is not None else -0.4

        self.model = OBBNet(width=width, dropout=dropout).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(
            f"Loaded {weights_path} on {self.device} "
            f"(width={width}, n_points={self.n_points}, floor_z={self.floor_z})"
        )

        val_root = Path(dataset_path) / "val"
        all_paths = sorted(val_root.glob("*/*/recording_data.json"))
        if orig_only:
            filtered = []
            for p in all_paths:
                with open(p, "r") as f:
                    doc = json.load(f)
                aug_type = doc.get("sample_metadata", {}).get("augmentation", {}).get("type", "orig")
                if aug_type == "orig":
                    filtered.append(p)
            self.paths = filtered
        else:
            self.paths = all_paths

        if not self.paths:
            raise RuntimeError(f"No validation samples found in {val_root}/")
        print(f"Found {len(self.paths)} validation samples.")

        self.idx = 0
        self.rng = np.random.default_rng(0)
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.initialized = False

    def load_scene(self, step=0):
        self.idx = (self.idx + step) % len(self.paths)
        path = self.paths[self.idx]

        with open(path, "r") as f:
            data = json.load(f)

        raw_points = np.asarray(data["3d_data_camera"]["object_points"], dtype=np.float64).reshape(-1, 3)
        if len(raw_points) < 3:
            print(f"Skipping {path.parent.name}: too few points ({len(raw_points)}).")
            return

        cam = cam_center(data["calibration"])
        item, origin, psi = prepare(raw_points, cam, self.floor_z, self.n_points, self.rng)

        with torch.no_grad():
            pts_t = torch.as_tensor(item["pts"], dtype=torch.float32, device=self.device).unsqueeze(0)
            cond_t = torch.as_tensor(item["cond"], dtype=torch.float32, device=self.device).unsqueeze(0)
            prior_c_t = torch.as_tensor(item["prior_c"], dtype=torch.float32, device=self.device).unsqueeze(0)
            prior_d_t = torch.as_tensor(item["prior_d"], dtype=torch.float32, device=self.device).unsqueeze(0)

            dc, dd, dy = self.model(pts_t, cond_t)
            pc_dm, pd_dm, py_can = decode(
                {"prior_c": prior_c_t, "prior_d": prior_d_t}, dc, dd, dy
            )
            pred_c, pred_d, pred_yaw = to_world(
                origin, psi, pc_dm[0].cpu().numpy(), pd_dm[0].cpu().numpy(), py_can[0].item()
            )

        # Rotation matrix in world/camera frame for predicted box (yaw around Z)
        cy, sy = math.cos(pred_yaw), math.sin(pred_yaw)
        pred_R = np.array(
            [
                [cy, -sy, 0.0],
                [sy, cy, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Ground truth
        gt_obb = data["obb_label"]
        gt_c = np.asarray(gt_obb["center"], dtype=np.float64)
        gt_R_raw = np.asarray(gt_obb["rotation"], dtype=np.float64).reshape(3, 3)
        gt_d_raw = np.asarray(gt_obb["dimensions"], dtype=np.float64)

        # Match train_obb.py axis convention for 3D IoU calculation
        vi = int(np.argmax(np.abs(gt_R_raw[2, :])))
        a0, a1 = [k for k in range(3) if k != vi]
        gt_d_eval = np.array([gt_d_raw[a0], gt_d_raw[a1], gt_d_raw[vi]], dtype=np.float64)
        gt_yaw_eval = math.atan2(gt_R_raw[1, a0], gt_R_raw[0, a0])

        iou = iou3d(pred_c, pred_d, pred_yaw, gt_c, gt_d_eval, gt_yaw_eval)
        center_err_cm = np.linalg.norm(pred_c - gt_c) * 100.0
        aug_type = data.get("sample_metadata", {}).get("augmentation", {}).get("type", "orig")

        print("-" * 60)
        print(f"[{self.idx + 1}/{len(self.paths)}] {path.parent.parent.name}/{path.parent.name} ({aug_type})")
        print(f"3D IoU:            {iou:.4f}")
        print(f"Center Dist Error: {center_err_cm:.1f} cm")
        print(f"Pred dims (m):     [{pred_d[0]:.3f}, {pred_d[1]:.3f}, {pred_d[2]:.3f}]")
        print(f"GT dims (m):       [{gt_d_raw[0]:.3f}, {gt_d_raw[1]:.3f}, {gt_d_raw[2]:.3f}]")

        # Update Open3D geometries
        self.vis.clear_geometries()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_points)
        pcd.paint_uniform_color([0.6, 0.8, 1.0])
        self.vis.add_geometry(pcd, reset_bounding_box=not self.initialized)

        # Ground Truth Box (Green)
        o3d_gt_box = o3d.geometry.OrientedBoundingBox(gt_c, gt_R_raw, gt_d_raw)
        o3d_gt_box.color = (0.1, 0.8, 0.1)
        self.vis.add_geometry(o3d_gt_box, reset_bounding_box=not self.initialized)

        # Predicted Box (Red)
        o3d_pred_box = o3d.geometry.OrientedBoundingBox(pred_c, pred_R, pred_d)
        o3d_pred_box.color = (0.9, 0.1, 0.1)
        self.vis.add_geometry(o3d_pred_box, reset_bounding_box=not self.initialized)

        self.initialized = True
        self.vis.update_renderer()

    def run(self):
        self.vis.create_window(window_name="OBBNet Evaluation", width=1400, height=850)
        self.vis.register_key_callback(KEY_RIGHT, lambda _: self.load_scene(step=+1) or False)
        self.vis.register_key_callback(KEY_LEFT, lambda _: self.load_scene(step=-1) or False)

        print("\nKeys: Right/Left = next/prev sample | Q = quit | Scroll = zoom | Click+Drag = rotate")
        self.load_scene(step=0)

        opt = self.vis.get_render_option()
        if opt is not None:
            opt.point_size = 4.0
            opt.background_color = np.array([0.05, 0.05, 0.08])

        self.vis.run()
        self.vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Inspect OBBNet predictions vs Ground Truth.")
    parser.add_argument("--dataset", default="dataset_aug", help="Root folder of dataset")
    parser.add_argument("--weights", default="obb_model.pth", help="Path to trained checkpoint")
    parser.add_argument("--floor-z", type=float, default=None, help="Override floor_z from checkpoint")
    parser.add_argument("--orig-only", action="store_true", help="Show only un-augmented ('orig') val samples")
    args = parser.parse_args()

    inspector = EvaluationInspector(
        args.dataset,
        args.weights,
        floor_z_override=args.floor_z,
        orig_only=args.orig_only,
    )
    inspector.run()


if __name__ == "__main__":
    main()