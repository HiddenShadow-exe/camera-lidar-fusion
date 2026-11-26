#!/usr/bin/env python3
"""Synthetic dataset in EXACTLY your layout, to dry-run the pipeline without real data:

    <out>/<category>/recording_01/{recording_data.json, pointcloud_lidar.pcd}    odd  = occluder + target
    <out>/<category>/recording_02/{recording_data.json, pointcloud_lidar.pcd}    even = target only (GT)

A Velodyne-like LiDAR sits at the origin (height 0.4 m over the floor), an overhead camera (your intrinsics) looks
down from 2.25 m above the LiDAR. The target's camera cluster is ray-cast through the real pixel grid.
The true target box is stored under "synthetic_truth" (never read by the pipeline) so you can check the oracle.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import R_C2L, Rig, box_sdf, ray_box_interval, rot_z, write_pcd_ascii

K = [[389.344, 0, 319.966], [0, 389.344, 244.377], [0, 0, 1]]
FLOOR = -0.40


def lidar_scan(boxes, rng, az_phase):
    az = np.deg2rad(np.arange(-60, 60, 0.30) + az_phase)
    el = np.deg2rad(np.linspace(-45, 45, 96))
    AZ, EL = np.meshgrid(az, el)
    d = np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], -1).reshape(-1, 3)
    best = np.full(len(d), np.inf)
    ident = np.full(len(d), -1)
    dz = d[:, 2]
    tf = np.where(dz < -1e-9, FLOOR / np.where(dz < -1e-9, dz, 1.0), np.inf)
    best, ident = tf.copy(), np.where(np.isfinite(tf), 0, -1)
    for bi, (c, dims, R) in enumerate(boxes, start=1):
        tin, tout = ray_box_interval((np.zeros(3) - c) @ R, d @ R, -dims / 2, dims / 2)
        hit = (tout >= tin) & (tin > 0) & (tin < best)
        best, ident = np.where(hit, tin, best), np.where(hit, bi, ident)
    keep = np.isfinite(best) & (best < 5.0)
    t = best[keep] + rng.normal(0, 0.004, keep.sum())
    return d[keep] * t[:, None], ident[keep]


def camera_cluster(rig, box, rng):
    c, dims, R = box
    H, W = rig.hw
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    uu, vv = uu.ravel().astype(float), vv.ravel().astype(float)
    dirs = rig.pixel_dirs(uu, vv)
    tin, tout = ray_box_interval((rig.cam_center - c) @ R, dirs @ R, -dims / 2, dims / 2)
    hit = (tout >= tin) & (tin > 0)
    p = rig.cam_center + tin[hit][:, None] * dirs[hit]
    z = np.maximum(rig.t[2] - p[:, 2], 0.5)
    p = p + (p - rig.cam_center) / z[:, None] * rng.normal(0, 0.0015 * z, len(p))[:, None]
    return p


def write_rec(folder, n, scan, obj_pts, cam_pts, rig_params, hw=(480, 640)):
    folder.mkdir(parents=True, exist_ok=True)
    doc = {"sample_metadata": {"label": "occluded boxes", "recording_number": n},
           "2d_data": {"image_dimensions": list(hw)},
           "3d_data_lidar": {"total_point_count": len(scan),
                             "object_points": np.hstack([obj_pts, np.full((len(obj_pts), 1), 0.05)]).round(5).tolist()},
           "3d_data_camera": {"total_point_count": len(cam_pts), "object_points": cam_pts.round(5).tolist()},
           "calibration": {"camera_intrinsics": K, "lidar_to_cam_extrinsic": rig_params}}
    return doc, folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_synth")
    ap.add_argument("--n-pairs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cats = ["full occlusion boxes", "partial occlusion boxes"]
    out = Path(a.out)
    for p in range(a.n_pairs):
        cat = cats[p % 2]
        yaw_cal = float(rng.uniform(-0.1, 0.1))
        params = {"lidar_position": [-0.878 + rng.normal(0, 0.005), -0.022, 2.25], "lidar_yaw": yaw_cal}
        rig = Rig.from_calib({"camera_intrinsics": K, "lidar_to_cam_extrinsic": params}, (480, 640))
        dims = np.array([rng.uniform(0.25, 0.45), rng.uniform(0.25, 0.45), rng.uniform(0.25, 0.38)])
        xy = np.array([rng.uniform(1.0, 1.4), rng.uniform(-0.25, 0.25)])
        target = (np.array([xy[0], xy[1], FLOOR + dims[2] / 2]), dims, rot_z(rng.uniform(-0.7, 0.7)))
        ang = np.arctan2(xy[1], xy[0])
        od = np.array([0.22, rng.uniform(0.5, 0.75), rng.uniform(0.3, 0.4)])
        frac = rng.uniform(0.40, 0.55) if cat.startswith("full") else rng.uniform(0.40, 0.55)
        shift_y = 0.0 if cat.startswith("full") else rng.uniform(0.15, 0.3)
        oc = np.array([frac * xy[0], frac * xy[1] + shift_y, FLOOR + od[2] / 2])
        occluder = (oc, od, rot_z(ang))
        n_odd = 2 * p + 1
        scan_o, id_o = lidar_scan([occluder, target], rng, rng.uniform(0, 0.3))
        scan_e, id_e = lidar_scan([target], rng, rng.uniform(0, 0.3))
        cam = camera_cluster(rig, target, rng)
        for n, scan, obj, camp in ((n_odd, scan_o, np.zeros((1, 3)), cam), (n_odd + 1, scan_e, scan_e[id_e == 1], cam)):
            doc, folder = write_rec(out / cat / f"recording_{n:02d}", n, scan, obj, camp, params)
            doc["synthetic_truth"] = {"center": target[0].tolist(), "dimensions": target[1].tolist(),
                                      "rotation": target[2].reshape(-1).tolist()}
            (folder / "recording_data.json").write_text(json.dumps(doc))
            write_pcd_ascii(folder / "pointcloud_lidar.pcd", scan)
        print(f"pair {p + 1}/{a.n_pairs} [{cat}]: target {np.round(dims, 2)} at {np.round(xy, 2)}, "
              f"occluder {np.round(od, 2)}; odd scan {len(scan_o)} pts, even target pts {int((id_e == 1).sum())}")
    print(f"written to {out.resolve()}")


if __name__ == "__main__":
    main()
