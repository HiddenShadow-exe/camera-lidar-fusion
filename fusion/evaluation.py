import os
import numpy as np
from scipy.spatial import cKDTree

VOXEL_RECORD_DIR = "voxel_recordings"

pairs = sorted(f for f in os.listdir(VOXEL_RECORD_DIR) if f.startswith("occluded_"))
if not pairs:
    print("[EVALUATE] No recordings found.")
else:
    def to_voxel_keys(pts, voxel_size):
        return set(map(tuple, np.floor(pts / voxel_size).astype(int)))

    total_precision = 0.0
    valid_pairs = 0
    for fname in pairs:
        idx = fname.split("_")[1]
        v_size = int(fname.split("_")[2].split(".")[0][1:])
        gt_path = os.path.join(VOXEL_RECORD_DIR, f"gt_{idx}_v{v_size}.npy")
        occ_path = os.path.join(VOXEL_RECORD_DIR, fname)
        
        if v_size == 5:
            continue

        if not os.path.exists(gt_path):
            print(f"  [pair {idx}] Missing GT file, skipping.")
            continue

        green_keys = to_voxel_keys(np.load(occ_path), v_size / 100.0)
        gt_keys    = to_voxel_keys(np.load(gt_path), v_size / 100.0)

        hits = green_keys & gt_keys
        tp = len(hits)
        precision = tp / len(green_keys) if green_keys else 0.0
        total_precision += precision

        # Identify predicted voxels that did not hit GT (False Positives)
        extra_keys = green_keys - gt_keys

        if extra_keys and gt_keys:
            # Convert keys back to world coordinates (centered in the voxel)
            gt_pts = np.array(list(gt_keys)) * (v_size / 100.0) + (v_size / 100.0) / 2
            extra_pts = np.array(list(extra_keys)) * (v_size / 100.0) + (v_size / 100.0) / 2

            # Use KDTree for fast "closest point" lookup
            tree = cKDTree(gt_pts)
            distances, _ = tree.query(extra_pts, k=1) 
            avg_dist_error = np.mean(distances)
            max_dist_error = np.max(distances)
        else:
            avg_dist_error = 0.0

        valid_pairs += 1
        print(f"  [Scene {idx} V{v_size}] Predicted: {len(green_keys)} | GT hits: {tp} | Precision: {precision:.2%} | Avg Dist Error: {avg_dist_error * 100:.1f}cm | Max Dist Error: {max_dist_error * 100:.1f}cm")

    print(f"[EVALUATE] Avg precision across {valid_pairs} scenes: {total_precision / valid_pairs:.2%}")
