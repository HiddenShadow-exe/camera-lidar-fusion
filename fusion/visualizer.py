import numpy as np
import open3d as o3d
import os
import copy

# --- CONFIGURATION ---
RECORD_DIR = "voxel_recordings"

def points_to_voxel_mesh(points, color, voxel_size):
    """Convert points into a merged mesh of cubes, one per voxel."""
    if len(points) == 0:
        return None

    # Snap to voxel grid centers to ensure perfect alignment
    voxel_indices = np.floor(points / voxel_size).astype(int)
    unique_indices = np.unique(voxel_indices, axis=0)
    centers = (unique_indices + 0.5) * voxel_size 

    combined = o3d.geometry.TriangleMesh()
    box = o3d.geometry.TriangleMesh.create_box(voxel_size, voxel_size, voxel_size)

    for center in centers:
        cube = copy.deepcopy(box)
        # Center the box on the voxel coordinate
        cube.translate(center - np.array([voxel_size / 2] * 3))
        combined += cube

    combined.paint_uniform_color(color)
    combined.compute_vertex_normals()
    return combined

def run_visualizer():
    # 1. Select Scene Index
    try:
        idx_input = input("Enter Scene Index to review (e.g., 0): ")
        idx = int(idx_input)
    except ValueError:
        print("Invalid index entered.")
        return
    
    files = os.listdir(RECORD_DIR)

    v_size = 0

    for f in files:
        if f.startswith(f"occluded_{idx:02d}_v") and f.endswith(".npy"):
            # Extract voxel size from filename
            try:
                v_size = int(f.split("_v")[1].split(".")[0]) / 100.0
                print(f"Found occluded file for scene {idx:02d} with voxel size {v_size * 100}cm.")
                break
            except (IndexError, ValueError):
                continue

    occ_path = os.path.join(RECORD_DIR, f"occluded_{idx:02d}_v{v_size * 100:.0f}.npy")
    gt_path = os.path.join(RECORD_DIR, f"gt_{idx:02d}_v{v_size * 100:.0f}.npy")

    if not os.path.exists(occ_path) or not os.path.exists(gt_path):
        print(f"Error: Could not find pair for scene {idx:02d}")
        print(f"Looked for: {occ_path} and {gt_path}")
        return

    # 2. Load and Voxelize Data
    # We use sets of tuples (voxel grid indices) to perform math
    pred_pts = np.load(occ_path)
    gt_pts = np.load(gt_path)

    def to_keys(pts, voxel_size):
        return set(map(tuple, np.floor(pts / voxel_size).astype(int)))

    pred_keys = to_keys(pred_pts, v_size)
    gt_keys = to_keys(gt_pts, v_size)

    # 3. Categorize Voxels
    # Green: Predicted voxels that exist in Ground Truth (Correct)
    hit_keys = pred_keys & gt_keys
    # Red: Predicted voxels that do NOT exist in Ground Truth (False Positives)
    miss_keys = pred_keys - gt_keys
    # Blue: Ground Truth voxels that were NOT hit by prediction (Remaining Lidar)
    remaining_gt_keys = gt_keys - pred_keys

    def keys_to_pts(keys):
        if not keys: return np.array([])
        return (np.array(list(keys)) + 0.5) * v_size

    hit_pts = keys_to_pts(hit_keys)
    miss_pts = keys_to_pts(miss_keys)
    remaining_pts = keys_to_pts(remaining_gt_keys)

    # 4. Generate Meshes
    geometries = []
    
    # Correct Hits -> GREEN
    mesh_hit = points_to_voxel_mesh(hit_pts, [0.1, 0.8, 0.1], v_size)
    if mesh_hit: geometries.append(mesh_hit)

    # Incorrect Predictions -> RED
    mesh_miss = points_to_voxel_mesh(miss_pts, [0.9, 0.1, 0.1], v_size)
    if mesh_miss: geometries.append(mesh_miss)

    # Remaining LiDAR -> BLUE
    mesh_remain = points_to_voxel_mesh(remaining_pts, [0.1, 0.1, 0.8], v_size)
    if mesh_remain: geometries.append(mesh_remain)

    # 5. UI and Render
    print(f"\n--- Scene {idx:02d} Analysis ---")
    print(f"Green (Hits):      {len(hit_keys)}")
    print(f"Red   (False Pos): {len(miss_keys)}")
    print(f"Blue  (Remaining GT): {len(remaining_gt_keys)}")
    
    precision = len(hit_keys) / len(pred_keys) if pred_keys else 0
    print(f"Precision: {precision:.2%}")
    print("\nOpening Visualizer... (Close window to exit)")

    # Add a coordinate frame for orientation
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(coord_frame)

    o3d.visualization.draw_geometries(geometries, 
                                      window_name=f"Voxel Evaluation - Scene {idx:02d}",
                                      width=1280, height=720)

if __name__ == "__main__":
    if not os.path.exists(RECORD_DIR):
        print(f"Directory '{RECORD_DIR}' not found.")
    else:
        run_visualizer()