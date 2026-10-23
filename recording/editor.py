# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!                          WARNING                           !!!
# !!! 100% vibe coded. Quickly ran through it so it shouldn't do !!!
# !!! anything too stupid. No guarantees though, and definitely  !!!
# !!! not taking any responsibility for it. Use with caution...  !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

"""
Dataset Editor
Lets you browse and edit recorded object captures.

Editor Modes:
  1. 2D BBox Editor : Use keyboard to fine-tune 2D bounding box edges.
  2. 3D OBB Editor  : Use keyboard to push/pull the 6 faces of the 3D OBB.
                      Simultaneously shows the 2D projection on the image.
  3. 3D Point Editor: Shift-click to select points, add/remove chunks.
"""

from __future__ import annotations

import os
import sys
import json
import glob
import time
import copy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.luminance import calculate_luminance

ROOT_DATA_FOLDER = "dataset"

# ==============================================================================
# I/O & HELPERS
# ==============================================================================

def get_highlighted_pcd(pcd, meta):
    """Colors object points red, everything else gray using KD-Tree."""
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.paint_uniform_color([0.5, 0.5, 0.5]) # Default gray
    
    obj_pts = np.array(meta["3d_data_lidar"].get("object_points", []))
    if len(obj_pts) > 0:
        colors = np.asarray(pcd_copy.colors)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd_copy)
        for op in obj_pts:
            # Find nearest neighbor for each object point coordinate
            _, idx, _ = pcd_tree.search_knn_vector_3d(np.array(op[:3], dtype=np.float64), 1)
            colors[idx[0]] = [1.0, 0.0, 0.0] # Red
        pcd_copy.colors = o3d.utility.Vector3dVector(colors)
    return pcd_copy

def list_recordings(root: str) -> list[dict]:
    recordings = []
    for meta_path in sorted(glob.glob(f"{root}/**/recording_data.json", recursive=True)):
        folder = os.path.dirname(meta_path)
        parts  = folder.replace("\\", "/").split("/")
        label  = parts[-2] if len(parts) >= 2 else "unknown"
        rec_id = parts[-1]
        recordings.append({"label": label, "rec_id": rec_id, "folder": folder})
    return recordings

def load_recording(folder: str) -> dict:
    meta_path   = os.path.join(folder, "recording_data.json")
    lidar_path  = os.path.join(folder, "pointcloud_lidar.pcd")
    camera_path = os.path.join(folder, "pointcloud_camera.pcd")
    image_path  = os.path.join(folder, "image_overhead.png")

    result = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            result["meta"] = json.load(f)
    else:
        result["meta"] = {}

    result["meta_path"] = meta_path
    result["lidar_pcd"]  = o3d.io.read_point_cloud(lidar_path)  if os.path.exists(lidar_path)  else None
    result["camera_pcd"] = o3d.io.read_point_cloud(camera_path) if os.path.exists(camera_path) else None
    result["image_path"] = image_path if os.path.exists(image_path) else None

    return result

def save_metadata(meta_path: str, meta: dict):
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"  [INFO] Saved updates to {meta_path}")

def build_obb(meta: dict) -> o3d.geometry.OrientedBoundingBox | None:
    try:
        obb_data = meta["3d_data_lidar"]["obb_3d"]
        center   = np.array(obb_data["center"],    dtype=np.float64)
        extent   = np.array(obb_data["dimensions"], dtype=np.float64)
        rot_flat = np.array(obb_data["rotation"],   dtype=np.float64)

        obb = o3d.geometry.OrientedBoundingBox()
        obb.center = center
        obb.extent = extent
        if len(rot_flat) == 9:
            obb.R = rot_flat.reshape(3, 3)
        elif len(rot_flat) == 4:
            qw, qx, qy, qz = rot_flat
            obb.R = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
        return obb
    except (KeyError, TypeError, ValueError):
        return None

def project_3d_to_2d(points_3d, meta):
    """
    Inverse of screen_to_world.
    Projects a Nx3 array of world-space points into 2D image coordinates (u, v).
    """
    intrinsics_matrix = meta["calibration"]["camera_intrinsics"]
    if intrinsics_matrix is None:
        return np.zeros((points_3d.shape[0], 2))

    intrinsics = {
        'fx': float(intrinsics_matrix[0][0]),
        'fy': float(intrinsics_matrix[1][1]),
        'ppx': float(intrinsics_matrix[0][2]),
        'ppy': float(intrinsics_matrix[1][2]),
        'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]
    }

    lidar_yaw = meta["calibration"]["lidar_to_cam_extrinsic"]["lidar_yaw"]
    lidar_pos = np.array(meta["calibration"]["lidar_to_cam_extrinsic"]["lidar_position"], dtype=np.float64)

    print(intrinsics) # Looking good
    print(lidar_yaw) # in radians
    
    cos_y = np.cos(lidar_yaw)
    sin_y = np.sin(lidar_yaw)
    
    # Forward rotation matrices from your screen_to_world function
    R_yaw = np.array([[cos_y, -sin_y, 0],
                      [sin_y,  cos_y, 0],
                      [  0,      0,   1]], dtype=np.float64)
                      
    R_cam = np.array([[ 1,  0,  0],
                      [ 0, -1,  0],
                      [ 0,  0, -1]], dtype=np.float64)
                      
    # Invert the rotations: 
    # (vec_cam @ R_cam.T @ R_yaw.T) is inverted by right-multiplying by R_yaw then R_cam
    pts_cam = (points_3d @ R_cam) + lidar_pos
    
    # Prevent division by zero if z is exactly 0
    z = pts_cam[:, 2]
    z_safe = np.where(z == 0, 1e-6, z)
    
    # Invert the intrinsic equations:
    # dx = (u - ppx) * z / fx  =>  u = (dx * fx / z) + ppx
    u = (pts_cam[:, 0] * intrinsics['fx'] / z_safe) + intrinsics['ppx']
    v = (pts_cam[:, 1] * intrinsics['fy'] / z_safe) + intrinsics['ppy']
    
    return np.column_stack((u, v))


# ==============================================================================
# EDITORS
# ==============================================================================

def edit_2d_bbox(rec: dict, data: dict):
    meta = data["meta"]
    if data["image_path"] is None or "bbox_pixels" not in meta.get("2d_data", {}):
        print("  [ERROR] Missing image or bbox data.")
        return

    img = mpimg.imread(data["image_path"])
    bbox = list(meta["2d_data"]["bbox_pixels"]) # [xmin, ymin, xmax, ymax]

    print("\n" + "="*50)
    print(" 2D BBOX EDITOR CONTROLS")
    print(" - 'w' / 's' : Move TOP edge up/down")
    print(" - 'e' / 'd' : Move BOTTOM edge up/down")
    print(" - 'a' / 'd' (or 'f'): Move LEFT edge left/right") # using custom mapping
    print(" - 'j' / 'k' : Move RIGHT edge left/right")
    print(" - 'enter'   : Save and Exit")
    print("="*50 + "\n")

    fig, ax = plt.subplots(num=f"Editing 2D BBox: {rec['label']} / {rec['rec_id']}")
    ax.imshow(img)
    rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor='lime', linewidth=1)
    ax.add_patch(rect)

    def update_rect():
        rect.set_x(bbox[0])
        rect.set_y(bbox[1])
        rect.set_width(bbox[2] - bbox[0])
        rect.set_height(bbox[3] - bbox[1])
        fig.canvas.draw()

    def on_key(event):
        step = 2
        # Left edge
        if event.key == 'left': bbox[0] -= step
        elif event.key == 'right': bbox[0] += step
        # Right edge
        elif event.key == 'shift+left': bbox[2] -= step
        elif event.key == 'shift+right': bbox[2] += step
        # Top edge
        elif event.key == 'up': bbox[1] -= step
        elif event.key == 'down': bbox[1] += step
        # Bottom edge
        elif event.key == 'shift+up': bbox[3] -= step
        elif event.key == 'shift+down': bbox[3] += step
        # Save
        elif event.key == 'enter':
            meta["2d_data"]["bbox_pixels"] = bbox

            # Convert RGB to BGR
            final_image = img[:, :, ::-1]

            # Convert to 0-255
            if final_image.max() <= 1.0 or final_image.dtype in [np.float32, np.float64]:
                final_image = (final_image * 255).astype(np.uint8)
            else:
                final_image = final_image.astype(np.uint8)

            # Upadate luminance
            lum_total, lum_object = calculate_luminance(final_image, bbox)
            meta["sample_metadata"]["mean_luminance_total"] = lum_total
            meta["sample_metadata"]["mean_luminance_object"] = lum_object

            save_metadata(data["meta_path"], meta)
            plt.close(fig)
            return
            
        update_rect()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def edit_3d_obb(rec: dict, data: dict):
    meta = data["meta"]
    obb = build_obb(meta)
    pcd = data["lidar_pcd"]
    img = mpimg.imread(data["image_path"]) if data["image_path"] else None

    if obb is None or pcd is None or img is None:
        print("  [ERROR] Missing OBB, LiDAR, or Image data.")
        return

    # Highlight object points in red and background points in gray
    pcd = get_highlighted_pcd(pcd, meta)

    # Setup non-blocking matplotlib window for 2D projection
    plt.ion()
    fig, ax = plt.subplots(num="2D Projection View")
    ax.imshow(img)
    lines = [ax.plot([], [], 'r-', linewidth=1.5)[0] for _ in range(12)]
    
    # OBB Edge indices mapping
    edges = [
        # X-axis aligned edges
        [0, 1], [2, 7], [3, 6], [5, 4],
        # Y-axis aligned edges
        [0, 2], [1, 7], [3, 5], [6, 4],
        # Z-axis aligned edges
        [0, 3], [1, 6], [2, 5], [7, 4]
    ]

    def update_2d_view(current_obb):
        corners_3d = np.asarray(current_obb.get_box_points())
        corners_2d = project_3d_to_2d(corners_3d, meta)

        print(corners_2d)
        
        for i, (start, end) in enumerate(edges):
            x = [corners_2d[start, 0], corners_2d[end, 0]]
            y = [corners_2d[start, 1], corners_2d[end, 1]]
            lines[i].set_data(x, y)
        
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Initial 2D draw
    update_2d_view(obb)

    print("\n" + "="*60)
    print(" 3D OBB EDITOR CONTROLS (Ensure Open3D window is focused)")
    print(" [Top Key = Expand Outward | Bottom Key = Shrink Inward]")
    print(" - Local X Axis (+X / -X): 'Q'/'A' (+X Face) | 'W'/'S' (-X Face)")
    print(" - Local Y Axis (+Y / -Y): 'E'/'D' (+Y Face) | 'R'/'F' (-Y Face)")
    print(" - Local Z Axis (+Z / -Z): 'T'/'G' (+Z Face) | 'Y'/'H' (-Z Face)")
    print(" - [ENTER]               : Save and Exit")
    print("="*60 + "\n")

    obb.color = (0, 1, 0) # Green OBB lines contrast better over red/gray points
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="3D OBB Editor", width=1280, height=800)
    vis.add_geometry(pcd)
    vis.add_geometry(obb)

    step = 0.01 # 1cm per keystroke

    def adjust_face(axis, direction, sign):
        # direction: 1 for outward, -1 for inward
        # sign: 1 for positive face, -1 for negative face
        delta = step * direction
        
        # 1. Update Extent (Safely circumventing read-only array)
        new_extent = np.copy(np.asarray(obb.extent))
        new_extent[axis] += delta
        obb.extent = new_extent
        
        # 2. Update Center (Safely circumventing read-only array)
        shift_local = np.array([0., 0., 0.])
        shift_local[axis] = (delta / 2.0) * sign
        
        # FIX: Copy the center array to avoid ValueError: output array is read-only
        new_center = np.copy(np.asarray(obb.center))
        new_center += obb.R @ shift_local
        obb.center = new_center
        
        vis.update_geometry(obb)
        update_2d_view(obb)
        return False

    # Logical Keybindings (Q/A, W/S for X | E/D, R/F for Y | T/G, Y/H for Z)
    # +X Face
    vis.register_key_callback(ord('Q'), lambda v: adjust_face(0,  1,  1)) # Expand
    vis.register_key_callback(ord('A'), lambda v: adjust_face(0, -1,  1)) # Shrink
    # -X Face
    vis.register_key_callback(ord('W'), lambda v: adjust_face(0,  1, -1)) # Expand
    vis.register_key_callback(ord('S'), lambda v: adjust_face(0, -1, -1)) # Shrink
    
    # +Y Face
    vis.register_key_callback(ord('E'), lambda v: adjust_face(1,  1,  1)) # Expand
    vis.register_key_callback(ord('D'), lambda v: adjust_face(1, -1,  1)) # Shrink
    # -Y Face
    vis.register_key_callback(ord('R'), lambda v: adjust_face(1,  1, -1)) # Expand
    vis.register_key_callback(ord('F'), lambda v: adjust_face(1, -1, -1)) # Shrink
    
    # +Z Face (Top)
    vis.register_key_callback(ord('T'), lambda v: adjust_face(2,  1,  1)) # Expand
    vis.register_key_callback(ord('G'), lambda v: adjust_face(2, -1,  1)) # Shrink
    # -Z Face (Bottom)
    vis.register_key_callback(ord('Y'), lambda v: adjust_face(2,  1, -1)) # Expand
    vis.register_key_callback(ord('H'), lambda v: adjust_face(2, -1, -1)) # Shrink

    keep_running = [True]

    def save_and_close(v):
        meta["3d_data_lidar"]["obb_3d"]["center"] = obb.center.tolist()
        meta["3d_data_lidar"]["obb_3d"]["dimensions"] = obb.extent.tolist()
        save_metadata(data["meta_path"], meta)
        print(f"[{rec}] Saved 3D OBB parameters!")
        keep_running[0] = False
        return False

    def crop_points_to_obb(v):
        obj_pts_list = meta["3d_data_lidar"].get("object_points", [])
        obj_pts_set = set(tuple(p[:3]) for p in obj_pts_list)

        pts = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        full_pcd = data["lidar_pcd"]
        full_colors = np.asarray(full_pcd.colors).copy()
        
        # Get indices of all points currently falling inside the 3D Oriented Bounding Box
        indices = obb.get_point_indices_within_bounding_box(pcd.points)
        
        if not indices:
            print("  [INFO] No points found inside the current OBB volume.")
            return

        add_count = 0
        for idx in indices:
            point_coords = pts[idx].tolist()
            point_tuple = tuple(point_coords[:3])
            
            if point_tuple not in obj_pts_set:
                intensity = float(full_colors[idx, 0])

                final_point = [float(point_coords[0]), float(point_coords[1]), float(point_coords[2]), intensity]

                obj_pts_list.append(final_point)
                obj_pts_set.add(point_tuple)

                colors[idx] = [1.0, 0.0, 0.0] # Red
                add_count += 1

        remove_count = 0

        # Create an instant spatial lookup table out of the points
        pts_dict = {tuple(np.round(item[:3], 5)): item for item in obj_pts_list}

        for idx, p in enumerate(pts.tolist()):
            p_tuple_unrounded = tuple(p)             # Use unrounded for your original set
            p_tuple_rounded = tuple(np.round(p, 5))  # Use rounded for the dict lookup
            
            if p_tuple_unrounded in obj_pts_set and idx not in indices:
                # Find and remove the actual 4D item by matching its first 3 elements
                item = pts_dict.get(p_tuple_rounded)
                if item is not None:
                    obj_pts_list.remove(item)
                    obj_pts_set.remove(tuple(p)) 
                
                colors[idx] = [0.5, 0.5, 0.5] # Gray
                remove_count += 1

        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(pcd)
        
        print(f"  [SUCCESS] Added {add_count} and removed {remove_count} points.")
        print(f"  [INFO] Total object points: {len(obj_pts_list)}")

    # GLFW Keycode for ENTER is 257 for SPACE it's 32
    vis.register_key_callback(257, save_and_close)
    vis.register_key_callback(32, crop_points_to_obb)


    # Safe Event Loop prevents segfaults
    while keep_running[0]:
        if not vis.poll_events():
            break
        vis.update_renderer()
        plt.pause(0.01) # Keeps 2D projection window updated

    vis.destroy_window()
    plt.close(fig)
    plt.ioff()


def edit_points(rec: dict, data: dict):
    meta = data["meta"]
    full_pcd = data["lidar_pcd"]

    if full_pcd is None or "object_points" not in meta.get("3d_data_lidar", {}):
        print("  [ERROR] Missing LiDAR or object point data.")
        return
    
    full_colors = np.asarray(full_pcd.colors).copy()

    pcd = get_highlighted_pcd(full_pcd, meta)
    
    # Ensure reference points map correctly to metadata dictionary
    if "object_points" not in meta["3d_data_lidar"]:
        meta["3d_data_lidar"]["object_points"] = []
    obj_pts_list = meta["3d_data_lidar"]["object_points"]

    # We must use a custom event loop and a Matplotlib GUI 
    # because VisualizerWithVertexSelection lacks register_key_callback.
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name="Shift+Click points, then use Control Panel")
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 8.0

    # Matplotlib Control Panel
    plt.ion()
    fig, ax = plt.subplots(figsize=(4, 3), num="Control Panel")
    fig.canvas.manager.set_window_title("Point Editor Controls")
    ax.axis('off')
    
    ax_add = plt.axes([0.1, 0.6, 0.8, 0.2])
    ax_rem = plt.axes([0.1, 0.35, 0.8, 0.2])
    ax_save = plt.axes([0.1, 0.1, 0.8, 0.2])
    
    btn_add = Button(ax_add, 'Add Picked Points')
    btn_rem = Button(ax_rem, 'Remove Picked Points')
    btn_save = Button(ax_save, 'Save & Exit')

    keep_running = True

    def modify_points(add=True):
        picked = vis.get_picked_points()
        if not picked:
            print("No points selected! Shift+Click in 3D window first.")
            return
            
        pts = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        modified = False

        # Convert list to a NumPy array ONCE up front for lightning-fast vector operations
        obj_pts_arr = np.array(obj_pts_list) if obj_pts_list else np.empty((0, 4))
        
        for p_info in picked:
            idx = p_info.index
            target_pt = pts[idx] # numpy array of the clicked point
            
            if add:
                # Check if it already exists using a tight tolerance (e.g., within 1mm)
                exists = False
                if obj_pts_arr.size > 0:
                    exists = np.any(np.linalg.norm(obj_pts_arr[:, :3] - target_pt, axis=1) < 1e-3)
                
                if not exists:
                    intensity = float(full_colors[idx, 0])

                    final_point = [float(target_pt[0]), float(target_pt[1]), float(target_pt[2]), intensity]
                    obj_pts_list.append(final_point)

                    # Update our tracking array so subsequent picked points in this loop see it
                    obj_pts_arr = np.vstack([obj_pts_arr, final_point]) 

                    colors[idx] = [1.0, 0.0, 0.0] # Red for added
                    modified = True
                        
            else:
                # Find any existing object point that is practically identical (within 1mm)
                matching_idx = -1
                if obj_pts_arr.size > 0:
                    distances = np.linalg.norm(obj_pts_arr[:, :3] - target_pt, axis=1)
                    indices = np.where(distances < 1e-3)[0]
                    if indices.size > 0:
                        matching_idx = indices[0]
                
                if matching_idx != -1:
                    obj_pts_list.pop(matching_idx) # Remove by index safely

                    # Sync our temporary tracking array in case multiple points are being removed at once
                    obj_pts_arr = np.delete(obj_pts_arr, matching_idx, axis=0)

                    colors[idx] = [0.5, 0.5, 0.5]  # Gray for removed
                    modified = True

        if modified:
            meta["3d_data_lidar"]["object_points"] = obj_pts_list
            print(f"Points {'added' if add else 'removed'}. Total points: {len(obj_pts_list)}")
            save_metadata(data["meta_path"], meta)
            print(f"[{rec}] Successfully saved {len(obj_pts_list)} object points!")

        vis.clear_picked_points()
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        vis.remove_geometry(pcd, reset_bounding_box=False)
        vis.add_geometry(pcd, reset_bounding_box=False)
        

    def on_add(val): modify_points(True)
    def on_rem(val): modify_points(False)

    def on_save(val):
        nonlocal keep_running
        keep_running = False

    btn_add.on_clicked(on_add)
    btn_rem.on_clicked(on_rem)
    btn_save.on_clicked(on_save)
    print(rec)

    while keep_running:
        if not vis.poll_events():
            break  
        vis.update_renderer()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.05) 

    vis.destroy_window()
    plt.close(fig)
    plt.ioff()

# ==============================================================================
# MENU SELECTION
# ==============================================================================

EDIT_OPTIONS = {
    "1": "Edit 2D Bounding Box (Overhead Image)",
    "2": "Edit 3D OBB (Simultaneous 2D/3D View)",
    "3": "Edit Object Points (Add/Remove chunks)",
}

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def prompt(msg: str) -> str:
    return input(msg).strip().lower()

def select_label(by_label: dict) -> str | None:
    clear()
    labels = list(by_label.keys())
    print("\n" + "━" * 80)
    print("  DATASET EDITOR — Step 1: Select object")
    print("━" * 80)
    for i, label in enumerate(labels, 1):
        count = len(by_label[label])
        print(f"{i:>3}.  {label}  ({count} recording{'s' if count != 1 else ''})")
    print("\n  Q  →  quit")

    choice = prompt("\n  Object: ")
    if choice == "q": return None
    if choice.isdigit() and 1 <= int(choice) <= len(labels):
        return labels[int(choice) - 1]
    return ""

def select_recording(label: str, recs: list) -> dict | None:
    clear()
    print("\n" + "━" * 80)
    print(f"  DATASET EDITOR — Step 2: Select recording  [{label}]")
    print("━" * 80)
    for i, r in enumerate(recs, 1):
        print(f"{i:>3}.  {r['rec_id']}")
    print("\n  B  →  back     Q  →  quit")

    choice = prompt("\n  Recording: ")
    if choice == "q": return None
    if choice == "b": return "back"
    if choice.isdigit() and 1 <= int(choice) <= len(recs):
        return recs[int(choice) - 1]
    return ""

def select_edit_mode(label: str, rec: dict) -> str | None:
    clear()
    print("\n" + "━" * 80)
    print(f"  DATASET EDITOR — Step 3: Select edit mode [{label} / {rec['rec_id']}]")
    print("━" * 80)
    for k, v in EDIT_OPTIONS.items():
        print(f"    {k}.  {v}")
    print("\n  B  →  back     Q  →  quit")

    choice = prompt("\n  Mode: ").upper()
    if choice == "Q": return None
    if choice == "B": return "back"
    if choice in EDIT_OPTIONS: return choice
    return ""

def main():
    recordings = list_recordings(ROOT_DATA_FOLDER)
    if not recordings:
        print(f"No recordings found under '{ROOT_DATA_FOLDER}'.")
        sys.exit(1)

    by_label = {}
    for r in recordings:
        by_label.setdefault(r["label"], []).append(r)

    step = 1
    label = None
    rec = None

    while True:
        if step == 1:
            result = select_label(by_label)
            if result is None: break
            if result == "": continue
            label = result
            step = 2

        elif step == 2:
            result = select_recording(label, by_label[label])
            if result is None: break
            if result == "back":
                step = 1
                continue
            if result == "": continue
            rec = result
            step = 3

        elif step == 3:
            result = select_edit_mode(label, rec)
            if result is None: break
            if result == "back":
                step = 2
                continue
            if result == "": continue

            data = load_recording(rec["folder"])
            
            if result == "1":
                edit_2d_bbox(rec, data)
            elif result == "2":
                edit_3d_obb(rec, data)
            elif result == "3":
                edit_points(rec, data)

if __name__ == "__main__":
    main()