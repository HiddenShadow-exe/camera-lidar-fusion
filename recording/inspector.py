# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!                          WARNING                           !!!
# !!! 100% vibe coded. Quickly ran through it so it shouldn't do !!!
# !!! anything too stupid. No guarantees though, and definitely  !!!
# !!! not taking any responsibility for it. Use with caution...  !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

"""
Dataset Inspector
Lets you browse recorded object captures and inspect them in Open3D.

Controls (Open3D window):
  - Mouse drag   : rotate
  - Scroll       : zoom
  - Shift+drag   : pan
  - Q / Escape   : close window and go back to selection
"""

import os
import sys
import json
import glob
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT_DATA_FOLDER = "data"
SLIDESHOW_DURATION_SECONDS = 0.75

def list_recordings(root: str) -> list[dict]:
    """Return a sorted list of {label, number, path} dicts for every recording."""
    recordings = []
    for meta_path in sorted(glob.glob(f"{root}/**/recording_data.json", recursive=True)):
        folder = os.path.dirname(meta_path)
        parts  = folder.replace("\\", "/").split("/")
        label  = parts[-2] if len(parts) >= 2 else "unknown"
        rec_id = parts[-1]
        recordings.append({"label": label, "rec_id": rec_id, "folder": folder})
    return recordings


def load_recording(folder: str) -> dict:
    """Load all artefacts from a recording folder."""
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

    result["lidar_pcd"]  = o3d.io.read_point_cloud(lidar_path)  if os.path.exists(lidar_path)  else None
    result["camera_pcd"] = o3d.io.read_point_cloud(camera_path) if os.path.exists(camera_path) else None
    result["image_path"] = image_path if os.path.exists(image_path) else None

    return result


def build_object_pcd(meta: dict, pcd_type: str) -> o3d.geometry.PointCloud | None:
    """Reconstruct the object point cloud from the JSON metadata."""
    if pcd_type == "camera":
        label = "3d_data_camera"
    elif pcd_type == "lidar":
        label = "3d_data_lidar"
    else:
        return None

    try:
        pts = meta[label]["object_points"]
        if not pts:
            return None
        arr = np.array(pts, dtype=np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(arr[:, :3])
        if pcd_type == "lidar":
            cmap = plt.get_cmap("plasma")
            rgb_intensities = cmap(arr[:, 3])[:, :3] # Slice alpha
            pcd.colors = o3d.utility.Vector3dVector(rgb_intensities)
        return pcd
    except (KeyError, TypeError, ValueError):
        return None


def build_obb(meta: dict) -> o3d.geometry.OrientedBoundingBox | None:
    """Reconstruct the OBB from the JSON metadata."""
    try:
        obb_data = meta["3d_data_lidar"]["obb_3d"]
        center   = np.array(obb_data["center"],     dtype=np.float64)
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
    
def build_oov(meta: dict) -> np.ndarray | None:
    """Reconstruct the Object Orientation Vector (OOV) from JSON metadata."""
    try:
        oov_data = meta["3d_data_lidar"]["object_orientation_vector"]
        if not oov_data:
            return None
        return np.array(oov_data, dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None

def build_bbox(meta: dict) -> np.ndarray | None:
    """Reconstruct the 2D bounding box of the object from JSON metadata."""
    try:
        bbox_data = meta["2d_data"]["bbox_pixels"]
        if not bbox_data:
            return None
        return np.array(bbox_data, dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None

def print_summary(rec: dict, meta: dict) -> None:
    """Print a compact summary of the recording."""
    print("\n" + "═" * 80)
    print(f"  Label      : {rec['label']}")
    print(f"  Recording  : {rec['rec_id']}")
    print(f"  Folder     : {rec['folder']}")

    sm = meta.get("sample_metadata", {})
    print(f"  Timestamp  : {sm.get('timestamp', 'N/A')}")

    d2 = meta.get("2d_data", {})
    print(f"  BBox px    : {d2.get('bbox_pixels', 'N/A')}")
    print(f"  Img dims   : {d2.get('image_dimensions', 'N/A')}")

    d3l = meta.get("3d_data_lidar", {})
    print(f"  LiDAR pts  : {d3l.get('total_point_count', 'N/A')} total  |  {len(d3l.get('object_points', []))} object")

    obb = d3l.get("obb_3d", {})
    if obb:
        c = obb.get("center", [])
        d = obb.get("dimensions", [])
        print(f"  OBB center : {[f'{v:.3f}' for v in c]}")
        print(f"  OBB dims   : {[f'{v:.3f}' for v in d]}")

    d3c = meta.get("3d_data_camera", {})
    print(f"  Camera pts : {d3c.get('total_point_count', 'N/A')} total  |  {len(d3c.get('object_points', []))} object")
    print("═" * 80)


def slideshow(recs: list[dict]) -> None:
    """Automatically loop through all recordings and show them in mode '4' for 3 seconds each."""
    for rec in recs:
        data = load_recording(rec["folder"])
        print_summary(rec, data["meta"])
        print("\n  Opening visualiser …  (close window to return)\n")
        visualise(rec, data, "4")
        # After closing, continue to the next recording

def slideshow_images(recs: list[dict]) -> None:
    """Automatically loop through all recordings and show their overhead images for 1 second each."""
    
    # Turn on interactive mode and create the window ONCE
    plt.ion() 
    plt.figure() 

    for rec in recs:
        data = load_recording(rec["folder"])
        print_summary(rec, data["meta"])
        
        if data["image_path"] is None:
            print("  No overhead image found.")
            continue
        
        img = mpimg.imread(data["image_path"])
        
        # Clear the figure to prepare for the next image
        plt.clf() 
        
        plt.imshow(img)
        plt.title(f"{rec['label']} / {rec['rec_id']}")
        plt.axis("off")
        plt.tight_layout()
        
        # Pause updates the existing window automatically
        plt.pause(SLIDESHOW_DURATION_SECONDS)  

    # Keep the final image open at the very end
    plt.ioff() 
    plt.show()


SHOW_OPTIONS = {
    "1": "Object points only (coloured by intensity)",
    "2": "Object + OBB",
    "3": "Full LiDAR scan + object (highlighted)",
    "4": "Full LiDAR scan + camera + object + OBB",
    "5": "Camera points only",
    "6": "Overhead image",
}

def visualise(rec: dict, data: dict, mode: str) -> None:
    meta             = data["meta"]
    lidar_pcd        = data["lidar_pcd"]
    camera_pcd       = data["camera_pcd"]
    object_pcd_lidar = build_object_pcd(meta, "lidar")
    object_pcd_cam   = build_object_pcd(meta, "camera")
    obb              = build_obb(meta)
    oov              = build_oov(meta)
    bbox             = build_bbox(meta)

    if mode == "6":
        if data["image_path"] is None:
            print("  No overhead image found.")
            return
        
        img = mpimg.imread(data["image_path"])
        fig, ax = plt.subplots(num=f"{rec['label']} / {rec['rec_id']} — overhead")
        ax.imshow(img)

        # Show bounding rect
        x_min, y_min, x_max, y_max = bbox
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                            fill=False, edgecolor='lime', linewidth=2)
        
        center_x = x_min + (x_max - x_min) / 2
        center_y = y_min + (y_max - y_min) / 2

        # Normalize OOV
        dx, dy = oov[0], oov[1]
        mag = np.hypot(dx, dy)
        if mag > 0:
            dx, dy = dx / mag, dy / mag

        arrow_len = 80
        arrow_x = center_x - (dx * arrow_len / 2)
        arrow_y = center_y - (-dy * arrow_len / 2)
        
        # Draw the orientation arrow
        ax.arrow(arrow_x, arrow_y, dx * arrow_len, -dy * arrow_len, 
                    head_width=12, head_length=12, fc='cyan', ec='cyan', linewidth=2)
        
        ax.add_patch(rect)

        plt.axis("off")
        plt.tight_layout()
        plt.show()
        return

    geometries = []
    label = f"{rec['label']}  /  {rec['rec_id']}"

    def colored(pcd, rgb):
        c = o3d.geometry.PointCloud(pcd)
        c.paint_uniform_color(rgb)
        return c

    if mode == "1":
        if object_pcd_lidar is None:
            print("  No object points in metadata.")
            return
        geometries = [object_pcd_lidar]

    elif mode == "2":
        if object_pcd_lidar is None:
            print("  No object points in metadata.")
            return
        geometries = [colored(object_pcd_lidar, [1.0, 0.3, 0.0])]
        if obb:
            obb.color = (1, 0, 0)
            geometries.append(obb)

    elif mode == "3":
        if lidar_pcd:
            geometries.append(colored(lidar_pcd, [0.6, 0.6, 0.6]))
        if object_pcd_lidar:
            geometries.append(colored(object_pcd_lidar, [1.0, 0.3, 0.0]))

    elif mode == "4":
        if lidar_pcd:
            geometries.append(colored(lidar_pcd, [0.6, 0.6, 0.6]))
        if camera_pcd:
            geometries.append(colored(camera_pcd, [0.0, 0.4, 1.0]))
        if object_pcd_lidar:
            geometries.append(colored(object_pcd_lidar, [1.0, 0.3, 0.0]))
        if object_pcd_cam:
            geometries.append(colored(object_pcd_cam, [0.0, 1.0, 0.0]))
        if obb:
            obb.color = (1, 0, 0)
            geometries.append(obb)

    elif mode == "5":
        if camera_pcd is None or object_pcd_cam is None:
            print("  No camera point cloud file found.")
            return
        geometries.append(colored(camera_pcd, [0.0, 0.4, 1.0]))
        geometries.append(colored(object_pcd_cam, [0.0, 1.0, 0.0]))

    if not geometries:
        print("  Nothing to display.")
        return

    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=label, width=1280, height=800)

    for g in geometries:
        vis.add_geometry(g)

    ro = vis.get_render_option()
    ro.point_size = 3.0
    ro.background_color = np.array([0.05, 0.05, 0.1])

    view_ctl = vis.get_view_control()

    view_ctl.set_up(np.array([0.0, 0.0, 1.0]))       # Z-axis points straight UP
    view_ctl.set_front(np.array([-1.0, -1.0, 0.0]))  # Look at the scene from an isometric angle
    view_ctl.set_lookat(np.array([0.0, 0.0, 0.0]))   # Target camera focal point

    vis.run()
    vis.destroy_window()


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def prompt(msg: str) -> str:
    return input(msg).strip().lower()


def select_label(by_label: dict) -> str | None:
    """Returns chosen label, '' on invalid input, None on quit."""
    clear()
    labels = list(by_label.keys())
    print("\n" + "━" * 80)
    print("  DATASET INSPECTOR — Step 1: Select object")
    print("━" * 80)
    for i, label in enumerate(labels, 1):
        count = len(by_label[label])
        print(f"{i:>3}.  {label}  ({count} recording{'s' if count != 1 else ''})")
    print("\n  Q  →  quit")

    choice = prompt("\n  Object: ")
    if choice == "q":
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(labels):
        return labels[int(choice) - 1]
    print("  Invalid selection.")
    return ""


def select_recording(label: str, recs: list) -> dict | None:
    """Returns chosen rec dict, 'back' to go up, None on quit."""
    clear()
    print("\n" + "━" * 80)
    print(f"  DATASET INSPECTOR — Step 2: Select recording  [{label}]")
    print("━" * 80)
    for i, r in enumerate(recs, 1):
        print(f"{i:>3}.  {r['rec_id']}")
    print("\n  B  →  back      Q  →  quit     S  →  slideshow     I  →  images")

    choice = prompt("\n  Recording: ")
    if choice == "q":
        return None
    if choice == "b":
        return "back"
    if choice == "s":
        return "slideshow"
    if choice == "i":
        return "images"
    if choice.isdigit() and 1 <= int(choice) <= len(recs):
        return recs[int(choice) - 1]
    print("  Invalid selection.")
    return ""


def select_view_mode(label: str, rec: dict) -> str | None:
    """Returns mode key, 'back' to go up, None on quit."""
    clear()
    print("\n" + "━" * 80)
    print(f"  DATASET INSPECTOR — Step 3: Select view mode [{label} / {rec['rec_id']}]")
    print("━" * 80)
    for k, v in SHOW_OPTIONS.items():
        print(f"    {k}.  {v}")
    print("\n  B  →  back      Q  →  quit")

    choice = prompt("\n  Mode: ").upper()
    if choice == "Q":
        return None
    if choice == "B":
        return "back"
    if choice in SHOW_OPTIONS:
        return choice
    print("  Invalid selection.")
    return ""


def main():
    recordings = list_recordings(ROOT_DATA_FOLDER)

    if not recordings:
        print(f"No recordings found under '{ROOT_DATA_FOLDER}'. "
              f"Check ROOT_DATA_FOLDER at the top of the script.")
        sys.exit(1)

    by_label: dict[str, list] = {}
    for r in recordings:
        by_label.setdefault(r["label"], []).append(r)

    # step: 1 = label, 2 = recording, 3 = view mode
    step   = 1
    label  = None
    rec    = None

    while True:
        if step == 1:
            result = select_label(by_label)
            if result is None:
                print("  Bye!")
                break
            if result == "":
                continue
            label = result
            step  = 2

        elif step == 2:
            result = select_recording(label, by_label[label])
            if result is None:
                print("  Bye!")
                break
            if result == "":
                continue
            if result == "back":
                step = 1
                continue
            if result == "slideshow":
                slideshow(by_label[label])
                continue
            if result == "images":
                slideshow_images(by_label[label])
                continue
            rec  = result
            step = 3

        elif step == 3:
            result = select_view_mode(label, rec)
            if result is None:
                print("  Bye!")
                break
            if result == "":
                continue
            if result == "back":
                step = 2
                continue

            data = load_recording(rec["folder"])
            print_summary(rec, data["meta"])
            if data["image_path"]:
                print(f"  Overhead image : {data['image_path']}")
            print("\n  Opening visualiser …  (close window to return)\n")
            visualise(rec, data, result)
            # After closing, stay at step 3 to pick another mode for the same recording


if __name__ == "__main__":
    main()