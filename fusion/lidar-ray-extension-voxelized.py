import numpy as np
import cv2
import socket
import struct
import pickle
import sys
import os
import open3d as o3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2 # type: ignore
import sensor_msgs_py.point_cloud2 as pc2 # type: ignore
import copy
from scipy.spatial import cKDTree

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.camera_intrinsics import load_camera_intrinsics

# Network Setup
RPI_IP = '192.168.0.10'
PORT = 8485

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(f"Connecting to Depth Camera at {RPI_IP}:{PORT}...")
client_socket.connect((RPI_IP, PORT))
print("Connected! Receiving frames...")

data = b""
payload_size = struct.calcsize("Q")

LIDAR_HEIGHT = 0.65
ARUCO_LIDAR_OFFSET = [0, 0, 0]

CALIBRATION_BASED = False

VOXEL_SIZE = 0.03 # meters

# Evaluation / Recording Setup
MODE = "RECORD_OCCLUDED"   # "RECORD_OCCLUDED" | "RECORD_GT" | "EVALUATE"
VOXEL_RECORD_DIR  = "voxel_recordings"

# Detection parameters
MIN_AREA = 2000        # Minimum contour area (in pixels) to ignore noise
MIN_EXTENT = 0.8       # How rectangular the object must be (1.0 = perfect rectangle)


# Voxel cube mesh builder
def points_to_voxel_mesh(points, color, voxel_size):
    """Convert a list of points into a merged mesh of cubes, one per voxel."""
    if len(points) == 0:
        return o3d.geometry.TriangleMesh()

    # Snap to voxel grid centers
    voxel_indices = np.floor(points / voxel_size).astype(int)
    unique_indices = np.unique(voxel_indices, axis=0)
    centers = (unique_indices + 0.5) * voxel_size  # center of each voxel cube

    combined = o3d.geometry.TriangleMesh()
    box = o3d.geometry.TriangleMesh.create_box(voxel_size, voxel_size, voxel_size)

    for center in centers:
        cube = copy.deepcopy(box)
        cube.translate(center - np.array([voxel_size / 2] * 3))
        combined += cube

    combined.paint_uniform_color(color)
    combined.compute_vertex_normals()
    return combined

def get_box_sidewalls(raw_depth, color_image):
    # --- IMAGE PRE-PROCESSING ---
    hole_mask = (raw_depth == 0).astype(np.uint8)

    # Convert to 8-bit for processing
    depth_8bit = cv2.convertScaleAbs(raw_depth, alpha=0.07)

    # Fill holes using inpainting
    depth_8bit = cv2.inpaint(depth_8bit, hole_mask, 5, cv2.INPAINT_TELEA)

    # --- IMAGE PROCESSING ---

    # 1. Work with RAW depth (mm) for the height mask
    # Apply a light blur to raw depth to handle sensor jitter
    raw_depth_float = raw_depth.astype(np.float32)
    raw_depth_blur = cv2.medianBlur(raw_depth_float, 5)

    # 2. Find the floor in Millimeters
    valid_depths_mm = raw_depth_blur[raw_depth_blur > 500].flatten() # Ignore camera noise < 50cm
    
    if len(valid_depths_mm) > 0:
        hist, bins = np.histogram(valid_depths_mm, bins=100)
        floor_mm = bins[np.argmax(hist)]
        
        # 1. Height Mask
        height_mask = ((raw_depth_blur > 100) & (raw_depth_blur < (floor_mm - 150))).astype(np.uint8) * 255

        # Flatness Mask
        dzdx = cv2.Sobel(raw_depth_blur, cv2.CV_32F, 1, 0, ksize=1)
        dzdy = cv2.Sobel(raw_depth_blur, cv2.CV_32F, 0, 1, ksize=1)
        mag = np.sqrt(dzdx**2 + dzdy**2)
        
        # Binary mask: Black where flat, White where steep (the walls)
        # Raise this number if the box tops are too "holey"
        flat_mask = (mag > 30.0).astype(np.uint8) * 255

        # Erode then dilate to fill small holes in the flat_mask (the box tops)
        flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        # Then erode to get rid of edge noise
        flat_mask = cv2.erode(flat_mask, np.ones((11, 11), np.uint8), iterations=1)

        # 4. COMBINE: Subtract non_flat_mask points from height_mask to get our final box mask
        final_box_mask = cv2.subtract(height_mask, flat_mask)

        # 5. FINAL POLISH
        # A quick 'Open' to remove any tiny stray pixels
        final_box_mask = cv2.morphologyEx(final_box_mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    # Find contours on the cleaned mask
    contours, _ = cv2.findContours(final_box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = raw_depth.shape
    boxes = []

    for cnt in contours:
        # Bridge gaps using a Convex Hull, this "wraps" the points even if the line is broken
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)

        if area < MIN_AREA:
            continue

        # Fit a minimum area rectangle to the contour and compute its extent
        rect = cv2.minAreaRect(hull)
        box_w, box_h = rect[1]
        if box_w * box_h == 0:
            continue

        extent = area / (box_w * box_h)
        
        epsilon = 0.04 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if extent > MIN_EXTENT and len(approx) <= 6:
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # If any of the corners are near the image edge, it's likely a false detection
            if np.any(box < 10) or np.any(box[:, 0] > (raw_depth.shape[1] - 10)) or np.any(box[:, 1] > (raw_depth.shape[0] - 10)):
                continue

            boxes.append(box)

            cv2.drawContours(color_image, [box], 0, (0, 255, 0), 2)
            cv2.drawContours(color_image,    [box], 0, (0, 255, 0), 2)
            for p in box:
                cv2.circle(color_image, tuple(p), 5, (0, 0, 255), -1)
                cv2.circle(color_image,    tuple(p), 5, (0, 0, 255), -1)


    # --- EXTRUSION LOGIC ---
    intr = load_camera_intrinsics()
    
    # Contains: [
    #  [p1, p2, p3],  # 3 corners for one triangle of the sidewall
    #  [p1, p3, p4],  # 3 corners for the other triangle of the sidewall
    # ]
    # Length: number of boxes * 4 sides * 2 triangles per side
    sidewall_tris = []

    for box in boxes:
        # Calculate box depth
        box_center = np.mean(box, axis=0).astype(int)
        cx, cy = int(np.clip(box_center[0], 0, w - 1)), int(np.clip(box_center[1], 0, h - 1))
        box_depth_m = raw_depth[cy, cx] / 1000.0

        if box_depth_m == 0: continue

        # Convert each of the 4 box corners from screen space to 3D
        top_corners_3d = []

        for px, py in box:
            px_c = np.clip(px, 0, w - 1)
            py_c = np.clip(py, 0, h - 1)
            zx = box_depth_m
            xx = (px_c - intr['ppx']) * zx / intr['fx']
            yx = (py_c - intr['ppy']) * zx / intr['fy']
            top_corners_3d.append([xx, yx, zx])

        top_corners_3d = np.array(top_corners_3d)  # (4, 3)

        # Calculate floor depth by sampling a point slightly outside the box
        offset = 15  # pixels outside the box to sample
        floor_samples = []

        for p in box:
            # Sample outside the corners
            sx = int(np.clip(p[0] + offset if p[0] > box_center[0] else p[0] - offset, 0, w - 1))
            sy = int(np.clip(p[1] + offset if p[1] > box_center[1] else p[1] - offset, 0, h - 1))
            d = raw_depth[sy, sx]
            if d > 0: floor_samples.append(d / 1000.0)

        floor_depth = np.median(floor_samples) if floor_samples else np.nanmax(raw_depth) / 1000.0

        # We need box center to calculate normals
        box_center_3d = np.mean(top_corners_3d, axis=0)
        box_center_3d[2] = (box_depth_m + floor_depth) / 2.0

        def enforce_outward(tri):
            """Ensures the triangle normal points away from the center of the box."""
            v0, v1, v2 = np.array(tri[0]), np.array(tri[1]), np.array(tri[2])
            normal = np.cross(v1 - v0, v2 - v0)
            tri_center = (v0 + v1 + v2) / 3.0
            
            # If the normal points toward the center, swap v1 and v2 to flip it
            if np.dot(normal, tri_center - box_center_3d) < 0:
                return [tri[0], tri[2], tri[1]] 
            return tri

        # Get the side rectangles (consisting of 4 points each) by extruding each edge of the box top down to floor_depth
        for i in range(len(top_corners_3d)):
            p1 = top_corners_3d[i]
            p2 = top_corners_3d[(i + 1) % len(top_corners_3d)]

            t1 = [p1, p2, [p2[0], p2[1], floor_depth]]
            t2 = [p1, [p1[0], p1[1], floor_depth], [p2[0], p2[1], floor_depth]]

            sidewall_tris.append(enforce_outward(t1))
            sidewall_tris.append(enforce_outward(t2))

    return sidewall_tris

def ray_intersect_triangles_batch(origins, directions, triangles, eps=1e-7, ignore_first=True):
    """
    Vectorized Möller–Trumbore.
    origins:    (N, 3)
    directions: (N, 3)  unit vectors
    triangles:  (M, 3, 3)  each row: [v0, v1, v2]
    Returns: closest_t (N,), closest_tri (N,), hit_mask (N,)
    """
    N, M = len(directions), len(triangles)
    if N == 0 or M == 0: return np.zeros((0, 3)), np.zeros((0,), dtype=int), np.zeros((0,), dtype=bool)

    v0    = triangles[:, 0]              # (M, 3)
    edge1 = triangles[:, 1] - v0         # (M, 3)
    edge2 = triangles[:, 2] - v0         # (M, 3)

    # Expand to (N, M, 3) for broadcasting
    D  = directions[:, None, :]                              # (N, 1, 3) → broadcast
    O  = origins[:, None, :]                                 # (N, 1, 3)
    E1 = edge1[None, :, :]                                   # (1, M, 3)
    E2 = edge2[None, :, :]                                   # (1, M, 3)
    V0 = v0[None, :, :]                                      # (1, M, 3)

    H  = np.cross(D, E2)                                     # (N, M, 3)
    A  = np.einsum('nmi,nmi->nm', E1, H)                     # (N, M)  dot(E1, H)

    # For backface culling, we can check if A < eps. For no culling, we check abs(A) < eps.
    # invalid = np.abs(A) < eps
    invalid = A < eps

    F  = np.where(invalid, 0.0, 1.0 / A)                     # (N, M)

    S  = O - V0                                              # (N, M, 3)
    U  = F * np.einsum('nmi,nmi->nm', S, H)                  # (N, M)

    Q  = np.cross(S, E1)                                     # (N, M, 3)
    V  = F * np.einsum('nmi,nmi->nm', D, Q)                  # (N, M)
    T  = F * np.einsum('mi,nmi->nm',  edge2, Q)              # (N, M)

    # (N, M) boolean mask of valid hits
    hit = (~invalid) & (U >= 0) & (V >= 0) & (U + V <= 1) & (T > eps)

    # Indices of every hit: ray index n, triangle index m
    ray_idx, tri_idx = np.where(hit)                    # (K,), (K,)

    # No hits at all
    if len(ray_idx) == 0: return np.zeros((0, 3)), np.zeros((0,), dtype=int), np.zeros((0,), dtype=bool)

    # Compute each hit point: origin + t * direction
    t_vals   = T[ray_idx, tri_idx]                      # (K,)

    # If we want all hits, just get all points and return
    if not ignore_first:
        hit_pts  = origins[ray_idx] + t_vals[:, None] * directions[ray_idx]  # (K, 3)
        return hit_pts, ray_idx, tri_idx
    
    # If we want to ignore the first hit
    sort_idx = np.lexsort((t_vals, ray_idx))
    sorted_ray_idx = ray_idx[sort_idx]
    sorted_tri_idx = tri_idx[sort_idx]
    sorted_t_vals  = t_vals[sort_idx]

    # Create a mask to identify the FIRST occurrence of every ray index
    # (Since they are sorted by T, the first occurrence is the closest hit)
    # This identifies the start of a new ray's group of hits
    first_hit_mask = np.concatenate(([True], sorted_ray_idx[1:] != sorted_ray_idx[:-1]))

    # We want to KEEP everything that is NOT the first hit
    keep_mask = ~first_hit_mask

    final_ray_idx = sorted_ray_idx[keep_mask]
    final_t_vals  = sorted_t_vals[keep_mask]

    if len(final_ray_idx) == 0:
        return np.zeros((0, 3)), np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)

    # Calculate hit points for 2nd, 3rd... intersections
    hit_pts = origins[final_ray_idx] + final_t_vals[:, None] * directions[final_ray_idx]

    # Also return the indices of the triangles that were hit
    return hit_pts, final_ray_idx, sorted_tri_idx[keep_mask]

class VelodyneVisualizer(Node):
    def __init__(self):
        super().__init__('velodyne_visualizer')

        # Subscribe to Velodyne PointCloud2 topic
        self.subscription = self.create_subscription(
            PointCloud2,
            '/velodyne_points',
            self.pointcloud_callback,
            10
        )

        self.raw_lidar_points = np.array([])

        # Open3D visualizer setup
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Velodyne Live PointCloud")
        
        self.pcd = o3d.geometry.PointCloud()
        self.mesh = o3d.geometry.TriangleMesh()
        self.is_first_frame = True

        self.camera_pcd = o3d.geometry.PointCloud()
        self.camera_mesh = o3d.geometry.TriangleMesh()
        self.is_first_camera_frame = True

        self.extended_pcd = o3d.geometry.PointCloud()
        self.extended_mesh = o3d.geometry.TriangleMesh()
        self.is_first_extended_frame = True

    def pointcloud_callback(self, msg):
        # Read points as a generator
        points_gen = pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)
        
        # Convert to plain list of [x,y,z]
        points = [[p[0], p[1], p[2]] for p in points_gen]
        if not points:
            return

        # Convert to numpy array
        np_points = np.array(points, dtype=np.float32)

        # Drop points that are too far away (> 5m)
        distances = np.linalg.norm(np_points, axis=1)
        np_points = np_points[distances < 5.0]

        self.raw_lidar_points = np_points

        # Update Open3D point cloud
        self.pcd.points = o3d.utility.Vector3dVector(np_points)
        self.pcd.paint_uniform_color([1.0, 0.0, 0.0])

        voxelized_lidar = node.pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

        new_mesh = points_to_voxel_mesh(np.asarray(voxelized_lidar.points), [1.0, 0.0, 0.0], VOXEL_SIZE)
        node.mesh.vertices  = new_mesh.vertices
        node.mesh.triangles = new_mesh.triangles
        node.mesh.vertex_colors = new_mesh.vertex_colors
        node.mesh.vertex_normals = new_mesh.vertex_normals

        if self.is_first_frame:
            self.vis.add_geometry(self.mesh)
            self.is_first_frame = False

        self.vis.update_geometry(self.mesh)
        self.vis.poll_events()
        self.vis.update_renderer()

        # Check if window closed by user
        if not self.vis.poll_events():
            print("Closing Open3D window...")
            self.vis.destroy_window()
            print("Shutting down...")
            exit(0)

rclpy.init()
node = VelodyneVisualizer()

lidar_pos = [0.0, 0.0, 0.0]
lidar_yaw_rad = 0.0

# --- MAIN LOOP ---

try:
    while True:
        # Let ROS process incoming Velodyne messages in this loop
        rclpy.spin_once(node, timeout_sec=0.01)

        # Retrieve message size
        while len(data) < payload_size:
            packet = client_socket.recv(65536)
            if not packet: break
            data += packet
            
        if len(data) < payload_size: break
            
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # Retrieve frame data
        while len(data) < msg_size:
            packet = client_socket.recv(65536)
            if not packet: break
            data += packet
            
        if len(data) < msg_size: break
            
        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Deserialize the dictionary
        frame_dict = pickle.loads(frame_data)
        raw_depth = frame_dict['depth']
        color_image = frame_dict['color']

        # --- ARUCO DETECTION & LIDAR POSITION DETECTION ---

        # Define the dictionary and parameters
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        parameters = cv2.aruco.DetectorParameters()
        
        # Initialize detector
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        
        # Detect markers in the color image
        corners, ids, rejected = detector.detectMarkers(color_image)
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, corners)
            
            # Print the ID and center coordinate of the marker
            c = corners[0][0]
            center_x = int(c[:, 0].mean())
            center_y = int(c[:, 1].mean())
            
            # print(f"Detected ArUco Marker at pixel coordinates: ({center_x}, {center_y})")

            # Convert pixel coordinates to 3D point using depth and camera intrinsics
            depth_value = raw_depth[center_y, center_x]
            if depth_value > 0:
                # Camera intrinsics
                intristics = load_camera_intrinsics()

                # Convert to meters
                z = depth_value / 1000.0
                x = (center_x - intristics['ppx']) * z / intristics['fx']
                y = (center_y - intristics['ppy']) * z / intristics['fy']
                lidar_pos = (x + ARUCO_LIDAR_OFFSET[0], y + ARUCO_LIDAR_OFFSET[1], z + ARUCO_LIDAR_OFFSET[2])

                # print(f"Marker 3D coordinates: ({x:.2f}m, {y:.2f}m, {z:.2f}m)")

            # Assuming the marker is flat on the ground we get the yaw angle
            if len(c) >= 2:
                vec = c[1] - c[0]
                lidar_yaw_rad = np.arctan2(vec[1], vec[0])

        # --- GET POINT CLOUD FROM CAMERA DEPTH ---

        processed_depth = cv2.bilateralFilter(raw_depth.astype(np.float32), 5, 50, 50)
        processed_depth[raw_depth == 0] = 0

        # Get 3D point cloud from depth frame
        intrinsics = load_camera_intrinsics()
        h, w = processed_depth.shape
        i, j = np.meshgrid(np.arange(w), np.arange(h))
        z = processed_depth.astype(np.float32) / 1000.0  # Convert mm to meters
        x = (i - intrinsics['ppx']) * z / intrinsics['fx']
        y = (j - intrinsics['ppy']) * z / intrinsics['fy']
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

        # Filter out points that are too far away or have zero depth
        valid_mask = (z > 0) & (z < 5.0)
        points = points[valid_mask.reshape(-1)]

        # Remove ground plane points
        if lidar_pos[2] > 0:
            ground_threshold = lidar_pos[2] + LIDAR_HEIGHT - ARUCO_LIDAR_OFFSET[2] - 0.1
            
            # Keep only points that are "higher" (closer to the ceiling) than the floor level.
            non_ground_mask = points[:, 2] < ground_threshold
            points = points[non_ground_mask]

        # Downsample camera points for performance
        points = points[::2]

        # --- TRANSFORMATION LOGIC & RENDERING ---

        # Coordinate frame rotation (camera OpenCV → LiDAR ROS)
        R_cam_to_lidar = np.array([[ 1,  0,  0],
                                   [ 0, -1,  0],
                                   [ 0,  0, -1]], dtype=np.float64)

        # Yaw rotation in LiDAR/world frame (from ArUco marker)
        R_yaw = np.array([[ np.cos(lidar_yaw_rad), -np.sin(lidar_yaw_rad), 0],
                          [ np.sin(lidar_yaw_rad),  np.cos(lidar_yaw_rad), 0],
                          [           0,                      0,           1]], dtype=np.float64)

        # Update the points in the camera PCD object
        node.camera_pcd.points = o3d.utility.Vector3dVector(points)

        if CALIBRATION_BASED:
            R_calib = np.load("calib/calib_R.npy")
            t_calib = np.load("calib/calib_t.npy")

            # Apply calibration-based transform to camera points
            node.camera_pcd.rotate(R_calib, center=(0, 0, 0))
            node.camera_pcd.translate(t_calib, relative=True)

        else:

            # Apply rotation first, as per the correct order of transformations
            node.camera_pcd.rotate(R_cam_to_lidar, center=(0, 0, 0))

            # Inverse translation to align camera points with LiDAR position
            lidar_pos_transformed = R_cam_to_lidar @ np.array(lidar_pos)
            node.camera_pcd.translate(-lidar_pos_transformed, relative=True)

            # Apply yaw rotation to align with the orientation of the ArUco marker
            node.camera_pcd.rotate(R_yaw, center=(0, 0, 0))

        # Paint Camera points BLUE
        node.camera_pcd.paint_uniform_color([0.0, 0.0, 1.0])

        voxelized_cam = node.camera_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        voxelized_cam.paint_uniform_color([0.0, 0.0, 1.0])
        node.camera_pcd.points = voxelized_cam.points
        node.camera_pcd.colors = voxelized_cam.colors

        new_mesh = points_to_voxel_mesh(np.asarray(voxelized_cam.points), [0.0, 0.0, 1.0], VOXEL_SIZE)
        node.camera_mesh.vertices  = new_mesh.vertices
        node.camera_mesh.triangles = new_mesh.triangles
        node.camera_mesh.vertex_colors = new_mesh.vertex_colors
        node.camera_mesh.vertex_normals = new_mesh.vertex_normals

        if node.is_first_camera_frame:
            node.vis.add_geometry(node.camera_mesh)
            node.is_first_camera_frame = False
            
        node.vis.update_geometry(node.camera_mesh)

        # --- LIDAR RAY EXTENSION LOGIC ---

        # Get box sidewalls
        sidewall_tris = get_box_sidewalls(raw_depth, color_image)

        triangles_cam = np.array(sidewall_tris, dtype=np.float64)  # (M, 3, 3)

        # Transform triangles from camera space to world space
        M_count = triangles_cam.shape[0]
        tris_flat = triangles_cam.reshape(-1, 3)          # (M*3, 3)

        tris_flat = (R_cam_to_lidar @ tris_flat.T).T

        lidar_pos_transformed = R_cam_to_lidar @ np.array(lidar_pos)
        tris_flat = tris_flat - lidar_pos_transformed

        tris_flat = (R_yaw @ tris_flat.T).T
        triangles_world = tris_flat.reshape(M_count, 3, 3)  # (M, 3, 3)

        # For every LiDAR point, get the vector from the LiDAR to the point
        pts        = node.raw_lidar_points
        if len(pts) == 0: continue
        origins    = np.zeros_like(pts)  # LiDAR is at the origin in its own frame
        vecs       = pts - origins       # (N, 3)
        
        norms      = np.linalg.norm(vecs, axis=1, keepdims=True)
        valid_rays = norms[:, 0] > 0
        directions = np.where(valid_rays[:, None], vecs / np.where(norms > 0, norms, 1), 0)

        hit_points, hit_ray_idx, hit_tri_idx = ray_intersect_triangles_batch(
            origins[valid_rays],
            directions[valid_rays],
            np.array(triangles_world),
            ignore_first=True
        )

        # Add some random noise in the direction of the plane normal to somewhat simulate the LiDAR hitting the side of the box instead of perfectly grazing it
        if len(hit_points) > 0 and len(hit_tri_idx) > 0:
            # 1. Get the pre-calculated normals for the specific triangles hit
            # (Assuming you updated ray_intersect to return tri_indices)
            tri_normals = np.cross(triangles_world[:, 1] - triangles_world[:, 0], 
                                   triangles_world[:, 2] - triangles_world[:, 0])
            # Normalize them
            norms = np.linalg.norm(tri_normals, axis=1, keepdims=True)
            tri_normals = np.divide(tri_normals, norms, out=np.zeros_like(tri_normals), where=norms > 0)

            # 2. Get the specific normal for each hit point
            # No KDTree needed if you return tri_indices from the raycaster
            normals_at_hits = tri_normals[hit_tri_idx]  # (K, 3)

            # 3. Generate RANDOM noise for each point
            # This creates values between -0.01 and +0.01 (2cm total spread)
            noise_magnitude = 0.02 
            random_offsets = (np.random.rand(len(hit_points), 1) - 0.5) * noise_magnitude

            # 4. Apply the noise along the normal vector
            hit_points += normals_at_hits * random_offsets

            # print(f"Extended {len(hit_points)} LiDAR points to the detected box sidewalls")

            node.extended_pcd.points = o3d.utility.Vector3dVector(hit_points)

            # Paint extended points green
            node.extended_pcd.paint_uniform_color([0.0, 1.0, 0.0])
            voxelized_ext = node.extended_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
            voxelized_ext.paint_uniform_color([0.0, 1.0, 0.0])
            node.extended_pcd.points = voxelized_ext.points
            node.extended_pcd.colors = voxelized_ext.colors

            new_mesh = points_to_voxel_mesh(np.asarray(voxelized_ext.points), [0.0, 1.0, 0.0], VOXEL_SIZE)
            node.extended_mesh.vertices  = new_mesh.vertices
            node.extended_mesh.triangles = new_mesh.triangles
            node.extended_mesh.vertex_colors = new_mesh.vertex_colors
            node.extended_mesh.vertex_normals = new_mesh.vertex_normals

            if node.is_first_extended_frame:
                node.vis.add_geometry(node.extended_mesh)
                node.is_first_extended_frame = False

            node.vis.update_geometry(node.extended_mesh)

        else:
            # Clear the mesh if there are no hits
            node.extended_pcd.points = o3d.utility.Vector3dVector()
            node.extended_pcd.colors = o3d.utility.Vector3dVector()
            
            node.extended_mesh.vertices = o3d.utility.Vector3dVector()
            node.extended_mesh.triangles = o3d.utility.Vector3iVector()
            node.extended_mesh.vertex_colors = o3d.utility.Vector3dVector()
            node.extended_mesh.vertex_normals = o3d.utility.Vector3dVector()
            
            if not node.is_first_extended_frame:
                node.vis.update_geometry(node.extended_mesh)

        node.vis.poll_events()
        node.vis.update_renderer()

        # Show camera feed with detected markers
        cv2.putText(color_image, f"Lidar Pos: ({lidar_pos[0]:.2f}, {lidar_pos[1]:.2f}, {lidar_pos[2]:.2f})m", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Raw RGB Feed", color_image)

        # Recording & Evaluation
        os.makedirs(VOXEL_RECORD_DIR, exist_ok=True)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == ord('o'):
            green_pts = np.asarray(node.extended_pcd.points)
            if len(green_pts) > 0:
                # Find next available index (skip indices that already have an occluded file)
                idx = 0
                while os.path.exists(os.path.join(VOXEL_RECORD_DIR, f"occluded_{idx:02d}.npy")):
                    idx += 1
                np.save(os.path.join(VOXEL_RECORD_DIR, f"occluded_{idx:02d}.npy"), green_pts)
                print(f"[RECORD_OCCLUDED] Saved occluded_{idx:02d}.npy — "
                        f"{len(green_pts)} green voxels. Now change scene and run RECORD_GT.")
            else:
                print("[RECORD_OCCLUDED] Skipped — no green points detected")

        if key == ord('g'):
            red_pts = np.asarray(node.pcd.points)
            if len(red_pts) > 0:
                # Find the latest occluded file that has no GT pair yet
                idx = 0
                while os.path.exists(os.path.join(VOXEL_RECORD_DIR, f"gt_{idx:02d}.npy")):
                    idx += 1
                if not os.path.exists(os.path.join(VOXEL_RECORD_DIR, f"occluded_{idx:02d}.npy")):
                    print(f"[RECORD_GT] No matching occluded_{idx:02d}.npy found. Record occluded scene first.")
                else:
                    np.save(os.path.join(VOXEL_RECORD_DIR, f"gt_{idx:02d}.npy"), red_pts)
                    print(f"[RECORD_GT] Saved gt_{idx:02d}.npy — {len(red_pts)} red voxels. Pair {idx:02d} complete.")
            else:
                print("[RECORD_GT] Skipped — no red points detected")

        if key == ord('e'):
            pairs = sorted(f for f in os.listdir(VOXEL_RECORD_DIR) if f.startswith("occluded_"))
            if not pairs:
                print("[EVALUATE] No recordings found.")
            else:
                def to_voxel_keys(pts):
                    return set(map(tuple, np.floor(pts / VOXEL_SIZE).astype(int)))

                total_precision = 0.0
                valid_pairs = 0
                for fname in pairs:
                    idx = fname.split("_")[1].split(".")[0]
                    gt_path = os.path.join(VOXEL_RECORD_DIR, f"gt_{idx}.npy")
                    occ_path = os.path.join(VOXEL_RECORD_DIR, fname)

                    if not os.path.exists(gt_path):
                        print(f"  [pair {idx}] Missing GT file, skipping.")
                        continue

                    green_keys = to_voxel_keys(np.load(occ_path))
                    gt_keys    = to_voxel_keys(np.load(gt_path))

                    hits = green_keys & gt_keys
                    tp = len(hits)
                    precision = tp / len(green_keys) if green_keys else 0.0
                    total_precision += precision

                    # Identify predicted voxels that did not hit GT (False Positives)
                    extra_keys = green_keys - gt_keys

                    if extra_keys and gt_keys:
                        # Convert keys back to world coordinates (centered in the voxel)
                        gt_pts = np.array(list(gt_keys)) * VOXEL_SIZE + VOXEL_SIZE / 2
                        extra_pts = np.array(list(extra_keys)) * VOXEL_SIZE + VOXEL_SIZE / 2

                        # Use KDTree for fast "closest point" lookup
                        tree = cKDTree(gt_pts)
                        distances, _ = tree.query(extra_pts, k=1) 
                        avg_dist_error = np.mean(distances)
                        max_dist_error = np.max(distances)
                    else:
                        avg_dist_error = 0.0

                    valid_pairs += 1
                    print(f"  [Scene {idx}] Predicted: {len(green_keys)} | GT hits: {tp} | Precision: {precision:.2%} | Avg Dist Error: {avg_dist_error * 100:.1f}cm | Max Dist Error: {max_dist_error * 100:.1f}cm")

                print(f"[EVALUATE] Avg precision across {valid_pairs} scenes: {total_precision / valid_pairs:.2%}")

except Exception as e:
    print(f"Network stream ended or error: {e}")
    raise e

finally:
    client_socket.close()
    cv2.destroyAllWindows()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()