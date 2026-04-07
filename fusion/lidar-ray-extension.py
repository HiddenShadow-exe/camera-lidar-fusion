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

LIDAR_HEIGHT = 0.7
ARUCO_LIDAR_OFFSET = [0, 0, -0.05]

CALIBRATION_BASED = False

# Detection parameters
MIN_AREA = 1000        # Minimum contour area (in pixels) to ignore noise
MIN_EXTENT = 0.8       # How rectangular the object must be (1.0 = perfect rectangle)

def get_box_sidewalls(raw_depth, color_image):
    # --- IMAGE PRE-PROCESSING ---
    hole_mask = (raw_depth == 0).astype(np.uint8)

    # Convert to 8-bit for processing
    depth_8bit = cv2.convertScaleAbs(raw_depth, alpha=0.07)

    # Fill holes using inpainting
    depth_8bit = cv2.inpaint(depth_8bit, hole_mask, 5, cv2.INPAINT_TELEA)

    # --- IMAGE PROCESSING ---

    # 1. Apply a median blur to remove raw depth sensor noise/speckles
    blurred_depth = cv2.medianBlur(depth_8bit, 7)

    # 2. Find depth discontinuities (edges)
    # https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    edges = cv2.Canny(blurred_depth, 20, 35)

    # 3. Clean edges and denoise (erosion, then dilation)
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)


    # 4. Detect shapes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = raw_depth.shape
    boxes = []

    for cnt in contours:
        # Bridge gaps using a Convex Hull, this "wraps" the points even if the line is broken
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)
        
        # Ignore small noise
        if area < MIN_AREA:
            continue

        # Simplify the shape (Polygonal Approximation)
        # This helps ignore small wiggles or "missing" chunks of the edge
        epsilon = 0.04 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        # Rectangle Check
        # We look for shapes with roughly 4 corners OR a high "Extent"
        rect = cv2.minAreaRect(hull)
        box_width, box_height = rect[1]
        box_area = box_width * box_height

        if box_area == 0: continue

        # Extent is the ratio of contour area to bounding rectangle area.
        extent = area / box_area

        # If it has 4-6 vertices (approx) AND it's mostly rectangular (extent)
        if extent > MIN_EXTENT and len(approx) <= 6:
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # If one of the corners are near the edge of the frame, skip this box (probably a false detection or incomplete box)
            if np.any(box[:, 0] < 10) or np.any(box[:, 0] > w - 10) or np.any(box[:, 1] < 10) or np.any(box[:, 1] > h - 10):
                continue

            boxes.append(box)
            
            # Draw the valid box on the colormap
            cv2.drawContours(color_image, [box], 0, (0, 255, 0), 2)
            for p in box:
                cv2.circle(color_image, tuple(p), 5, (0, 0, 255), -1)


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
    if N == 0 or M == 0: return np.zeros((0, 3))

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
    if len(ray_idx) == 0: return np.zeros((0, 3))

    # Compute each hit point: origin + t * direction
    t_vals   = T[ray_idx, tri_idx]                      # (K,)

    # If we want all hits, just get all points and return
    if not ignore_first:
        hit_pts  = origins[ray_idx] + t_vals[:, None] * directions[ray_idx]  # (K, 3)
        return hit_pts
    
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
        return np.zeros((0, 3))

    # Calculate hit points for 2nd, 3rd... intersections
    hit_pts = origins[final_ray_idx] + final_t_vals[:, None] * directions[final_ray_idx]

    return hit_pts


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

        # Open3D visualizer setup
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Velodyne Live PointCloud")
        
        self.pcd = o3d.geometry.PointCloud()
        self.is_first_frame = True

        self.camera_pcd = o3d.geometry.PointCloud()
        self.is_first_camera_frame = True

        self.extended_pcd = o3d.geometry.PointCloud()
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

        # Update Open3D point cloud
        self.pcd.points = o3d.utility.Vector3dVector(np_points)

        # Color LiDAR points red
        self.pcd.paint_uniform_color([1.0, 0.0, 0.0])

        if self.is_first_frame:
            self.vis.add_geometry(self.pcd)
            self.is_first_frame = False

        self.vis.update_geometry(self.pcd)
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
            
            print(f"Detected ArUco Marker at pixel coordinates: ({center_x}, {center_y})")

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

                print(f"Marker 3D coordinates: ({x:.2f}m, {y:.2f}m, {z:.2f}m)")

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
        
        if node.is_first_camera_frame:
            node.vis.add_geometry(node.camera_pcd)
            node.is_first_camera_frame = False
            
        node.vis.update_geometry(node.camera_pcd)

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
        pts        = np.asarray(node.pcd.points)
        origins    = np.zeros_like(pts)  # LiDAR is at the origin in its own frame
        vecs       = pts - origins       # (N, 3)
        
        norms      = np.linalg.norm(vecs, axis=1, keepdims=True)
        valid_rays = norms[:, 0] > 0
        directions = np.where(valid_rays[:, None], vecs / np.where(norms > 0, norms, 1), 0)

        hit_points = ray_intersect_triangles_batch(
            origins[valid_rays],
            directions[valid_rays],
            np.array(triangles_world),
            ignore_first=True
        )

        print(f"Extended {len(hit_points)} LiDAR points to the detected box sidewalls")

        node.extended_pcd.points = o3d.utility.Vector3dVector(hit_points)

        # Paint extended points green
        node.extended_pcd.paint_uniform_color([0.0, 1.0, 0.0])

        if node.is_first_extended_frame:
            node.vis.add_geometry(node.extended_pcd)
            node.is_first_extended_frame = False

        node.vis.update_geometry(node.extended_pcd)

        node.vis.poll_events()
        node.vis.update_renderer()

        # Save pcd on every frame
        combined_pcd = node.pcd + node.camera_pcd + node.extended_pcd
        o3d.io.write_point_cloud("combined.ply", combined_pcd)

        # Show camera feed with detected markers
        cv2.putText(color_image, f"Lidar Pos: ({lidar_pos[0]:.2f}, {lidar_pos[1]:.2f}, {lidar_pos[2]:.2f})m", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Raw RGB Feed", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

except Exception as e:
    print(f"Network stream ended or error: {e}")

finally:
    client_socket.close()
    cv2.destroyAllWindows()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()