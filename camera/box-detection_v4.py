import numpy as np
import cv2
import socket
import struct
import pickle
from collections import deque
import open3d as o3d
import sys
import os

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

# Detection parameters
MIN_AREA = 2000        # Minimum contour area (in pixels) to ignore noise
MIN_EXTENT = 0.8       # How rectangular the object must be (1.0 = perfect rectangle)
WINDOW_SIZE = 5        # Number of frames to average

GROUND_THRESHOLD = 2.6  # Meters. Points above this are considered non-ground.

# Initialize the frame buffer
frame_buffer = deque(maxlen=WINDOW_SIZE)

def get_3d_box_top(camera_pcd, intrinsics, h, w, floor_z):
    """Detect boxes in the point cloud and return the 3D coordinates of their top faces as well as the sidewall triangles."""

    # Downsample for speed
    pcd_small = camera_pcd.voxel_down_sample(voxel_size=0.01)
    
    # Cluster in 3D to find individual boxes
    labels = np.array(pcd_small.cluster_dbscan(eps=0.15, min_points=200))
    if len(labels) == 0: return [], []
    
    box_tops = []
    sidewall_tris = []
    for i in range(labels.max() + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_pts = np.asarray(pcd_small.points)[cluster_indices]

        if len(cluster_pts) < 200: continue  # Skip small clusters
        
        # Find the heighest point in the cluster (the box top)
        top_z = np.min(cluster_pts[:, 2])

        # Only keep points that are within 10cm of the top
        # This discards the potential side points that cause stretching
        top_plane_mask = cluster_pts[:, 2] < (top_z + 0.1)
        top_plane_pts = cluster_pts[top_plane_mask]
        
        # Project cluster to 2D pixels to find the bounding box
        u = (top_plane_pts[:, 0] * intrinsics['fx'] / top_plane_pts[:, 2]) + intrinsics['ppx']
        v = (top_plane_pts[:, 1] * intrinsics['fy'] / top_plane_pts[:, 2]) + intrinsics['ppy']
        pixel_pts = np.stack((u, v), axis=1).astype(np.float32)
        
        # Fit a 2D rectangle to these pixels
        rect = cv2.minAreaRect(pixel_pts)
        box_2d_pixels = cv2.boxPoints(rect) # The 4 corners in pixels

        # If the box is too close to the edge of the image, ignore it (likely noise)
        if np.any(box_2d_pixels < 10) or np.any(box_2d_pixels[:, 0] > w - 10) or np.any(box_2d_pixels[:, 1] > h - 10):
            continue
        
        # Back-project those 4 pixel corners back to 3D at the 'top_z' height
        # This gives us a mathematically perfect 3D rectangle at the box's top
        box_3d_top = []
        for px, py in box_2d_pixels:
            world_x = (px - intrinsics['ppx']) * top_z / intrinsics['fx']
            world_y = (py - intrinsics['ppy']) * top_z / intrinsics['fy']
            box_3d_top.append([world_x, world_y, top_z])

        # Calculate volume of box and drop if it's too small
        box_width = np.linalg.norm(np.array(box_3d_top[1]) - np.array(box_3d_top[0]))
        box_length = np.linalg.norm(np.array(box_3d_top[2]) - np.array(box_3d_top[1]))
        box_height = np.abs(top_z - floor_z)
        box_volume = box_width * box_length * box_height

        if box_volume < 0.01:  # Less than 10 liters, likely noise
            continue
            
        box_tops.append(np.array(box_3d_top))

        box_center_3d = np.array([
            (box_3d_top[0][0] + box_3d_top[2][0]) / 2,
            (box_3d_top[0][1] + box_3d_top[2][1]) / 2,
            (top_z + floor_z) / 2.0
        ])

        print(f"Detected box with volume {box_volume:.4f} m^3 (Width: {box_width:.2f}m, Length: {box_length:.2f}m, Height: {box_height:.2f}m)")

        def enforce_outward(tri):
            """Ensures the triangle normal points away from the center of the box."""
            v0, v1, v2 = np.array(tri[0]), np.array(tri[1]), np.array(tri[2])
            normal = np.cross(v1 - v0, v2 - v0)
            tri_center = (v0 + v1 + v2) / 3.0
            if np.dot(normal, tri_center - box_center_3d) < 0:
                return [tri[0], tri[2], tri[1]] 
            return tri
        
        for j in range(4):
            # 3D corners of the TOP face
            p1_top = box_3d_top[j]
            p2_top = box_3d_top[(j + 1) % 4]
            
            # 3D corners of the BOTTOM face (same X,Y but floor Z)
            p1_bot = [p1_top[0], p1_top[1], floor_z]
            p2_bot = [p2_top[0], p2_top[1], floor_z]
            
            # Create the two triangles for this vertical wall
            t1 = [p1_top, p2_top, p2_bot]
            t2 = [p1_top, p1_bot, p2_bot]
            
            sidewall_tris.append(enforce_outward(t1))
            sidewall_tris.append(enforce_outward(t2))
        
    return box_tops, sidewall_tris
        
def project_3d_to_2d(pts_3d, intrinsics):
    """Project Nx3 points (meters) to Nx2 pixel coordinates."""
    x = pts_3d[:, 0]
    y = pts_3d[:, 1]
    z = pts_3d[:, 2]
    
    # Avoid division by zero
    z_eff = np.where(z == 0, 0.001, z)
    
    u = (x * intrinsics['fx'] / z_eff) + intrinsics['ppx']
    v = (y * intrinsics['fy'] / z_eff) + intrinsics['ppy']
    
    return np.stack((u, v), axis=-1).astype(np.int32)

try:
    while True:
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
        current_raw_depth = frame_dict['depth']
        color_image = frame_dict['color']

        # --- SLIDING WINDOW AVERAGING ---
        
        # Add current frame to buffer
        frame_buffer.append(current_raw_depth.astype(np.float32))

        # Compute average depth over the buffer
        stack = np.array(frame_buffer)

        # Create a mask of valid pixels (where depth > 0)
        valid_mask = (stack > 0)

        # Sum only the valid pixels
        sum_valid = np.sum(stack, axis=0)
        
        # Count how many frames had valid data for each pixel
        count_valid = np.sum(valid_mask, axis=0)
        
        # Avoid division by zero: if a pixel was 0 in ALL frames, keep it 0.
        # Otherwise, divide the sum by the count of valid frames.
        raw_depth = np.divide(sum_valid, count_valid, 
                              out=np.zeros_like(sum_valid), 
                              where=count_valid > 0).astype(np.uint16)

        processed_depth = cv2.bilateralFilter(raw_depth.astype(np.float32), 5, 50, 50)
        processed_depth[raw_depth == 0] = 0

        # --- IMAGE PRE-PROCESSING ---

        hole_mask = (raw_depth == 0).astype(np.uint8)

        # Convert to 8-bit for processing
        depth_8bit = cv2.convertScaleAbs(raw_depth, alpha=0.07)

        # Fill holes using inpainting
        depth_8bit = cv2.inpaint(depth_8bit, hole_mask, 5, cv2.INPAINT_TELEA)

        # Generate Colormap for visualization
        depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

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
        non_ground_mask = points[:, 2] < GROUND_THRESHOLD
        points = points[non_ground_mask]

        # Downsample camera points for performance
        points = points[::2]

        # Get 3d pcd from depth
        camera_pcd = o3d.geometry.PointCloud()
        camera_pcd.points = o3d.utility.Vector3dVector(points)

        h, w = processed_depth.shape

        # Get box corners
        box_corners, sidewall_tris = get_3d_box_top(camera_pcd, intrinsics, h, w, GROUND_THRESHOLD)

        # Visualize
        vis_image = depth_colormap.copy()

        # Visualize sidewall triangles
        for tri in sidewall_tris:
            pixel_coords = project_3d_to_2d(np.array(tri), intrinsics)
            cv2.polylines(vis_image, [pixel_coords], isClosed=True, color=(255, 0, 0), thickness=2)

        for box in box_corners:
            pixel_coords = project_3d_to_2d(box, intrinsics)
            cv2.polylines(vis_image, [pixel_coords], isClosed=True, color=(0, 0, 255), thickness=4)


        cv2.imshow("Box Detection", vis_image)
        cv2.imshow("Color Image", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

except Exception as e:
    print(f"Network stream ended or error: {e}")
    raise e

finally:
    client_socket.close()
    cv2.destroyAllWindows()