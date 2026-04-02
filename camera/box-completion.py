import numpy as np
import cv2
import socket
import struct
import pickle
from collections import deque
import pyrealsense2 as rs
import open3d as o3d

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
MIN_AREA = 1000        # Minimum contour area (in pixels) to ignore noise
MIN_EXTENT = 0.8       # How rectangular the object must be (1.0 = perfect rectangle)
WINDOW_SIZE = 5        # Number of frames to average

# Initialize the frame buffer
frame_buffer = deque(maxlen=WINDOW_SIZE)

class CameraIntrinsics:
    def __init__(self, w=640, h=480):
        self.fx = 460.0  # Focal length x
        self.fy = 460.0  # Focal length y
        self.ppx = 320.0 # Principal point x
        self.ppy = 240.0 # Principal point y
        self.coeffs = [0, 0, 0, 0, 0] # Distortion

# Params:
# depth_frame: the raw depth frame from the camera
# boxes: a list of 4 corners of the detected boxes in pixel coordinates
def save_pointcloud(raw_depth, boxes):
    intr = CameraIntrinsics()
    h, w = raw_depth.shape

    # --- BUILD INITAL POINT CLOUD ---

    # Create a grid of pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Filter out zero-depth pixels
    z = raw_depth.astype(np.float32) / 1000.0  # Convert mm to meters
    mask = z > 0
    
    z_valid = z[mask]
    u_valid = u[mask]
    v_valid = v[mask]

    # Math for Deprojection (Pixel to 3D)
    x_valid = (u_valid - intr.ppx) * z_valid / intr.fx
    y_valid = (v_valid - intr.ppy) * z_valid / intr.fy
    
    # Scene points (N, 3)
    scene_points = np.stack((x_valid, y_valid, z_valid), axis=-1)

    # Save raw scene
    filename_raw = "before.ply"
    pcd_before = o3d.geometry.PointCloud()
    pcd_before.points = o3d.utility.Vector3dVector(scene_points)
    o3d.io.write_point_cloud(filename_raw, pcd_before)
    print(f"Saved raw point cloud to {filename_raw}")

    # --- EXTRUSION LOGIC ---

    extruded_points_list = []
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
            xx = (px_c - intr.ppx) * zx / intr.fx
            yx = (py_c - intr.ppy) * zx / intr.fy
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

        # Extrude each edge of the box top down to floor_depth
        extruded_points = []
        num_steps = 150
        for i in range(len(top_corners_3d)):
            p1 = top_corners_3d[i]
            p2 = top_corners_3d[(i + 1) % len(top_corners_3d)]
            for t in np.linspace(0, 1, num_steps):
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                for z in np.linspace(p1[2], floor_depth, num_steps):
                    extruded_points_list.append([x, y, z])

    if extruded_points_list:
        extruded_np = np.array(extruded_points_list)
        completed_verts = np.vstack([scene_points, extruded_np])
    else:
        completed_verts = scene_points

    # Save completed scene
    pcd_after = o3d.geometry.PointCloud()
    pcd_after.points = o3d.utility.Vector3dVector(completed_verts)
    o3d.io.write_point_cloud("after.ply", pcd_after)
    print("Saved completed point cloud to after.ply")

    print(f"Number of boxes processed: {len(boxes)}")

    
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

        # --- IMAGE PRE-PROCESSING ---

        hole_mask = (raw_depth == 0).astype(np.uint8)

        # Convert to 8-bit for processing
        depth_8bit = cv2.convertScaleAbs(raw_depth, alpha=0.07)

        # Fill holes using inpainting
        depth_8bit = cv2.inpaint(depth_8bit, hole_mask, 5, cv2.INPAINT_TELEA)

        # Generate Colormap for visualization
        depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

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
            extent = area / box_area

            # Extent is the ratio of contour area to bounding rectangle area.
            extent = area / box_area

            # If it has 4-6 vertices (approx) AND it's mostly rectangular (extent)
            if extent > MIN_EXTENT and len(approx) <= 6:
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                boxes.append(box)
                
                # Draw the valid box on the colormap
                cv2.drawContours(depth_colormap, [box], 0, (0, 255, 0), 2)
                for p in box:
                    cv2.circle(depth_colormap, tuple(p), 5, (0, 0, 255), -1)


        # Show both feeds
        cv2.putText(depth_colormap, f"Number of boxes: {len(boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Box Detection (Depth)", depth_colormap)
        cv2.imshow("Raw RGB Feed", color_image)
        cv2.imshow("Edge Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            save_pointcloud(raw_depth, boxes)
        if key == ord('q'):
            break

except Exception as e:
    print(f"Network stream ended or error: {e}")

finally:
    client_socket.close()
    cv2.destroyAllWindows()