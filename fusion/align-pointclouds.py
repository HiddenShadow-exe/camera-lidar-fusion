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
ARUCO_LIDAR_OFFSET = (0.0, 0.0, 0.1)

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
        
        # If markers are found, draw them on the color_image
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
            
            # Keep only points that are "higher" (closer to the ceiling) 
            # than the floor level.
            # In raw camera coords, smaller Z is closer to the ceiling.
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
        node.vis.poll_events()
        node.vis.update_renderer()

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