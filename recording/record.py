import os
import sys

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.get_camera_data import get_camera_data, get_pcd_from_camera_depth, get_pcd_from_camera_depth
from utils.get_aruco import get_aruco
from utils.transformation import get_lidar_position, screen_to_world, transform_camera_to_lidar
from utils.detect_object import get_bounding_box
from utils.crop_and_obb import crop_points_to_bounding_box, get_vertical_obb
from utils.save_data import save_data
from utils.luminance import calculate_luminance

# Recording parameters
TARGET_OBJECT = "clothes iron"
LABEL = "clothes iron"

mouse_click_position = None 
object_orientation_vector = None
saved_flag = False
rotation_number = 1

class RoboSenseVisualizer(Node):
    def __init__(self):
        super().__init__('robosense_visualizer')

        # Subscribe to RoboSense PointCloud2 topic
        self.subscription = self.create_subscription(
            PointCloud2,
            '/rslidar_points',
            self.pointcloud_callback,
            10
        )

        # Open3D visualizer setup
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="RoboSense Airy Live PointCloud")
        
        self.intensities = np.empty(0, dtype=np.float32)

        self.pcd = o3d.geometry.PointCloud()
        self.is_first_frame = True

        self.camera_pcd = o3d.geometry.PointCloud()
        self.is_first_camera_frame = True

        self.object_pcd = o3d.geometry.PointCloud()
        self.object_obb = o3d.geometry.OrientedBoundingBox()
        self.is_first_object_frame = True

        # Cache the colormap lookup function to avoid recreating it
        self.cmap = plt.get_cmap("plasma")

    def pointcloud_callback(self, msg):
        # Read points as a generator
        fields = ("x", "y", "z", "intensity")
        points_gen = pc2.read_points(msg, field_names=fields, skip_nans=True)
        
        # Convert generator to structured numpy array instantly
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')]
        res_array = np.fromiter(points_gen, dtype=dtype)
        
        if res_array.size == 0:
            return

        # Extract points and intensities into separate numpy arrays
        np_points = np.column_stack((res_array['x'], res_array['y'], res_array['z']))
        self.intensities = res_array['intensity']

        # Drop points that are too far away (> 5m)
        distances = np.linalg.norm(np_points, axis=1)
        mask = distances < 5.0
        np_points = np_points[mask]
        self.intensities = self.intensities[mask]

        # Normalize intensities to [0, 1] range for color mapping
        max_int = self.intensities.max()
        min_int = self.intensities.min()
        if max_int > min_int:
            self.intensities = (self.intensities - min_int) / (max_int - min_int)
        else:
            self.intensities = np.zeros_like(self.intensities)

        colors = self.cmap(self.intensities)[:, :3] # Slice out the alpha channel

        # Since the LiDAR is mounted sideways, rotate the LiDAR points right-side up
        # 90 degree clockwise rotation around Y-axis (x, y, z) -> (z, y, -x)
        R_lidar = np.array([[ 0,  0,  1],
                            [ 0,  1,  0],
                            [-1,  0,  0]], dtype=np.float64)

        np_points = np_points @ R_lidar.T

        # Update Open3D point cloud
        self.pcd.points = o3d.utility.Vector3dVector(np_points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

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


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global mouse_click_position
        mouse_click_position = (x, y)

def main(args=None):
    rclpy.init(args=args)
    node = RoboSenseVisualizer()
    
    prev_marker_corners = None
    marker_seen = False

    try:
        while True:
            # Spin ROS node to process incoming point cloud messages
            rclpy.spin_once(node, timeout_sec=0.1)

            # Get frame from camera
            color_image, raw_depth = get_camera_data()
            if color_image is None: continue

            # Get ArUco marker position
            marker_corners, marker_id = get_aruco(color_image, draw_on_image=False)
            if marker_corners is None or marker_id is None:
                marker_corners = prev_marker_corners
                marker_seen = False
            else:
                prev_marker_corners = marker_corners
                marker_seen = True

            # Get LiDAR position and yaw from ArUco marker
            offset = (0.09, 0.0, 0.17)
            ceiling_height = 2.65
            lidar_height = 0.4
            lidar_pos, lidar_yaw = get_lidar_position(
                aruco_corners      = marker_corners,
                z                  = ceiling_height - lidar_height - offset[2],
                aruco_lidar_offset = offset
            )

            if lidar_pos is None or lidar_yaw is None:
                # Dummy values to prevent crashes when marker isn't detected
                lidar_pos = (0.0, 0.0, ceiling_height - lidar_height)
                lidar_yaw = 0.0

            # Get camera depth pcd and remove ground points based on LiDAR height
            ground_threshold = lidar_pos[2] + lidar_height - 0.08
            camera_points = get_pcd_from_camera_depth(raw_depth, ground_threshold)
            # camera_points = camera_points[::10]  # Downsample for visualization

            # Transform camera points to LiDAR/world frame
            final_camera_points = transform_camera_to_lidar(camera_points, lidar_pos, lidar_yaw)

            # Update the points in the camera PCD object
            node.camera_pcd.points = o3d.utility.Vector3dVector(final_camera_points)
            node.camera_pcd.paint_uniform_color([0.0, 0.0, 1.0]) # Blue
            
            # Add camera PCD to visualizer if it's the first frame
            if node.is_first_camera_frame:
                node.vis.add_geometry(node.camera_pcd)
                node.is_first_camera_frame = False
                
            node.vis.update_geometry(node.camera_pcd)

            # Object detection
            bbox = get_bounding_box(color_image, TARGET_OBJECT)

            lidar_points = np.asarray(node.pcd.points)
            object_points_lidar, object_intensities, object_points_cam = crop_points_to_bounding_box(
                lidar_points, node.intensities, final_camera_points, bbox, lidar_pos, lidar_yaw, floor_z_threshold=ceiling_height - 0.04
            )

            # Calculate the 3D Oriented Bounding Box
            global object_orientation_vector
            obb = get_vertical_obb(object_points_lidar, object_orientation_vector)

            # Render points and OBB
            if object_points_lidar is not None:
                if node.is_first_object_frame:
                    node.vis.add_geometry(node.object_pcd)
                    node.vis.add_geometry(node.object_obb)
                    node.object_obb.color = (1, 0, 0) # Red
                    node.is_first_object_frame = False

                node.object_pcd.points = o3d.utility.Vector3dVector(object_points_lidar)
                node.object_pcd.paint_uniform_color([1.0, 0.0, 0.0]) # Red
                node.vis.update_geometry(node.object_pcd)

                if obb is not None:
                    node.object_obb.center = obb.center
                    node.object_obb.R = obb.R
                    node.object_obb.extent = obb.extent
                    node.vis.update_geometry(node.object_obb)
                else:
                    cv2.putText(color_image, "OBB cannot be calculated", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                cv2.putText(color_image, f"'{TARGET_OBJECT}' not detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Update the visualizer
            node.vis.poll_events()
            node.vis.update_renderer()

            key = cv2.waitKey(1) & 0xFF

            if key == ord('v'):
                # Prompt to select a direction vector on the object for OBB orientation
                global mouse_click_position
                mouse_click_position = None
                
                cv2.putText(color_image, "Please click a point on the object to define OBB orientation...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                while mouse_click_position is None:
                    cv2.imshow("Raw RGB Feed", color_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Shutting down...")
                        exit(0)

                x_0, y_0 = mouse_click_position

                mouse_click_position = None

                # Prompt for a second click to define the orientation vector
                cv2.putText(color_image, "Please click a second point to define the OBB orientation vector...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                while mouse_click_position is None:
                    cv2.imshow("Raw RGB Feed", color_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Shutting down...")
                        exit(0)

                x_1, y_1 = mouse_click_position

                # Orientation vector is in screen space, we need to convert it to a 3D vector in world
                # space. We assume the vector lies in the XY plane of the world frame and ignore Z
                object_orientation_vector = screen_to_world(x_1, y_1, 1.0, lidar_yaw) - screen_to_world(x_0, y_0, 1.0, lidar_yaw)


            global saved_flag
            global rotation_number

            if key == ord('s'):
                print("Saving current frame data...")

                if object_orientation_vector is None:
                    print("No orientation vector defined. Please press 'v' and select an orientation vector before saving.")
                    continue

                if object_points_lidar is None or obb is None:
                    print("No valid object detected. Data cannot be saved.")
                    continue

                luminance_total, luminance_object = calculate_luminance(color_image, bbox)
                
                save_data(
                    label=LABEL,
                    recording_number=rotation_number,
                    bbox=bbox,
                    lidar_points=lidar_points,
                    lidar_intensities=node.intensities,
                    object_lidar_points=object_points_lidar,
                    object_lidar_intensities=object_intensities,
                    obb_3d=obb,
                    object_orientation_vector=object_orientation_vector,
                    camera_points=final_camera_points,
                    object_camera_points=object_points_cam,
                    lidar_to_cam_extrinsic=(np.asarray(lidar_pos), lidar_yaw),
                    color_image=color_image,
                    luminance_total=luminance_total,
                    luminance_object=luminance_object
                )
                print("Data saved successfully.")
                saved_flag = True

            if key == ord('n'):
                rotation_number += 1
                print(f"Moving to next recording number: {rotation_number}")

                object_orientation_vector = None
                saved_flag = False
            

            # Draw the bounding box on the color image for visualization
            if bbox is not None:
                cv2.rectangle(color_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(color_image, TARGET_OBJECT, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show camera feed with detected marker position overlay
            if marker_seen:
                cv2.putText(color_image, f"Lidar Pos: ({lidar_pos[0]:.2f}, {lidar_pos[1]:.2f}, {lidar_pos[2]:.2f})m", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(color_image, "Marker not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if object_orientation_vector is not None:
                cv2.arrowedLine(
                    img=color_image,
                    pt1=(x_0, y_0),
                    pt2=(x_1, y_1),
                    color=(255, 255, 0),
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )
                cv2.putText(color_image, "OBB Orientation Vector", (x_0 + 10, y_0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if saved_flag:
                cv2.putText(color_image, "Data Saved!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Put the recording number on the image (bottom left corner)
            cv2.putText(color_image, f"Recording Number: {rotation_number}", (10, color_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Raw RGB Feed", color_image)
            cv2.setMouseCallback("Raw RGB Feed", click_event)

            if key == ord('q'): break

    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()