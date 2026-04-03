import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2 # type: ignore
import sensor_msgs_py.point_cloud2 as pc2 # type: ignore
import numpy as np
import open3d as o3d

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

def main(args=None):
    rclpy.init(args=args)
    node = VelodyneVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()