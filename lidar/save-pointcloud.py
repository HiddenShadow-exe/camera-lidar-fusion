import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2 # type: ignore
import sensor_msgs_py.point_cloud2 as pc2 # type: ignore
import numpy as np
import open3d as o3d
import os
import threading
import sys
import termios
import tty
import select

class VelodyneSaver(Node):
    def __init__(self, save_dir="lidar_frames"):
        super().__init__('velodyne_saver')

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.frame_count = 0
        self.last_points = None
        self.running = True

        # Subscribe to Velodyne PointCloud2 topic
        self.subscription = self.create_subscription(
            PointCloud2,
            '/velodyne_points',
            self.pointcloud_callback,
            10
        )

        # Start thread for keyboard input
        self.input_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.input_thread.start()

    def pointcloud_callback(self, msg):
        points_gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = [[p[0], p[1], p[2]] for p in points_gen]
        
        if not points:
            return

        np_points = np.array(points, dtype=np.float32)

        # Drop points that are too far away (> 5m)
        distances = np.linalg.norm(np_points, axis=1)
        self.last_points = np_points[distances < 5.0]

    def keyboard_listener(self):
        print("\n------------- Controls -------------")
        print("Press 'S' to save current LiDAR frame")
        print("Press 'Q' to quit")
        print("------------------------------------\n")

        while self.running:
            key = self.get_key_nonblocking()
            if key is not None:
                cmd = key.lower()
                if cmd == 's':
                    self.save_frame()
                elif cmd == 'q':
                    print("Exiting...")
                    self.running = False
                    # Signal ROS to shut down
                    rclpy.shutdown()
                    break

    def save_frame(self):
        if self.last_points is None:
            print("No LiDAR data received yet!")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.last_points)
        ply_filename = os.path.join(self.save_dir, f"frame_{self.frame_count:05d}.ply")
        o3d.io.write_point_cloud(ply_filename, pcd)
        print(f"Saved {ply_filename} with {len(self.last_points)} points")
        self.frame_count += 1

    @staticmethod
    def get_key_nonblocking():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            # select.select waits for input. The 0.1 is a short timeout.
            if select.select([sys.stdin], [], [], 0.1)[0]:
                return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return None

def main(args=None):
    rclpy.init(args=args)
    node = VelodyneSaver()
    
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.running = False
        node.destroy_node()
        # Check if rclpy is still active before trying to shut down
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()