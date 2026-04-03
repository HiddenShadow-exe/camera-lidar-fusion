import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2 # type: ignore
import sensor_msgs_py.point_cloud2 as pc2 # type: ignore

class TestVelodyne(Node):
    def __init__(self):
        super().__init__('test_velodyne')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/velodyne_points',
            self.callback,
            10
        )

    def callback(self, msg):
        points = list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True))
        print("Received points:", len(points))

def main(args=None):
    rclpy.init(args=args)
    node = TestVelodyne()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()