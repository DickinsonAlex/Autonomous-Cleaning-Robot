import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math

# This script subscribes to the /odom topic and logs the position and orientation of the robot.
# Testing confirms testting area is between -0.1 and 0.1 for both x and y coordinates.
class OdomLogger(Node):
    def __init__(self):
        super().__init__('odom_logger')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.current_position = (0.0, 0.0)
        self.current_angle = 0.0
        self.timer = self.create_timer(5.0, self.print_odom_info)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        yaw = math.degrees(2 * math.atan2(qz, qw))

        self.current_position = (x, y)
        self.current_angle = yaw

    def print_odom_info(self):
        x, y = self.current_position
        self.get_logger().info(f"Odom Position => x: {x:.2f}, y: {y:.2f}, yaw: {self.current_angle:.2f}°")

def main(args=None):
    rclpy.init(args=args)
    node = OdomLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
