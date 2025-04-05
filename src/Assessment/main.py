import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math

class TidyBotController(Node):
    def __init__(self):
        super().__init__('tidybot_controller')
        self.bridge = CvBridge()
        self.state = 'SCANNING'

        self.twist = Twist()
        self.current_position = (0.0, 0.0)
        self.current_angle = 0.0
        self.lidar_data = []

        self.target_angle = None
        self.pushing_forward = False
        self.block_detected = False

        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.image_callback, 1)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odometry_callback, 10)
        self.velocity_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.control_loop)

    def odometry_callback(self, msg):
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.current_angle = math.degrees(2 * math.atan2(qz, qw))

    def scan_callback(self, msg):
        self.lidar_data = msg.ranges

    def image_callback(self, msg):
        if not self.lidar_data:
            return

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        colors = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([50, 100, 100], [70, 255, 255])
        }

        hfov = 90  # horizontal field of view in degrees
        self.block_detected = False

        for color, (lower, upper) in colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if x <= 0 or y <= 0 or x + w >= image.shape[1] - 1 or y + h >= image.shape[0] - 1:
                    continue

                cx = x + w // 2
                angle_offset = ((cx / image.shape[1]) - 0.5) * hfov
                index = int(len(self.lidar_data) * (angle_offset + hfov / 2) / hfov)

                if 0 <= index < len(self.lidar_data):
                    distance = self.lidar_data[index]
                    if distance == float('inf') or distance <= 0.1:
                        continue

                    avg_distance = np.mean([r for r in self.lidar_data if r < float('inf')])
                    if distance > avg_distance - 0.05:
                        continue

                    self.target_angle = (self.current_angle + angle_offset) % 360
                    self.block_detected = True
                    return

        cv2.imshow("TidyBot View", image)
        cv2.waitKey(1)

    def control_loop(self):
        self.get_logger().info(f"State: {self.state}, Block Detected: {self.block_detected}, Target Angle: {self.target_angle}")

        if self.front_wall_detected():
            self.state = 'REVERSING'
            self.twist.linear.x = -5.0
            self.twist.angular.z = 0.0
        else:
            self.state = 'SCANNING'


        if self.state == 'SCANNING':
            if self.block_detected:
                self.state = 'TURNING'
            else:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.5
        elif self.state == 'TURNING':
            angle_diff = (self.target_angle - self.current_angle + 360) % 360
            if angle_diff > 180:
                angle_diff -= 360

            if abs(angle_diff) > 5:
                self.twist.linear.x = 0.0
                self.twist.angular.z = 0.5 if angle_diff > 0 else -0.5
            else:
                self.state = 'PUSHING'
                self.pushing_forward = True
        elif self.state == 'PUSHING':
            self.twist.linear.x =  5.0
            self.twist.angular.z = 0.0
            time.sleep(5)
            self.stop()
            self.state = 'SCANNING'
            self.target_angle = None

        self.velocity_publisher.publish(self.twist)

    def front_wall_detected(self):
        if not self.lidar_data:
            return False

        # Use a stricter definition of "wall": all central rays must be below the threshold
        center_start = len(self.lidar_data) // 3
        center_end = 2 * len(self.lidar_data) // 3
        center_region = self.lidar_data[center_start:center_end]

        threshold = 0.5
        required_ratio = 0.9  # At least 90% of center rays must be close
        valid_distances = [d for d in center_region if d > 0 and d < threshold]

        return len(valid_distances) >= required_ratio * len(center_region)

    def stop(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.velocity_publisher.publish(self.twist)

def main(args=None):
    rclpy.init(args=args)
    node = TidyBotController()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
