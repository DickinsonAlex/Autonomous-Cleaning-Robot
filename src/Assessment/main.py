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

# Class representing a detected object (either a block or a sign)
class DetectedObject:
    def __init__(self, type, x, y, color, timestamp=None):
        self.obj_type = type  # 'block' or 'sign'
        self.current_position = (x, y)  # (x, y) world coordinates
        self.color = color
        self.timestamp = timestamp if timestamp else time.time()

    def is_same_color(self, other):
        return self.color == other.color

    def distance_to(self, other):
        # Euclidean distance to another object
        return math.hypot(self.current_position[0] - other.current_position[0],
                          self.current_position[1] - other.current_position[1])

    def update_position(self, x, y, timestamp=None):
        # Update object's world position and timestamp
        self.current_position = (x, y)
        self.timestamp = timestamp if timestamp else time.time()

# Tracks and manages detected blocks and signs
class ObjectTracker:
    def __init__(self):
        self.blocks = []
        self.signs = []
        self.signs_angles = []
        self.completed_colors = set()

    def update_or_add(self, new_obj):
        # Ignore if color is already completed
        if new_obj.color in self.completed_colors:
            return
        obj_list = self.blocks if new_obj.obj_type == 'block' else self.signs
        for existing in obj_list:
            if existing.is_same_color(new_obj) and existing.distance_to(new_obj) < 0.5:
                existing.update_position(*new_obj.current_position)
                return
        obj_list.append(new_obj)

    def add_sign_angle(self, color, position, angle):
        if any(s[0] == color and s[1] == position for s in self.signs_angles):
            return
        if any(s.color == color for s in self.signs):
            return
        self.signs_angles.append((color, position, angle))
        if len([s for s in self.signs_angles if s[0] == color]) >= 3:
            # Triangulate sign location from multiple sightings
            positions = [s[1] for s in self.signs_angles if s[0] == color]
            avg_x = sum(p[0] for p in positions) / len(positions)
            avg_y = sum(p[1] for p in positions) / len(positions)
            new_sign = DetectedObject('sign', avg_x, avg_y, color)
            self.update_or_add(new_sign)
            self.signs_angles = [s for s in self.signs_angles if s[0] != color]

    def get_closest_pair(self):
        # Find the closest matching block-sign color pair
        best_pair = None
        best_distance = float('inf')
        for block in self.blocks:
            for sign in self.signs:
                if block.is_same_color(sign) and block.color not in self.completed_colors:
                    dist = block.distance_to(sign)
                    if dist < best_distance:
                        best_distance = dist
                        best_pair = (block, sign)
        return best_pair

    def all_blocks_moved(self):
        active_colors = set(block.color for block in self.blocks)
        return active_colors.issubset(self.completed_colors)

# Main controller class for TidyBot
class TidyBotController(Node):
    def __init__(self):
        super().__init__('tidybot_controller')
        self.bridge = CvBridge()
        self.state = 'SCANNING'
        self.push_phase = 'INIT'
        self.tracker = ObjectTracker()
        self.twist = Twist()
        self.scan_state = 'None'
        self.scan_stage = 0
        self.scan_points = []
        self.current_position = (0.0, 0.0)
        self.current_angle = 0.0
        self.lidar_data = []

        # Subscriptions
        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.image_callback, 1)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odometry_callback, 10)

        # Publisher
        self.velocity_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Loop timer
        self.timer = self.create_timer(0.1, self.control_loop)

    def odometry_callback(self, msg):
        # Convert odometry pose to position and yaw angle
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
        hfov = 90  # Camera horizontal field of view

        for color, (lower, upper) in colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if x <= 0 or y <= 0 or x + w >= image.shape[1] - 1 or y + h >= image.shape[0] - 1:
                    continue
                cx = x + w // 2
                cy = y + h // 2
                angle_offset = ((cx / image.shape[1]) - 0.5) * hfov
                index = int(len(self.lidar_data) * (angle_offset + hfov / 2) / hfov)
                if 0 <= index < len(self.lidar_data):
                    distance = self.lidar_data[index]
                    if distance == float('inf') or distance <= 0.1:
                        continue
                    abs_angle = math.radians(self.current_angle + angle_offset)
                    world_x = self.current_position[0] + distance * math.cos(abs_angle)
                    world_y = self.current_position[1] + distance * math.sin(abs_angle)
                    if cy > image.shape[0] // 2:
                        obj = DetectedObject('block', world_x, world_y, color)
                        self.tracker.update_or_add(obj)
                    else:
                        if color not in self.tracker.completed_colors:
                            self.tracker.add_sign_angle(color, self.current_position, abs_angle)

        cv2.imshow("TidyBot View", image)
        cv2.waitKey(1)

    def control_loop(self):
        self._loop_count = getattr(self, '_loop_count', 0) + 1
        if self._loop_count % 20 == 0:
            self.log_bot_info()  # Log bot info every 2 seconds

        # FSM state handler map
        state_handlers = {
            'SCANNING': self.handle_scanning,
            'PUSHING': self.handle_pushing,
            'ROAMING': self.handle_roaming,
            'FINISHED': self.handle_finished
        }
        state_handler = state_handlers.get(self.state)
        if state_handler:
            state_handler()

    def handle_scanning(self):
        if self.scan_stage == 0:
            # Generate three scan points in a circle
            self.scan_points = [self.current_position]
            for i in range(3):
                angle = math.radians(i * 120)
                x = self.current_position[0] + 0.2 * math.cos(angle)
                y = self.current_position[1] + 0.2 * math.sin(angle)
                self.scan_points.append((x, y))
            self.scan_stage = 1
        elif self.scan_stage <= 3:
            if self.move_to_target(self.scan_points[self.scan_stage]):
                if self.scan_state == "None":
                    self.scan_start_angle = self.current_angle
                    self.scan_state = "In-Progress"
                elif self.scan_state == "In-Progress":
                    self.twist.linear.x = 0.0
                    self.twist.angular.z = 1.0
                    self.velocity_publisher.publish(self.twist)
                    if (self.current_angle - self.scan_start_angle + 360) % 360 >= 350:
                        self.scan_state = "Finished"
                if self.scan_state == "Finished":
                    self.scan_state = "None"
                    self.scan_stage += 1
        else:
            # All scan points visited, transition
            self.scan_stage = 0
            self.scan_state = "None"
            self.stop()
            pair = self.tracker.get_closest_pair()
            if pair:
                self.target_block = pair[0]
                self.target_sign = pair[1]
                self.state = 'PUSHING'

    def handle_pushing(self):
        if self.push_phase == "INIT":
            # Compute direction and target behind block
            bx, by = self.target_block.current_position
            sx, sy = self.target_sign.current_position
            direction = np.array([sx - bx, sy - by])
            norm = np.linalg.norm(direction)
            if norm == 0:
                self.tracker.completed_colors.add(self.target_block.color)
                self.state = "SCANNING"
                return
            direction = direction / norm
            self.move_target = bx - direction[0] * 0.5, by - direction[1] * 0.5
            self.push_phase = "MOVE_BEHIND"
            self.arrived = False
        elif self.push_phase == "MOVE_BEHIND":
            if not self.arrived:
                self.arrived = self.move_to_target(self.move_target)
            else:
                self.push_phase = "PUSH_FORWARD"
                self.move_target = self.target_sign.current_position
                self.arrived = False
        elif self.push_phase == "PUSH_FORWARD":
            if not self.arrived:
                self.arrived = self.move_to_target(self.move_target)
            else:
                self.stop()
                self.tracker.completed_colors.add(self.target_block.color)
                self.state = "SCANNING"
                self.push_phase = "INIT"

        self.get_logger().info(f"Current phase: {self.push_phase}")
        self.get_logger().info(f"Moving to {self.move_target}")
        self.get_logger().info(f"Current position: {self.current_position}")

    def handle_roaming(self):
        self.twist.linear.x = 0.3
        self.twist.angular.z = 0.0
        self.velocity_publisher.publish(self.twist)
        time.sleep(2)
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.5
        self.velocity_publisher.publish(self.twist)
        time.sleep(2)
        self.stop()
        self.state = 'SCANNING'

    def handle_finished(self):
        self.stop()
        self.destroy_node()
        rclpy.shutdown()

    def move_to_target(self, target):
        # Rotate to face the target and drive toward it
        tx, ty = target
        dx = tx - self.current_position[0]
        dy = ty - self.current_position[1]
        distance = math.hypot(dx, dy)
        if distance < 0.2:
            self.stop()
            return True
        target_angle = math.degrees(math.atan2(dy, dx)) % 360
        angle_diff = (target_angle - self.current_angle + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360
        if abs(angle_diff) > 5:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.5 if angle_diff > 0 else -0.5
        else:
            self.twist.linear.x = 0.3
            self.twist.angular.z = 0.0
        self.velocity_publisher.publish(self.twist)
        return False

    def stop(self):
        # Stop robot motion
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.velocity_publisher.publish(self.twist)

    def log_bot_info(self):
        # Debug logging of state and tracked objects
        block_colors = [b.color for b in self.tracker.blocks if b.color not in self.tracker.completed_colors]
        sign_colors = [s.color for s in self.tracker.signs]
        self.get_logger().info(f"\nCurrent State: {self.state} \nPosition: {self.current_position} \nAngle: {self.current_angle:.2f} \nBlocks: {block_colors} \nSigns: {sign_colors}")

# Entry point

def main(args=None):
    rclpy.init(args=args)
    node = TidyBotController()
    rclpy.spin(node)

if __name__ == '__main__':
    main()