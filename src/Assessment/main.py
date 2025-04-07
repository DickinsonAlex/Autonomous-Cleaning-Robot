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

# This class subscribes to the /odom topic and logs the position and orientation of the robot.
# Testing confirms testting area is between -0.1 and 0.1 for both x and y coordinates.
class OdomTracker:
    def __init__(self, node):
        self.node = node  # Parent node to create the subscriber
        self.position = (0.0, 0.0)
        self.yaw = 0.0

        self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        yaw = math.degrees(2 * math.atan2(qz, qw))

        self.position = (x, y)
        self.yaw = yaw

    def get_position(self):
        return self.position

    def get_yaw(self):
        return self.yaw

    def log_info(self):
        x, y = self.position
        self.node.get_logger().info(f"[OdomTracker] Position: ({x:.2f}, {y:.2f}) | Yaw: {self.yaw:.2f}°")

# Class to represent each detected object (either a box or a marker)
class DetectedObject:
    def __init__(self, type, x, y, color, timestamp=None):
        self.obj_type = type  # Either 'box' or 'marker'
        self.position = (x, y)
        self.color = color
        self.timestamp = timestamp if timestamp else time.time()

    def is_same_color(self, other):
        return self.color == other.color

    def distance_to(self, other):
        return math.hypot(self.position[0] - other.position[0],
                          self.position[1] - other.position[1])

    def update_position(self, x, y, timestamp=None):
        self.position = (x, y)
        self.timestamp = timestamp if timestamp else time.time()

# Object tracker that stores seen boxes and markers, manages duplicates and pairing
class ObjectTracker:
    def __init__(self):
        self.boxes = []
        self.markers = []
        self.markers_angles = []
        self.completed_colors = set()

    def update_or_add(self, new_obj):
        obj_list = self.boxes if new_obj.obj_type == 'box' else self.markers

        for existing in obj_list:
            if existing.is_same_color(new_obj) and existing.distance_to(new_obj) < 1:
                # If close enough and same color, update
                existing.update_position(*new_obj.position)
                return

        # Otherwise it's a new object
        obj_list.append(new_obj)

    def add_marker_angle(self, color, position, angle):
        # Check if similar measurement exists already
        if any(s[0] == color and s[1] == position for s in self.markers_angles):
            return

        # Prevent adding if a marker with that color already confirmed
        if any(s.color == color for s in self.markers):
            return

        # Collect three angles for triangulation
        self.markers_angles.append((color, position, angle))

        if len([s for s in self.markers_angles if s[0] == color]) >= 3:
            positions = [s[1] for s in self.markers_angles if s[0] == color]
            avg_x = sum(p[0] for p in positions) / len(positions)
            avg_y = sum(p[1] for p in positions) / len(positions)
            new_marker = DetectedObject('marker', avg_x, avg_y, color)
            self.update_or_add(new_marker)
            self.markers_angles = [s for s in self.markers_angles if s[0] != color]

    def get_closest_pair(self):
        best_pair = None
        best_distance = float('inf')
        for box in self.boxes:
            for marker in self.markers:
                if box.is_same_color(marker) and box.color not in self.completed_colors:
                    dist = box.distance_to(marker)
                    if dist < best_distance:
                        best_distance = dist
                        best_pair = (box, marker)
        return best_pair

    def all_boxes_moved(self):
        active_colors = set(box.color for box in self.boxes)
        return active_colors.issubset(self.completed_colors)

class TidyBotController(Node):
    def __init__(self):
        super().__init__('tidybot_controller')

        # === INIT: Core States ===
        self.bridge = CvBridge()
        self.state = 'SCANNING'
        self.push_phase = 'INIT'
        self.tracker = ObjectTracker()
        self.twist = Twist()

        # === SCAN vars ===
        self.scan_state = 'None'
        self.scan_stage = 0
        self.scan_points = []

        # === Odometry vars ===
        self.odom = OdomTracker(self)
        self.create_timer(2.0, self.odom.log_info)
        
        ## === LIDAR vars ===
        self.lidar_data = []

        # === ROS Subscribers and Publishers ===
        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.image_callback, 1)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.velocity_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.control_loop)

        # === State Variables ===
        self.create_timer(2.0, self.log_bot_info)

    # Callback to update LIDAR data
    def scan_callback(self, msg):
        self.lidar_data = msg.ranges

    # Image callback that detects colored boxes and markers using color segmentation and LIDAR
    def image_callback(self, msg):
        if not self.lidar_data:
            return

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # HSV ranges for red and green
        colors = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([50, 100, 100], [70, 255, 255])
        }

        hfov = 90  # Camera horizontal FOV

        for color, (lower, upper) in colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Skip detections at edge of frame
                if x <= 0 or y <= 0 or x + w >= image.shape[1] - 1 or y + h >= image.shape[0] - 1:
                    continue

                cx = x + w // 2
                cy = y + h // 2
                angle_offset = ((cx / image.shape[1]) - 0.5) * hfov

                # Map pixel to LIDAR index
                index = int(len(self.lidar_data) * (angle_offset + hfov / 2) / hfov)

                if 0 <= index < len(self.lidar_data):
                    distance = self.lidar_data[index]

                    if distance == float('inf') or distance <= 0.1:
                        continue

                    abs_angle = math.radians(self.odom.get_yaw() + angle_offset)
                    world_x = self.odom.get_position()[0] + distance * math.cos(abs_angle)
                    world_y = self.odom.get_position()[1] + distance * math.sin(abs_angle)

                    # Lower half of image = box; upper = marker
                    if cy > image.shape[0] // 2:
                        obj = DetectedObject('box', world_x, world_y, color)
                        self.tracker.update_or_add(obj)
                    else:
                        if color not in self.tracker.completed_colors:
                            self.tracker.add_marker_angle(color, self.odom.get_position(), abs_angle)

        # Debug visualization
        cv2.imshow("TidyBot View", image)
        cv2.waitKey(1)

    # Main control loop dispatches based on state
    def control_loop(self):
        # FSM state handler map
        state_handlers = {
            'SCANNING': self.handle_scanning,
            'PUSHING': self.handle_pushing,
            'ROAMING': self.handle_roaming,
            'FINISHED': self.handle_finished
        }
        handler = state_handlers.get(self.state)
        if handler:
            handler()

    # === SCANNING: Move in triangle and rotate to detect objects ===
    def handle_scanning(self):
        if self.scan_stage == 0:
            self.scan_points = [self.odom.get_position()]
            for i in range(3):
                angle = math.radians(i * 120)
                x = self.odom.get_position()[0] + 0.2 * math.cos(angle)
                y = self.odom.get_position()[1] + 0.2 * math.sin(angle)
                self.scan_points.append((x, y))
            self.scan_stage = 1

        elif self.scan_stage <= 3:
            if self.move_to_target(self.scan_points[self.scan_stage]):
                if self.scan_state == "None":
                    self.scan_start_angle = self.odom.get_yaw()
                    self.scan_state = "In-Progress"

                elif self.scan_state == "In-Progress":
                    # Spin in place for scan
                    self.twist.linear.x = 0.0
                    self.twist.angular.z = 1.0
                    self.velocity_publisher.publish(self.twist)
                    if (self.odom.get_yaw() - self.scan_start_angle + 360) % 360 >= 350:
                        self.scan_state = "Finished"

                if self.scan_state == "Finished":
                    self.scan_state = "None"
                    self.scan_stage += 1
        else:
            self.scan_stage = 0
            self.scan_state = "None"
            self.stop()
            pair = self.tracker.get_closest_pair()
            if pair:
                self.target_box = pair[0]
                self.target_marker = pair[1]
                self.state = 'PUSHING'

    # === PUSHING: Move behind box and push to marker ===
    def handle_pushing(self):
        if self.push_phase == "INIT":
            bx, by = self.target_box.current_position
            sx, sy = self.target_marker.current_position

            direction = np.array([sx - bx, sy - by])
            norm = np.linalg.norm(direction)

            if norm == 0:
                self.tracker.completed_colors.add(self.target_box.color)
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
                self.move_target = self.target_marker.current_position
                self.arrived = False

        elif self.push_phase == "PUSH_FORWARD":
            if not self.arrived:
                self.arrived = self.move_to_target(self.move_target)
            else:
                self.stop()
                self.tracker.completed_colors.add(self.target_box.color)
                self.state = "SCANNING"
                self.push_phase = "INIT"

        self.get_logger().info(f"Current phase: {self.push_phase}")
        self.get_logger().info(f"Moving to {self.move_target}")

    # === ROAMING: Random motion used during testing ===
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

    # === FINISHED: Shutdown after all work ===
    def handle_finished(self):
        self.stop()
        self.destroy_node()
        rclpy.shutdown()

    # === Helper for moving to target point with rotation logic ===
    def move_to_target(self, target):
        tx, ty = target
        dx = tx - self.odom.get_position()[0]
        dy = ty - self.odom.get_position()[1]

        distance = math.hypot(dx, dy)
        if distance < 0.2:
            self.stop()
            return True

        target_angle = math.degrees(math.atan2(dy, dx)) % 360
        angle_diff = (target_angle - self.odom.get_yaw() + 360) % 360
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

    # Stop the robot by zeroing velocity
    def stop(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.velocity_publisher.publish(self.twist)

    # Debug printout for state, box status, and marker positions
    def log_bot_info(self):
        box_colors = [b.color for b in self.tracker.boxes if b.color not in self.tracker.completed_colors]
        marker_colors = [s.color for s in self.tracker.markers]
        self.get_logger().info(f"\nCurrent State: {self.state} \nBoxes: {box_colors} \nMarkers: {marker_colors}")

def main(args=None):
    rclpy.init(args=args)
    node = TidyBotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
