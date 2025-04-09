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
from scipy.spatial.transform import Rotation as R

# Class to represent each detected object (either a box or a marker)
class DetectedObject:
    def __init__(self, type, x, y, color, timestamp=None):
        self.obj_type = type  # Either 'box' or 'marker'
        self.current_position = (x, y)
        self.color = color
        self.timestamp = timestamp if timestamp else time.time()

    def is_same_color(self, other):
        return self.color == other.color

    def distance_to(self, other):
        return math.hypot(self.current_position[0] - other.current_position[0],
                          self.current_position[1] - other.current_position[1])

    def update_position(self, x, y, timestamp=None):
        self.current_position = (x, y)
        self.timestamp = timestamp if timestamp else time.time()

# Class to track detected objects and their states aswell as the robot's state
class TidyBotController(Node):
    def __init__(self):
        super().__init__('tidybot_controller')

        # === INIT: Core States ===
        self.bridge = CvBridge()
        self.state = 'SCANNING'
        self.phase = 'INIT'
        self.twist = Twist()

        # === Objects ===
        self.boxes = []
        self.markers = []

        # === ROS Subscribers and Publishers ===
        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/limo/depth_camera_link/depth/image_raw', self.depth_image_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odometry_callback, 10)
        self.velocity_publisher = self.create_publisher(Twist, 'cmd_vel', 1)
        self.timer = self.create_timer(0.1, self.control_loop)

        # === Odometry vars ===
        self.current_position = (0.0, 0.0)
        self.current_angle = 0.0
        self.lidar_data = []

    # Callback for updating the robot's position and orientation
    def odometry_callback(self, msg):
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        orientation = msg.pose.pose.orientation

        # Get the z axis rotation from the robot's orientation quaternion
        rotation = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        euler = rotation.as_euler('xyz', degrees=True)
        self.current_angle = euler[2]  # Yaw is rotation around Z

    # Callback to update LIDAR data
    def scan_callback(self, msg):
        self.lidar_data = msg.ranges

    # Depth image callback, detects the depth of boxes and markers using depth camera
    def depth_image_callback(self, data):
        # Show a depth image
        depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        depth_image = cv2.resize(depth_image, (640, 480))
        depth_array = np.array(depth_image, dtype=np.float32)
        depth_array = np.clip(depth_array, 0, 10)  # Limit depth values to a maximum of 10 meters
        depth_image = (depth_array * 255 / 10).astype(np.uint8)  # Normalize to 0-255 range
        depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)  # Apply a color map for better visualization
        
        self.depthImage = depth_image
        
        cv2.imshow("Depth Image", depth_image)
        cv2.waitKey(1)

    # Image callback that detects colored boxes and markers using color segmentation and depth image
    def image_callback(self, msg):
        if not hasattr(self, 'depthImage'):
            return

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # HSV ranges for red and green
        colors = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([50, 100, 100], [70, 255, 255])
        }

        for color, (lower, upper) in colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 3)  # Draw contours for debugging

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Skip detections at edge of frame
                if x <= 0 or y <= 0 or x + w >= image.shape[1] - 1 or y + h >= image.shape[0] - 1:
                    continue

                cx = x + w // 2
                cy = y + h // 2

                # Use depth image to calculate distance
                depth_value = self.depthImage[cy, cx, 0] / 255.0 * 10.0  # Convert back to meters
                if depth_value <= 0.1 or depth_value > 10.0:
                    continue

                abs_angle = math.radians(self.current_angle)
                world_x = self.current_position[0] + depth_value * math.cos(abs_angle)
                world_y = self.current_position[1] + depth_value * math.sin(abs_angle)

                # Lower half of image = box; upper = marker
                if cy > image.shape[0] // 2:
                    obj = DetectedObject('box', world_x, world_y, color)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, color + ' box', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    self.update_or_add(obj)
                else:
                    obj = DetectedObject('marker', world_x, world_y, color)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(image, color + ' marker', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    self.update_or_add(obj)


        # Debug visualization
        cv2.imshow("TidyBot View", image)
        cv2.waitKey(1)

    def update_or_add(self, new_obj):
        obj_list = self.boxes if new_obj.obj_type == 'box' else self.markers

        for existing in obj_list:
            if existing.is_same_color(new_obj) and existing.distance_to(new_obj) < 1:
                # If close enough and same color, update
                existing.update_position(*new_obj.current_position)
                return

        # Otherwise it's a new object
        obj_list.append(new_obj)
        
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

    # Main control loop dispatches based on state
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
        handler = state_handlers.get(self.state)
        if handler:
            handler()

    # === SCANNING: Rotate in place to detect objects ===
    def handle_scanning(self):
        if self.phase == "In-Progress":
            # Spin in place for scan
            self.twist.linear.x = 0.0
            self.twist.angular.z = 1.0
            self.velocity_publisher.publish(self.twist)

            if (self.current_angle - self.scan_start_angle + 360) % 360 >= 350:
                self.phase = "Finished"

            pair = self.get_closest_pair()
            if (pair):
                self.stop()
                self.target_box = pair[0]
                self.target_marker = pair[1]
                self.state = 'PUSHING'
                self.phase = 'INIT'
        elif self.phase == "Finished":
            self.stop()
            self.phase = "None"
            self.state = "ROAMING"
        else:
            # Start scanning
            self.scan_start_angle = self.current_angle
            self.phase = "In-Progress"

    # === PUSHING: Move behind box and push to marker ===
    def handle_pushing(self):
        if self.phase == "INIT":
            bx, by = self.target_box.current_position
            sx, sy = self.target_marker.current_position

            direction = np.array([sx - bx, sy - by])
            norm = np.linalg.norm(direction)

            if norm == 0:
                self.state = "SCANNING"
                return

            direction = direction / norm
            self.move_target = bx - direction[0] + 0.5, by - direction[1] + 0.5
            self.phase = "MOVE_BEHIND"
            self.arrived = False

        elif self.phase == "MOVE_BEHIND":
            if not self.arrived:
                self.arrived = self.move_to_target(self.move_target)
            else:
                self.phase = "PUSH_FORWARD"
                self.move_target = self.target_marker.current_position
                self.arrived = False

        elif self.phase == "PUSH_FORWARD":
            if not self.arrived:
                self.arrived = self.move_to_target(self.move_target)
            else:
                self.stop()
                self.state = "SCANNING"
                self.phase = "None"

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
        target_x, target_y = target
        robot_x, robot_y = self.current_position

        # Calculate the angle to the target position
        delta_x = target_x - robot_x
        delta_y = target_y - robot_y
        angle_to_target = math.atan2(delta_y, delta_x)

        # Normalize the angle difference
        angle_diff = (angle_to_target - math.radians(self.current_angle) + math.pi) % (2 * math.pi) - math.pi

        # Calculate the distance to the target position
        distance_to_target = math.hypot(delta_x, delta_y)

        # Set the robot's linear and angular velocities
        if abs(angle_diff) > 0.1:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.5 if angle_diff > 0 else -0.5
        else:
            self.twist.linear.x = 0.3
            self.twist.angular.z = 0.0

        self.velocity_publisher.publish(self.twist)

        # Check if the robot is close enough to the target position
        if distance_to_target < 0.2:
            self.stop()
            return True

        return False

    # Stop the robot by zeroing velocity
    def stop(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.velocity_publisher.publish(self.twist)

    # Debug printout for state, box status, and marker positions
    def log_bot_info(self):
        box_colors = [b.color for b in self.boxes]
        marker_colors = [s.color for s in self.markers]
        self.get_logger().info(f"\nCurrent State: {self.state},{self.phase} \nPosition: {self.current_position} \nAngle: {self.current_angle:.2f} \nBoxes: {box_colors} \nMarkers: {marker_colors}")
        if self.state == 'PUSHING':
            self.get_logger().info(f"Moving to {self.move_target} from {self.current_position}")

def main(args=None):
    rclpy.init(args=args)
    node = TidyBotController()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
