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

# Base class for detected objects
class DetectedObject:
    def __init__(self, type, x, y, color, timestamp=None):
        self.obj_type = type
        self.current_position = (x, y)
        self.color = color
        self.timestamp = timestamp if timestamp else time.time()

    def is_same_color(self, other):
        return self.color == other.color

    def distance_to(self, other):
        return math.hypot(self.current_position[0] - other.current_position[0], self.current_position[1] - other.current_position[1])

    def update_position(self, x, y, timestamp=None):
        self.current_position = (x, y)
        self.timestamp = timestamp if timestamp else time.time()
      
class ObjectTracker:
    def __init__(self):
        self.blocks = []
        self.signs = []
        self.completed_colors = set()

    def update_or_add(self, new_obj):
        if new_obj.color in self.completed_colors:
            return  # Skip already completed colors

        obj_list = self.blocks if new_obj.obj_type == 'block' else self.signs
        for existing in obj_list:
            if existing.is_same_color(new_obj) and existing.distance_to(new_obj) < 0.5:
                existing.update_position(*new_obj.current_position)
                return
        obj_list.append(new_obj)

    def get_closest_pair(self):
        # Return the closest unmatched block-sign pair of the same color
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
        # Check if all active colors have been completed
        active_colors = set(block.color for block in self.blocks)
        return active_colors.issubset(self.completed_colors)

# Main controller node for TidyBot
class TidyBotController(Node):
    def __init__(self):
        super().__init__('tidybot_controller')
        self.bridge = CvBridge()
        self.state = 'SCANNING'
        self.push_phase = "INIT"
        self.tracker = ObjectTracker()
        self.twist = Twist()

        self.current_position = (0.0, 0.0)
        self.current_angle = 0.0
        self.lidar_data = []

        self.scanning_started = False
        self.scan_start_time = None

        # ROS subscriptions and publisher
        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.image_callback, 1)
        self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.create_subscription(Odometry, 'odom', self.odometry_callback, 10)
        self.velocity_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.control_loop)

    def odometry_callback(self, msg):
        # Update robot position and orientation from odometry
        self.current_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.current_angle = math.degrees(2 * math.atan2(qz, qw))

    def scan_callback(self, msg):
        # Store current LiDAR scan data
        self.lidar_data = msg.ranges

    def image_callback(self, msg):
        if not self.lidar_data:
            return

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define HSV color ranges for detection
        colors = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([50, 100, 100], [70, 255, 255])
        }

        hfov = 90  # camera horizontal field of view in degrees

        # Iterate over color masks and detect objects
        for color, (lower, upper) in colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Skip objects touching the image edge (not fully visible)
                if x <= 0 or y <= 0 or x + w >= image.shape[1] - 1 or y + h >= image.shape[0] - 1:
                    continue

                cx = x + w // 2
                cy = y + h // 2
                angle_offset = ((cx / image.shape[1]) - 0.5) * hfov
                index = int(len(self.lidar_data) * (angle_offset + hfov/2) / hfov)

                if 0 <= index < len(self.lidar_data):
                    distance = self.lidar_data[index]
                    if distance == float('inf') or distance <= 0.1:
                        continue

                    abs_angle = math.radians(self.current_angle + angle_offset)

                    obj_type = 'block' if cy > image.shape[0] // 2 else 'sign' # 

                    # Calculate world coordinates 
                    if obj_type == 'block':
                        world_x = self.current_position[0] + distance * math.cos(abs_angle)
                        world_y = self.current_position[1] + distance * math.sin(abs_angle)
                        obj = DetectedObject('block', world_x, world_y, color)
                        self.tracker.update_or_add(obj)
                    else:
                        #signs can't rely on the distance from the lidar as blocks may be in the way
                        #will need to use triangulation FILL CODE HERE    
                        pass                  

        # Show the processed image
        cv2.imshow("TidyBot View", image)
        cv2.waitKey(1)

    def control_loop(self):
        self._loop_count = getattr(self, '_loop_count', 0) + 1
        if self._loop_count % 20 == 0:
            #self.log_bot_info()
            pass

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
        if not self.scanning_started:
            #self.get_logger().info("SCANNING: Starting rotation.")
            self.scan_start_angle = self.current_angle
            self.scanning_started = True
            self.twist.linear.x = 0.0
            self.twist.angular.z = 1.0  # Set desired rotation speed

        # Re-publish twist every loop to maintain motion
        self.velocity_publisher.publish(self.twist)

        # Check for early detection of both a block and a matching sign
        pair = self.tracker.get_closest_pair()
        if pair:
            #self.get_logger().info("SCANNING: Found a valid block-sign pair early. Interrupting scan.")
            self.stop()
            self.scanning_started = False
            self.target_block, self.target_sign = pair
            self.state = 'PUSHING'
            return

        # Compute angle delta accounting for wraparound
        delta = (self.current_angle - self.scan_start_angle + 360) % 360
        #self.get_logger().info(f"SCANNING: Rotated {delta:.2f} degrees")
        if delta >= 350:
            #self.get_logger().info("SCANNING: Rotation complete.")
            self.stop()
            self.scanning_started = False
            if pair:
                #self.get_logger().info("SCANNING: Found a valid block-sign pair. Switching to PUSHING.")
                self.target_block, self.target_sign = pair
                self.state = 'PUSHING'
            elif not self.tracker.blocks or not self.tracker.signs:
                #self.get_logger().info("SCANNING: No valid block-sign pair. Switching to ROAMING.")
                self.state = 'ROAMING'

    def handle_pushing(self):
        if self.push_phase == "INIT":
            self.get_logger().info("PUSHING: Planning route")
            bx, by = self.target_block.current_position
            sx, sy = self.target_sign.current_position
            direction = np.array([sx - bx, sy - by])
            direction = direction / np.linalg.norm(direction)
            behind_block = bx - direction[0] * 0.5, by - direction[1] * 0.5
            self.move_target = behind_block
            self.push_phase = "MOVE_BEHIND"
            self.arrived = False
            self.rotation_done = False

            # Print the route
            self.get_logger().info(f"Route: {self.current_position} -> {behind_block} -> {self.target_sign.current_position}")
            self.get_logger().info(f"Block: {self.target_block.color} at {self.target_block.current_position}")
            self.get_logger().info(f"Sign: {self.target_sign.color} at {self.target_sign.current_position}")

        elif self.push_phase == "MOVE_BEHIND":
            if not self.arrived:
                self.arrived = self.move_to_target(self.move_target)
            else:
                self.push_phase = "PUSH_FORWARD"
                self.move_target = self.target_sign.current_position
                self.arrived = False
                self.rotation_done = False

        elif self.push_phase == "PUSH_FORWARD":
            if not self.arrived:
                self.arrived = self.move_to_target(self.move_target)
            else:
                self.get_logger().info("PUSHING: Complete")
                self.stop()
                self.tracker.completed_colors.add(self.target_block.color)
                self.state = "SCANNING"
                self.push_phase = "INIT"

    def handle_roaming(self):
        self.get_logger().info("State: ROAMING")
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
        self.get_logger().info("State: FINISHED")
        self.stop()
        self.destroy_node()
        rclpy.shutdown()

    def move_to_target(self, target):
        tx, ty = target
        dx = tx - self.current_position[0]
        dy = ty - self.current_position[1]
        distance = math.hypot(dx, dy)

        if distance < 0.05:
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
        # Stop all robot motion
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.velocity_publisher.publish(self.twist)

    def log_bot_info(self):
        block_colors = [b.color for b in self.tracker.blocks if b.color not in self.tracker.completed_colors]
        sign_colors = [s.color for s in self.tracker.signs]
        self.get_logger().info(f"\nCurrent State: {self.state} \nPosition: {self.current_position} \nAngle: {self.current_angle:.2f} \nBlocks: {block_colors} \nSigns: {sign_colors}")

# Main entry point
def main(args=None):
    rclpy.init(args=args)
    node = TidyBotController()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
