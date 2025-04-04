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
        self.signs_angles = []
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

    def add_sign_angle(self, color, position, angle):

        if any(s[0] == color and s[1] == position for s in self.signs_angles):
            # If the color and position already exist in the signs_angles list, we can skip adding it
            return
        
        if any(s.color == color for s in self.signs):
            # If the color is already in the signs list, we can skip adding it to the signs_angles
            return

        # Add the colour, recorded position, and sign angle to the signs_angles
        self.signs_angles.append((color, position, angle))

        # If theres three angles of the same color, we can triangulate the position and add it to the signs, and remove the angles
        if len([s for s in self.signs_angles if s[0] == color]) >= 3:
            # Triangulate the position of the sign
            positions = [s[1] for s in self.signs_angles if s[0] == color]
            angles = [s[2] for s in self.signs_angles if s[0] == color]
            avg_x = sum(p[0] for p in positions) / len(positions)
            avg_y = sum(p[1] for p in positions) / len(positions)
            avg_angle = sum(angles) / len(angles)

            # Create a new sign object and add it to the signs list
            new_sign = DetectedObject('sign', avg_x, avg_y, color)
            self.update_or_add(new_sign)

            # Remove the angles from the list
            self.signs_angles = [s for s in self.signs_angles if s[0] != color]
        
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

        self.scan_state = "None"
        self.starting_angle = 0.0
        self.scan_stage = 0
        self.scan_points = []

        self.current_position = (0.0, 0.0)
        self.current_angle = 0.0
        self.lidar_data = []

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

    ### Image callback to process camera images in conjunction with LiDAR data
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
                        # Skip if the sign is a completed color
                        if color not in self.tracker.completed_colors:
                            #signs can't rely on the distance from the lidar as blocks may be in the way
                            self.tracker.add_sign_angle(color, self.current_position, abs_angle)
                        pass                  

        # Show the processed image
        cv2.imshow("TidyBot View", image)
        cv2.waitKey(1)

    ### CONTROL LOOP ###
    # This is the main control loop that runs at a fixed rate
    # It checks the current state of the robot and calls the appropriate handler
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

    ### SCANNING STATE ###
    # In this state, the robot spins in place 360, noting down the location of blocks, and angle of signs.
    # It will do this three times, in a small triangle, so that it can record three angles for each sign.
    # Once there are three angles, it will triangulate the location of the sign.
    # If it finds a block and a sign of the same color, it will stop scanning and move to the pushing state.
    # If it does not find a pair, it will stop scanning and move to the roaming state.
    def handle_scanning(self):
        if self.scan_stage == 0:
            self.scan_points = [self.current_position] #First point is the center
            # Add the three points of the triangle
            for i in range(3):
                angle = math.radians(i * 120)
                x = self.current_position[0] + 0.2 * math.cos(angle)
                y = self.current_position[1] + 0.2 * math.sin(angle)
                self.scan_points.append((x, y))
            self.scan_stage = 1
        elif self.scan_stage <= 3:
            # Check if we have reached the next scan point
            if self.move_to_target(self.scan_points[self.scan_stage]):
                if self.scan_state == "None":
                    # We must have just arrived at the next point
                    self.get_logger().info("Reached scan point " + str(self.scan_stage) + " out of 3")
                    self.scan_start_angle = self.current_angle
                    self.scan_state = "In-Progress"
                elif self.scan_state == "In-Progress":
                    # We are at the next point, so we can start scanning
                    self.twist.linear.x = 0.0
                    self.twist.angular.z = 1.0  # Set desired rotation speed
                    self.velocity_publisher.publish(self.twist) # Re-publish twist every loop to maintain motion

                    if (self.current_angle - self.scan_start_angle + 360) % 360 >= 350: # If we have rotated 360 degrees
                        # We have completed the scan
                        self.scan_state = "Finished"
                if self.scan_state == "Finished":
                    # If we have completed the scan, we can move onto the next point
                    self.get_logger().info("Completed scan stage " + str(self.scan_stage))
                    self.scan_state = "None"
                    self.scan_stage += 1
        else:
            # We have completed the scan, so we can stop
            self.scan_stage = 0
            self.scan_state = "None"
            self.stop()

            # Check if we have found a pair of blocks and signs
            pair = self.tracker.get_closest_pair()
            if pair:
                self.target_block = pair[0]
                self.target_sign = pair[1]
                self.get_logger().info(f"Found pair: Block {self.target_block.color} at {self.target_block.current_position}, Sign {self.target_sign.color} at {self.target_sign.current_position}")
                self.state = 'PUSHING'

    def handle_pushing(self):
        if self.push_phase == "INIT":
            self.get_logger().info("Planning route")
            bx, by = self.target_block.current_position
            sx, sy = self.target_sign.current_position
            direction = np.array([sx - bx, sy - by])
            direction = direction / np.linalg.norm(direction)
            behind_block = bx - direction[0] * 0.5, by - direction[1] * 0.5
            self.move_target = behind_block
            self.push_phase = "MOVE_BEHIND"
            self.arrived = False
            self.rotation_done = False

        elif self.push_phase == "MOVE_BEHIND":
            self.get_logger().info("Moving behind block")
            if not self.arrived:
                self.arrived = self.move_to_target(self.move_target)
            else:
                self.push_phase = "PUSH_FORWARD"
                self.move_target = self.target_sign.current_position
                self.arrived = False
                self.rotation_done = False

        elif self.push_phase == "PUSH_FORWARD":
            self.get_logger().info("Pushing block")
            if not self.arrived:
                self.arrived = self.move_to_target(self.move_target)
            else:
                self.get_logger().info("PUSHING: Complete")
                self.stop()
                self.tracker.completed_colors.add(self.target_block.color)
                self.state = "SCANNING"
                self.push_phase = "INIT"

    def handle_roaming(self):
        self.get_logger().info("Roaming")
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
