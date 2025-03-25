import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import numpy as np
import time

class TidyBotController(Node):
    def __init__(self):
        super().__init__('tidybot_controller')
        
        # Publisher to control robot velocity
        self.velocity_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscribers for camera and LiDAR data
        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.image_callback, 1)
        self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        
        # Timer to run the control loop
        self.loopspeed = 0.1 # Control loop speed in seconds
        self.timer = self.create_timer(self.loopspeed, self.control_loop)
        
        # Initialize Twist message for robot velocity
        self.twist = Twist()
        
        # Initialize CvBridge for converting ROS Image messages to OpenCV images
        self.bridge = CvBridge()
        
        # Variables to store processed data
        self.detected_blocks = []  # List of detected blocks (color, x, y, z)
        self.detected_signs = []   # List of detected signs (color, x, y, z)
        self.lidar_data = None     # LiDAR data
        
        # State machine variables
        self.state = "SCANNING" # Initial state

        # Location Variables
        self.current_angle = 0.0  # Initialize current angle
        self.current_position = (0.0, 0.0)  # Initialize current position

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Process image to detect objects
        self.detect_objects(cv_image)

        # Display image with detected objects
        cv2.imshow('Image window', cv_image)
        cv2.waitKey(1)

    def scan_callback(self, msg):
        # Store LiDAR data
        self.lidar_data = msg.ranges

    def detect_objects(self, image):
        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for red and green objects
        red_lower = np.array([0, 100, 100])
        red_upper = np.array([10, 255, 255])
        green_lower = np.array([50, 100, 100])
        green_upper = np.array([70, 255, 255])
        
        # Create masks for red and green objects
        red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
        green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
        
        # Find contours of the objects
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
        # Check if LiDAR data is available
        if self.lidar_data is None:
            self.get_logger().warn("LiDAR data is not available yet.")
            return
        
        # Classify objects based on color, size, and distance
        for color, contours in [('red', red_contours), ('green', green_contours)]:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                center_x, center_y = x + w // 2, y + h // 2
                
                # Approximate the angle of the object in the LiDAR scan
                angle = (center_x / image.shape[1]) * 360  # Map x-coordinate to 360 degrees
            
                # Get the distance (z) from the LiDAR data
                z = self.lidar_data[int(angle) % len(self.lidar_data)]

                # Initial classification by height
                if h > 50:
                    # Height suggests it's likely a sign
                    self.detected_signs.append((color, center_x, center_y, z))
                elif h < 20:
                    # Height suggests it's likely a close up block
                    self.detected_blocks.append((color, center_x, center_y, z))
                else:
                    # Height suggests it could be a block or a distant sign
                    if area > 2500:
                        # Small height but large area; likely a distant sign
                        self.detected_signs.append((color, center_x, center_y, z))
                    else:
                        # Medium height and area; likely a block
                        self.detected_blocks.append((color, center_x, center_y, z))
        
        
        # If the blocks are the same color, and close together, consider them as one block, average the position and remove the duplicates
        for i in range(len(self.detected_blocks)):
            for j in range(i+1, len(self.detected_blocks)):
                if self.detected_blocks[i][0] == self.detected_blocks[j][0]:
                    distance = np.sqrt((self.detected_blocks[i][1] - self.detected_blocks[j][1])**2 + (self.detected_blocks[i][2] - self.detected_blocks[j][2])**2)
                    if distance < 50:
                        self.detected_blocks[i] = (self.detected_blocks[i][0], (self.detected_blocks[i][1] + self.detected_blocks[j][1])//2, (self.detected_blocks[i][2] + self.detected_blocks[j][2])//2)
                        self.detected_blocks.pop(j)
                        break

        # If the signs are the same color, and close together, consider them as one sign, average the position and remove the duplicates
        for i in range(len(self.detected_signs)):
            for j in range(i+1, len(self.detected_signs)):
                if self.detected_signs[i][0] == self.detected_signs[j][0]:
                    distance = np.sqrt((self.detected_signs[i][1] - self.detected_signs[j][1])**2 + (self.detected_signs[i][2] - self.detected_signs[j][2])**2)
                    if distance < 50:
                        self.detected_signs[i] = (self.detected_signs[i][0], (self.detected_signs[i][1] + self.detected_signs[j][1])//2, (self.detected_signs[i][2] + self.detected_signs[j][2])//2)
                        self.detected_signs.pop(j)
                        break
    
        # Update the last object detection time
        if self.detected_blocks or self.detected_signs:
            self.last_object_detected_time = time.time()

    def roam(self):
        # Move forward for 3 seconds or until about to hit an obstacle
        start_time = time.time()
        while time.time() - start_time < 3:
            if self.lidar_data is not None:
                if min(self.lidar_data) < 0.5:
                    break
            self.twist.linear.x = 0.5
            self.twist.angular.z = 0.0
            self.velocity_publisher.publish(self.twist)

        # Rotate left 90 degrees
        start_angle = self.current_angle
        while abs(self.current_angle - start_angle) < 90:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.5
            self.velocity_publisher.publish(self.twist)
            # Update the current angle based on angular velocity and time
            self.current_angle += self.twist.angular.z * self.loopspeed
            self.current_angle %= 360

        # Change state to SCANNING
        self.change_state("SCANNING")

    def push(self):
        if not self.detected_blocks or not self.detected_signs:
            print("No blocks or signs detected.")
            self.change_state("SCANNING")
            return

        # Find closest matching block and sign
        closest_pair = min(
            ((block, sign) for block in self.detected_blocks for sign in self.detected_signs if block[0] == sign[0]),
            key=lambda pair: np.hypot(pair[0][1] - pair[1][1], pair[0][2] - pair[1][2]),
            default=(None, None)
        )

        block, sign = closest_pair
        if not block or not sign:
            print("No matching block-sign pair found.")
            return

        print(f"Pushing block '{block[0]}' towards sign '{sign[0]}'")

        # Calculate vector from sign to block
        push_direction = np.array([sign[1] - block[1], sign[2] - block[2]], dtype=float)
        push_direction /= np.linalg.norm(push_direction)

        # Position the robot slightly behind the block
        approach_distance = 0.2  # Adjust as necessary
        target_position = np.array([block[1], block[2]]) - push_direction * approach_distance

        # Step 1: Move robot to target_position behind block
        self.move_to_position(target_position)

        # Step 2: Rotate robot to face the sign
        target_angle = np.degrees(np.arctan2(push_direction[1], push_direction[0]))
        self.rotate_to_angle(target_angle)

        # Step 3: Push the block towards the sign
        block_to_sign_distance = np.linalg.norm([sign[1] - block[1], sign[2] - block[2]])
        self.move_forward(block_to_sign_distance + approach_distance)

    def move_to_position(self, target_position):
        while True:
            current_pos = np.array(self.current_position)
            distance = np.linalg.norm(target_position - current_pos)
            if distance <= 0.05:
                break

            angle_to_target = np.degrees(np.arctan2(target_position[1] - current_pos[1], target_position[0] - current_pos[0]))
            self.rotate_to_angle(angle_to_target, tolerance=5)

            self.twist.linear.x = 0.3
            self.twist.angular.z = 0.0
            self.velocity_publisher.publish(self.twist)

            # Update position estimate
            self.current_position = (
                self.current_position[0] + self.twist.linear.x * self.loopspeed * np.cos(np.radians(angle_to_target)),
                self.current_position[1] + self.twist.linear.x * self.loopspeed * np.sin(np.radians(angle_to_target))
            )

        self.stop_movement()

    def rotate_to_angle(self, target_angle, tolerance=3):
        target_angle %= 360
        while abs(self.current_angle - target_angle) > tolerance:
            angle_diff = (target_angle - self.current_angle + 540) % 360 - 180
            rotation_speed = 0.3 if angle_diff > 0 else -0.3

            self.twist.linear.x = 0.0
            self.twist.angular.z = rotation_speed
            self.velocity_publisher.publish(self.twist)

            self.current_angle += rotation_speed * self.loopspeed
            self.current_angle %= 360

        self.stop_movement()

    def move_forward(self, distance):
        moved_distance = 0
        while moved_distance < distance:
            self.twist.linear.x = 0.3
            self.twist.angular.z = 0.0
            self.velocity_publisher.publish(self.twist)

            moved_distance += self.twist.linear.x * self.loopspeed
            self.current_position = (
                self.current_position[0] + self.twist.linear.x * self.loopspeed * np.cos(np.radians(self.current_angle)),
                self.current_position[1] + self.twist.linear.x * self.loopspeed * np.sin(np.radians(self.current_angle))
            )

        self.stop_movement()

    def stop_movement(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.velocity_publisher.publish(self.twist)


    def scan(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.5  # Spin in place
        self.velocity_publisher.publish(self.twist)

        # Update the current angle based on angular velocity and time
        self.current_angle += self.twist.angular.z * self.loopspeed
        self.current_angle %= 360  # Keep angle within 0-360 degrees

        # Check if any objects are detected, and if the appropriate sign location with the same color is known
        if self.detected_blocks and self.detected_signs:
            for block in self.detected_blocks:
                for sign in self.detected_signs:
                    if block[0] == sign[0]:
                        self.change_state("PUSHING")
                        break

        # Check if a full 360-degree spin has been completed
        if abs(self.current_angle - 0.0) < 1e-2:  # Close to 0 degrees
            self.change_state("ROAMING")

    def finished(self):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.velocity_publisher.publish(self.twist)
        self.get_logger().info("Exploration finished.")
        self.destroy_node()
        rclpy.shutdown()
        
    def change_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            print(f"State: {self.state}")

    def control_loop(self):
        #Print the detected objects and signs
        print(f"Detected blocks: {self.detected_blocks}")
        print(f"Detected signs: {self.detected_signs}")
        print(f"Current state: {self.state}")        

        # State machine logic with a switch-case statement
        if self.state == "ROAMING":
            self.roam()
        elif self.state == "PUSHING":
            self.push()
        elif self.state == "SCANNING":
            self.scan()
        elif self.state == "FINISHED":
            self.finished()            

def main(args=None):
    rclpy.init(args=args)
    node = TidyBotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()