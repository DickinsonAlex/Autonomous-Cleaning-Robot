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
        self.loopspeed = 0.5 # Control loop speed in seconds
        self.timer = self.create_timer(self.loopspeed, self.control_loop)
        
        # Initialize Twist message for robot velocity
        self.twist = Twist()
        
        # Initialize CvBridge for converting ROS Image messages to OpenCV images
        self.bridge = CvBridge()
        
        # Variables to store processed data
        self.detected_blocks = []  # List of detected blocks (color, x, y, z)
        self.detected_signs = []   # List of detected signs (color, x, y, z)
        self.moved_blocks = []     # List of blocks that have been moved to signs
        self.lidar_data = None     # LiDAR data
        
        # State machine variables
        self.state = "SCANNING" # Initial state

        # Location Variables
        self.current_angle = 0.0  # Initialize current angle
        self.current_position = (0.0, 0.0)  # Initialize current position
        self.last_object_detected_time = None  # Time when an object was last detected

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
                self.get_logger().warn("LiDAR data is not available. Skipping object detection for now.")
                return
    
        # Ensure LiDAR data is valid before proceeding
        if all(np.isinf(range_val) or range_val == 0 for range_val in self.lidar_data):
            self.get_logger().warn("LiDAR data is invalid. Retrying object detection later.")
            return
        
        # Classify objects based on color, size, and distance
        for color, contours in [('red', red_contours), ('green', green_contours)]:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

                camera_hfov = 90  # Adjust based on camera's specs
                angle = ((center_x / image.shape[1]) - 0.5) * camera_hfov # Approximate the angle of the object
                distance = min(self.lidar_data[int(angle)], 10.0)  # Limit the distance to 10 meters

                # Calculate the estimated position of the object, using self.current_angle and self.current_position
                world_x = self.current_position[0] + distance * np.cos(np.radians(self.current_angle + angle))
                world_y = self.current_position[1] + distance * np.sin(np.radians(self.current_angle + angle))

                # Using the y position guess if the object is a block or a sign
                if center_y > image.shape[0] // 2:
                    self.detected_blocks.append((color, world_x, world_y))
                else:
                    self.detected_signs.append((color, world_x, world_y))
        
            # Remove duplicate blocks
            unique_blocks = []
            for block in self.detected_blocks:
                if not any(np.hypot(block[1]-b[1], block[2]-b[2]) < 0.5 and block[0]==b[0] for b in unique_blocks):
                    unique_blocks.append(block)
            self.detected_blocks = unique_blocks

            # Average the sign positions for each color
            sign_positions = {}
            for sign in self.detected_signs:
                if sign[0] not in sign_positions:
                    sign_positions[sign[0]] = []
                sign_positions[sign[0]].append(sign[1:])
            self.detected_signs = [(color, np.mean([pos[0] for pos in sign_positions[color]]), np.mean([pos[1] for pos in sign_positions[color]]) ) for color in sign_positions]

    
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

        print(f"Pushing block '{block} towards sign '{sign} from position {self.current_position}")

        # Calculate vector from sign to block
        target_position = np.array(sign[1:])
        block_position = np.array(block[1:])
        direction = target_position - block_position
        direction /= np.linalg.norm(direction)
        target_position = block_position - direction * 0.5

        print(f"Moving to position {target_position} from position {self.current_position}")

        # Position the robot slightly behind the block
        self.rotate_to_angle(np.degrees(np.arctan2(target_position[1] - self.current_position[1], target_position[0] - self.current_position[0])))
        self.move_forward(np.hypot(target_position[0] - self.current_position[0], target_position[1] - self.current_position[1]))
        
        # Rotate robot to face the sign
        angle_to_sign = np.degrees(np.arctan2(sign[2] - self.current_position[1], sign[1] - self.current_position[0]))
        self.rotate_to_angle(angle_to_sign)

        # Push the block towards the sign
        distance_to_sign = np.hypot(sign[1] - block[1], sign[2] - block[2])
        self.move_forward(distance_to_sign)

        # Remove the block from the list of detected blocks, add it to the list of pushed blocks
        self.detected_blocks.remove(block)
        self.moved_blocks.append(block)

    def rotate_to_angle(self, target_angle):
        while abs((self.current_angle - target_angle + 180) % 360 - 180) > 2:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.5 if ((target_angle - self.current_angle + 360) % 360) < 180 else -0.5
            self.velocity_publisher.publish(self.twist)
            rclpy.spin_once(self, timeout_sec=self.loopspeed)  # Allow ROS to process odometry updates

        self.stop_movement()

    def move_forward(self, distance):
        moved_distance = 0
        while moved_distance < distance:
            self.twist.linear.x = 0.3
            self.twist.angular.z = 0.0
            self.velocity_publisher.publish(self.twist)

            # Update position estimate
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