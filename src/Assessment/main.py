import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import numpy as np

class TidyBotController(Node):
    def __init__(self):
        super().__init__('tidybot_controller')
        
        # Publisher to control robot velocity
        self.velocity_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscribers for camera and LiDAR data
        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.image_callback, 1)
        self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        
        # Timer to run the control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Initialize Twist message for robot velocity
        self.twist = Twist()
        
        # Initialize CvBridge for converting ROS Image messages to OpenCV images
        self.bridge = CvBridge()
        
        # Variables to store processed data
        self.detected_objects = []
        self.lidar_data = None

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
        
        # Store detected objects with their color and position
        self.detected_objects = []
        for contour in red_contours:
            x, y, w, h = cv2.boundingRect(contour)
            self.detected_objects.append(('red', x + w // 2, y + h // 2))
        for contour in green_contours:
            x, y, w, h = cv2.boundingRect(contour)
            self.detected_objects.append(('green', x + w // 2, y + h // 2))

    def control_loop(self):
        # Implement control logic to push objects towards the designated sides
        if self.detected_objects:
            for color, x, y in self.detected_objects:
                if color == 'red':
                    self.push_object(x, y, 'left')
                elif color == 'green':
                    self.push_object(x, y, 'right')
        else:
            # Stop the robot if no objects are detected
            self.twist.linear.x = 0.0
            self.twist.angular.z = 1.0
            self.velocity_publisher.publish(self.twist)

    def push_object(self, x, y, direction):
        # Simple proportional controller to push objects
        error_x = x - 320  # Assuming image width is 640
        self.twist.linear.x = 0.2
        self.twist.angular.z = -error_x * 0.005
        
        # Publish velocity command
        self.velocity_publisher.publish(self.twist)

def main(args=None):
    rclpy.init(args=args)
    node = TidyBotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()