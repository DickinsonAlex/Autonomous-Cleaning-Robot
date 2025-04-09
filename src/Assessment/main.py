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
from scipy.spatial import ConvexHull

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

    def update_type(self, new_type):
        self.obj_type = new_type

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
        self.pushed_boxes = []
        self.markers = []
        self.wall_markers = {
            'red': None,   # e.g., North
            'green': None  # e.g., East
        }

        self.lidar_hit_points = []
        self.lidar_data = []

        # HSV ranges for red and green
        self.colors = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([50, 100, 100], [70, 255, 255])
        }

        # === ROS Subscribers and Publishers ===
        self.create_subscription(Image, '/limo/depth_camera_link/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/limo/depth_camera_link/depth/image_raw', self.depth_image_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odometry_callback, 10)
        self.velocity_publisher = self.create_publisher(Twist, 'cmd_vel', 1)
        self.create_timer(0.1, self.control_loop)
        self.create_timer(0.1, self.minimap_callback)

        # === Odometry vars ===
        self.current_position = (0.0, 0.0)
        self.current_angle = 0.0

    # Callback for updating the robot's position and orientation
    def odometry_callback(self, msg):
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        orientation = msg.pose.pose.orientation

        # Get yaw in radians
        rotation = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        euler = rotation.as_euler('xyz', degrees=False)  # Now outputs in radians
        self.current_angle = euler[2]  # Yaw

    def lidar_callback(self, msg):
        self.lidar_data = msg.ranges

        for i, distance in enumerate(msg.ranges):
            if 0.1 < distance < 10.0:
                angle = msg.angle_min + i * msg.angle_increment
                x_local = distance * math.cos(angle)
                y_local = distance * math.sin(angle)

                robot_x, robot_y = self.current_position
                robot_angle = self.current_angle
                x_world = robot_x + x_local * math.cos(robot_angle) - y_local * math.sin(robot_angle)
                y_world = robot_y + x_local * math.sin(robot_angle) + y_local * math.cos(robot_angle)

                # Makes sure the point isn't too close to an existing point
                if not any(math.hypot(x_world - px, y_world - py) < 0.1 for px, py in self.lidar_hit_points):
                    # Append the point to the list
                    self.lidar_hit_points.append((x_world, y_world))
                else:
                    # If it's too close, update the existing point
                    for j, (px, py) in enumerate(self.lidar_hit_points):
                        if math.hypot(x_world - px, y_world - py) < 0.1:
                            self.lidar_hit_points[j] = (x_world, y_world)
                            break
        
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

    # Image callback that detects colored boxes and markers using color segmentation and depth image
    def image_callback(self, msg):
        if not hasattr(self, 'depthImage'):
            return

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        fov_x = math.radians(90)
        image_center_x = image.shape[1] / 2

        for color, (lower, upper) in self.colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if x <= 0 or y <= 0 or x + w >= image.shape[1] - 1 or y + h >= image.shape[0] - 1:
                    continue

                cx = x + w // 2
                cy = y + h // 2
                depth_value = self.depthImage[cy, cx, 0] / 255.0

                pixel_offset = cx - image_center_x
                angle_offset = (pixel_offset / image.shape[1]) * fov_x
                abs_angle = self.current_angle + angle_offset

                world_x = self.current_position[0] + depth_value * math.cos(abs_angle)
                world_y = self.current_position[1] + depth_value * math.sin(abs_angle)

                pushed = any(d < 0.5 for d in self.lidar_data[::10])

                if cy > image.shape[0] // 2:
                    obj_type = 'pushed_box' if pushed else 'box'
                    color_text = (255, 0, 0) if pushed else (0, 255, 0)
                    label = f"{color} {obj_type}"
                    obj = DetectedObject(obj_type, world_x, world_y, color)
                    self.update_or_add(obj)
                else:
                    if self.wall_markers[color] is None:
                        angle = self.current_angle + angle_offset
                        self.wall_markers[color] = (world_x, world_y, angle)
                        marker = DetectedObject('marker', world_x, world_y, color)
                        self.markers.append(marker)

        cv2.imshow("TidyBot View", image)
        cv2.waitKey(1)

    def get_outer_square(self, points):
        if len(points) < 2:
            return []

        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)

        # Square side length = max of width or height
        width = max_x - min_x
        height = max_y - min_y
        side = max(width, height)

        # Recenter the square to ensure it's square-shaped
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2

        half = side / 2
        # Define corners clockwise from top-left
        return [
            (cx - half, cy + half),  # Top-left
            (cx + half, cy + half),  # Top-right
            (cx + half, cy - half),  # Bottom-right
            (cx - half, cy - half)   # Bottom-left
        ]

    # Using known locations, plot the boxes and markers on a minimap
    def minimap_callback(self):
        cv2.namedWindow("Debug Screen", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Debug Screen", 640, 480)
        map_image = np.zeros((500, 600, 3), dtype=np.uint8)
        map_image[:] = (200, 200, 200)

        for i in range(0, 600, 50):
            cv2.line(map_image, (i, 0), (i, 500), (150, 150, 150), 1)
        for i in range(0, 500, 50):
            cv2.line(map_image, (0, i), (600, i), (150, 150, 150), 1)

        minimap_scale = 50
        center_x = 300
        center_y = 250

        # Draw square from outermost points
        square_corners = self.get_outer_square(self.lidar_hit_points)
        if len(square_corners) == 4:
            for i in range(4):
                pt1 = square_corners[i]
                pt2 = square_corners[(i + 1) % 4]
                x1 = int(pt1[0] * minimap_scale) + center_x
                y1 = int(-pt1[1] * minimap_scale) + center_y
                x2 = int(pt2[0] * minimap_scale) + center_x
                y2 = int(-pt2[1] * minimap_scale) + center_y
                cv2.line(map_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Robot
        rx = int(self.current_position[0] * minimap_scale) + center_x
        ry = int(-self.current_position[1] * minimap_scale) + center_y
        cv2.circle(map_image, (rx, ry), 10, (255, 255, 0), -1)

        dx = int((self.current_position[0] + math.cos(self.current_angle)) * minimap_scale) + center_x
        dy = int((-self.current_position[1] - math.sin(self.current_angle)) * minimap_scale) + center_y
        cv2.arrowedLine(map_image, (rx, ry), (dx, dy), (0, 0, 255), 2)

        # Boxes / pushed boxes
        for obj in self.boxes + self.pushed_boxes:
            ox = int(obj.current_position[0] * minimap_scale) + center_x
            oy = int(-obj.current_position[1] * minimap_scale) + center_y
            color = (0, 255, 0) if obj.color == 'green' else (0, 0, 255)
            label = 'Box' if obj.obj_type == 'box' else 'Pushed Box'
            cv2.circle(map_image, (ox, oy), 5, color, -1)
            cv2.putText(map_image, label, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Draw out the Targets
        if self.state == 'PUSHING':
            if hasattr(self, 'move_target'):
                tx, ty = self.move_target
                tx = int(tx * minimap_scale) + center_x
                ty = int(-ty * minimap_scale) + center_y
                cv2.circle(map_image, (tx, ty), 5, (255, 0, 255), -1)
                cv2.putText(map_image, '1', (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # Draw line between robot and move target
                cv2.line(map_image, (rx, ry), (tx, ty), (255, 255, 255), 3)

                if hasattr(self, 'target_box'):
                    bx = int(self.target_box.current_position[0] * minimap_scale) + center_x
                    by = int(-self.target_box.current_position[1] * minimap_scale) + center_y
                    cv2.circle(map_image, (bx, by), 5, (255, 0, 0), -1)
                    cv2.putText(map_image, '2', (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    # Draw line between move target and box
                    cv2.line(map_image, (tx, ty), (bx, by), (255, 255, 255), 3)

                    if hasattr(self, 'target_marker'):
                        mx = int(self.target_marker.current_position[0] * minimap_scale) + center_x
                        my = int(-self.target_marker.current_position[1] * minimap_scale) + center_y
                        cv2.circle(map_image, (mx, my), 5, (255, 255, 0), -1)
                        cv2.putText(map_image, '3', (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                        # Draw line between box and marker
                        cv2.line(map_image, (bx, by), (mx, my), (255, 255, 255), 3)


        # In the top left, draw the current state
        cv2.putText(map_image, f"State: {self.state}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(map_image, f"Phase: {self.phase}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(map_image, f"Position: {self.current_position}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(map_image, f"Angle: {math.degrees(self.current_angle):.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw walls by color on correct square side
        for color, marker in self.wall_markers.items():
            if marker:
                wx, wy, wall_angle = marker
                angle_deg = (math.degrees(wall_angle) + 360) % 360

                # Determine which wall: N / E / S / W
                wall_index = None
                if 45 <= angle_deg < 135:
                    wall_index = 0  # Top wall
                elif 135 <= angle_deg < 225:
                    wall_index = 3  # Left wall
                elif 225 <= angle_deg < 315:
                    wall_index = 2  # Bottom wall
                else:
                    wall_index = 1  # Right wall

                if square_corners:
                    p1 = square_corners[wall_index]
                    p2 = square_corners[(wall_index + 1) % 4]
                    x1 = int(p1[0] * minimap_scale) + center_x
                    y1 = int(-p1[1] * minimap_scale) + center_y
                    x2 = int(p2[0] * minimap_scale) + center_x
                    y2 = int(-p2[1] * minimap_scale) + center_y

                    wall_color = (0, 255, 0) if color == 'green' else (0, 0, 255)
                    cv2.line(map_image, (x1, y1), (x2, y2), wall_color, 3)
                    cv2.putText(map_image, f"{color.capitalize()} Wall", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


        cv2.imshow("Minimap [Debug]", map_image)
        cv2.waitKey(1)

    def update_or_add(self, new_obj):
        if new_obj.obj_type == 'box':
            obj_list = self.boxes 
        elif new_obj.obj_type == 'pushed_box':
            obj_list = self.pushed_boxes
        else:
            obj_list = self.markers

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
                if box.is_same_color(marker):
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
        if self.phase == "INIT":
            self.scan_start_angle = self.current_angle
            self.phase = "In-Progress"

        elif self.phase == "In-Progress":
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.5
            self.velocity_publisher.publish(self.twist)

            angle_turned = (self.current_angle - self.scan_start_angle + 2 * math.pi) % (2 * math.pi)
            if angle_turned >= math.radians(350):
                self.phase = "Finished"

            pair = self.get_closest_pair()
            if pair:
                self.stop()
                self.target_box = pair[0]
                self.target_marker = pair[1]
                self.state = 'PUSHING'
                self.phase = 'INIT'

        elif self.phase == "Finished":
            self.stop()
            self.phase = "INIT"  # Reset for next time
            self.state = "ROAMING"

        else:
            self.phase = "INIT"  # Reset if phase is not recognized

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
                self.pushed_boxes.append(self.target_box)  # Add to pushed boxes
                self.boxes.remove(self.target_box) # Remove the box after pushing

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
        angle_diff = (angle_to_target - self.current_angle + math.pi) % (2 * math.pi) - math.pi

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

def main(args=None):
    rclpy.init(args=args)
    node = TidyBotController()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
