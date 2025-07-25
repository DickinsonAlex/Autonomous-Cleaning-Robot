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
class Box:
    def __init__(self, type, x, y, color, timestamp=None):
        self.position = (x, y)
        self.obj_type = type
        self.color = color

    def update_position(self, x, y, timestamp=None):
        self.position = (x, y)

class Marker:
    def __init__(self, x, y, color, angle):
        self.position = (x, y)
        self.color = color
        self.angle = angle  # Angle of the marker in the world

        # Normalize the angle to the range [0, 2π)
        normalized_angle = self.angle % (2 * math.pi)

        if math.pi / 4 <= normalized_angle < 3 * math.pi / 4:
            self.wall_direction = 'North'
        elif 3 * math.pi / 4 <= normalized_angle < 5 * math.pi / 4:
            self.wall_direction = 'West'
        elif 5 * math.pi / 4 <= normalized_angle < 7 * math.pi / 4:
            self.wall_direction = 'South'
        else:
            self.wall_direction = 'East'

# Class to track detected objects and their states aswell as the robot's state
class TidyBotController(Node):
    def __init__(self):
        super().__init__('tidybot_controller')

        # Core states and variables
        self.bridge = CvBridge()
        self.state = 'RETURNING'  # Initial state
        self.phase = 'INIT'
        self.twist = Twist()
        self.has_started = False
        self.has_Spun = False

        # Objects to track
        self.boxes = []
        self.markers = []
        self.pushed_boxes = []

        self.square_corners = []  # To store the corners of the outer square
        self.lidar_hit_points = []
        self.lidar_data = []
        self.raw_depth_array = []

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

        # odometry variables
        self.position = (0.0, 0.0)
        self.current_angle = 0.0

    # Callback for updating the robot's position and orientation
    def odometry_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        orientation = msg.pose.pose.orientation

        # Get yaw in radians
        rotation = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        euler = rotation.as_euler('xyz', degrees=False)  # Now outputs in radians
        self.current_angle = euler[2]  # Yaw

    # Lidar callback to process the laser scan data
    def lidar_callback(self, msg):
        self.lidar_data = msg.ranges

        for i, distance in enumerate(msg.ranges):
            if 0.1 < distance < 10.0:
                angle = msg.angle_min + i * msg.angle_increment
                x_local = distance * math.cos(angle)
                y_local = distance * math.sin(angle)

                robot_x, robot_y = self.position
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
        depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        self.raw_depth_array = np.array(depth_image, dtype=np.float32)

    # Image callback that detects colored boxes and markers using color segmentation and depth image
    def image_callback(self, msg):
        if not hasattr(self, 'raw_depth_array'):
            return

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # === Camera parameters gathered from node info ===
        fx = 448.625  # from CameraInfo
        fy = 448.625
        cx_intrinsic = 320.5
        cy_intrinsic = 240.5

        for color, (lower, upper) in self.colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Ignore contours near the edge
                margin = 5
                img_h, img_w = image.shape[:2]
                if x <= margin or y <= margin or x + w >= img_w - margin or y + h >= img_h - margin:
                    continue

                # Centroid of contour
                px = x + w // 2
                py = y + h // 2

                # Get depth at the centroid
                if not hasattr(self, 'raw_depth_array') or self.raw_depth_array is None or len(self.raw_depth_array) == 0:
                    continue

                if py >= self.raw_depth_array.shape[0] or px >= self.raw_depth_array.shape[1]:
                    continue

                depth = self.raw_depth_array[py, px]
                if np.isnan(depth) or depth <= 0.1 or depth > 10.0:
                    continue

                # === Back-project to 3D camera coordinates ===
                x_cam = (px - cx_intrinsic) * depth / fx
                y_cam = (py - cy_intrinsic) * depth / fy
                z_cam = depth

                # === Transform to world frame ===
                robot_x, robot_y = self.position
                robot_theta = self.current_angle

                # Convert from camera to robot frame (assuming camera faces forward)
                x_base = z_cam
                y_base = -x_cam

                # Now transform to world frame
                world_x = robot_x + x_base * math.cos(robot_theta) - y_base * math.sin(robot_theta)
                world_y = robot_y + x_base * math.sin(robot_theta) + y_base * math.cos(robot_theta)
                
                # Check if the object is against any wall
                against_wall = False
                for corner in self.square_corners:
                    next_corner = self.square_corners[(self.square_corners.index(corner) + 1) % 4]
                    midpoint = ((corner[0] + next_corner[0]) / 2, (corner[1] + next_corner[1]) / 2)
                    distance = math.hypot(world_x - midpoint[0], world_y - midpoint[1]) 
                    if distance < 0.5:  # Adjust this threshold as needed
                        against_wall = True

                if py > img_h // 2:
                    obj_type = 'pushed_box' if against_wall else 'box'
                    obj = Box(obj_type, world_x, world_y, color)
                    self.update_or_add_box(obj)
                else:
                    # Get the angle from 0,0 in radians
                    angle = math.atan2(world_y, world_x)
                    obj = Marker(world_x, world_y, color, angle)
                    self.add_marker(obj)

    # Function to get the outer square corners from the LiDAR hit points
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

        # Draw grid lines
        for i in range(0, 600, 50):
            cv2.line(map_image, (i, 0), (i, 500), (150, 150, 150), 1)
        for i in range(0, 500, 50):
            cv2.line(map_image, (0, i), (600, i), (150, 150, 150), 1)

        debug_scale = 50
        center_x = 300
        center_y = 250

        # Draw LiDAR points
        for pt in self.lidar_hit_points:
            x = int(pt[0] * debug_scale) + center_x
            y = int(-pt[1] * debug_scale) + center_y
            cv2.circle(map_image, (x, y), 1, (20, 20, 20), -1)

        # Draw square from outermost points
        self.square_corners = self.get_outer_square(self.lidar_hit_points)
        if len(self.square_corners) == 4:
            for i in range(4):
                pt1 = self.square_corners[i]
                pt2 = self.square_corners[(i + 1) % 4]
                x1 = int(pt1[0] * debug_scale) + center_x
                y1 = int(-pt1[1] * debug_scale) + center_y
                x2 = int(pt2[0] * debug_scale) + center_x
                y2 = int(-pt2[1] * debug_scale) + center_y
                cv2.line(map_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw robot position and direction
        rx = int(self.position[0] * debug_scale) + center_x
        ry = int(-self.position[1] * debug_scale) + center_y
        cv2.circle(map_image, (rx, ry), 12, (0, 0, 0), -1)
        cv2.circle(map_image, (rx, ry), 10, (255, 255, 255), -1)

        dx = int((self.position[0] + math.cos(self.current_angle)) * debug_scale) + center_x
        dy = int((-self.position[1] - math.sin(self.current_angle)) * debug_scale) + center_y
        cv2.arrowedLine(map_image, (rx, ry), (dx, dy), (0, 0, 0), 2)

        # Draw the boxes / pushed boxes
        for obj in self.boxes + self.pushed_boxes:
            ox = int(obj.position[0] * debug_scale) + center_x
            oy = int(-obj.position[1] * debug_scale) + center_y
            
            if obj.color == 'red':
                color = (0, 0, 255)
            elif obj.color == 'green':
                color = (0, 255, 0)

            #If pushed, darken the color
            if obj.obj_type == 'pushed_box':
                color = tuple(int(c * 0.3) for c in color)

            cv2.circle(map_image, (ox, oy), 5, color, -1)

        # Draw out the movement targets
        if self.state == 'PUSHING':
            if hasattr(self, 'behind_target'):
                tx, ty = self.behind_target
                tx = int(tx * debug_scale) + center_x
                ty = int(-ty * debug_scale) + center_y
                cv2.circle(map_image, (tx, ty), 10, (255, 255, 255), -1)
                cv2.circle(map_image, (tx, ty), 8, (255, 0, 0), -1)
                cv2.putText(map_image, 'BEHIND', (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            if hasattr(self, 'target_box'):
                bx = int(self.target_box.position[0] * debug_scale) + center_x
                by = int(-self.target_box.position[1] * debug_scale) + center_y
                cv2.circle(map_image, (bx, by), 10, (255, 255, 255), -1)
                cv2.circle(map_image, (bx, by), 8, (255, 0, 0), -1)
                cv2.putText(map_image, 'BOX', (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            if hasattr(self, 'target_marker'):
                mx = int(self.target_marker.position[0] * debug_scale) + center_x
                my = int(-self.target_marker.position[1] * debug_scale) + center_y
                cv2.circle(map_image, (mx, my), 10, (255, 255, 255), -1)
                cv2.circle(map_image, (mx, my), 8, (255, 0, 0), -1)
                cv2.putText(map_image, 'GOAL', (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # In the top left, draw the current state
        cv2.putText(map_image, f"State: {self.state}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(map_image, f"Phase: {self.phase}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(map_image, f"Has Started: {self.has_started}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(map_image, f"Position: {self.position}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(map_image, f"Angle: {math.degrees(self.current_angle):.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # If pushing, draw the target box and marker
        if self.state == 'PUSHING':
            cv2.putText(map_image, f"Target Box: {self.target_box.color} : {self.target_box.position}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(map_image, f"Target Marker: {self.target_marker.color} : {self.target_marker.position}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw walls by color on correct square side
        for marker in self.markers:
            if self.square_corners:

                wx, wy = marker.position
                p1, p2 = 0,0

                # Pair the points by direction (N/E/S/W)
                if marker.wall_direction == 'North':
                    p1 = self.square_corners[0]
                    p2 = self.square_corners[1]
                elif marker.wall_direction == 'East':
                    p1 = self.square_corners[1]
                    p2 = self.square_corners[2]
                elif marker.wall_direction == 'South':
                    p1 = self.square_corners[2]
                    p2 = self.square_corners[3]
                elif marker.wall_direction == 'West':
                    p1 = self.square_corners[3]
                    p2 = self.square_corners[0]


                x1 = int(p1[0] * debug_scale) + center_x
                y1 = int(-p1[1] * debug_scale) + center_y
                x2 = int(p2[0] * debug_scale) + center_x
                y2 = int(-p2[1] * debug_scale) + center_y

                # Update the position of the marker to be the midpoint of the two corners in world position
                midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                marker.position = (midpoint[0] , midpoint[1])

                if marker.color == 'red':
                    wall_color = (0, 0, 255)
                elif marker.color == 'green':
                    wall_color = (0, 255, 0)
                else: wall_color = (255, 255, 0)  # Default color for unknown

                cv2.line(map_image, (x1, y1), (x2, y2), wall_color, 3)
                #cv2.putText(map_image, f"{marker.wall_direction.capitalize()}, {marker.color.capitalize()} Wall", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Draw grey circles at all the different wall marker locations (square_corners)
            for corner in self.square_corners:
                wx = int(corner[0] * debug_scale) + center_x
                wy = int(-corner[1] * debug_scale) + center_y
                cv2.circle(map_image, (wx, wy), 7, (150, 150, 150), -1)

        cv2.imshow("Debug Screen", map_image)
        cv2.waitKey(1)

    # Function to update or add a box or pushed box to the list
    def update_or_add_box(self, new_obj):
        if not self.has_started:
            # If the robot is still in startup, don't add boxes yet
            return

        if new_obj.obj_type == 'box':
            for existing in self.boxes:
                # Check if the new object is close enough to an existing one and the same color
                distance = math.hypot(existing.position[0] - new_obj.position[0],
                                     existing.position[1] - new_obj.position[1])
                if existing.color == new_obj.color and distance < 0.6:
                    existing.update_position(*new_obj.position)
                    return
            self.boxes.append(new_obj)  # Otherwise it's a new box

        elif new_obj.obj_type == 'pushed_box':
            for existing in self.pushed_boxes:
                # Check if the new object is close enough to an existing one and the same color
                distance = math.hypot(existing.position[0] - new_obj.position[0],
                                     existing.position[1] - new_obj.position[1])
                if existing.color == new_obj.color and distance < 0.6:
                    existing.update_position(*new_obj.position)
                    return
            # If it doesn't exist, add it to the list
            self.pushed_boxes.append(new_obj)

        return
    
    # Function to add a marker to the list of markers
    def add_marker(self, new_obj):
        if not self.has_started:
            # If the robot is still in startup, don't add markers yet
            return
        
        # There can only be one marker of each color, so check if it exists
        for existing in self.markers:
            if existing.color == new_obj.color:
                return
        # If it doesn't exist, add it to the list
        self.markers.append(new_obj)
        
    # Function to get the closest box and marker pair
    def get_closest_pair(self):
        best_pair = None
        best_distance = float('inf')
        for box in self.boxes:
            for marker in self.markers:
                if box.color == marker.color:
                    dist = math.hypot(box.position[0] - marker.position[0],
                                     box.position[1] - marker.position[1])
                    if dist < best_distance:
                        best_distance = dist
                        best_pair = (box, marker)
        return best_pair

    # Main control loop dispatches based on state
    def control_loop(self):
        # FSM state handler map
        state_handlers = {
            'STARTUP': self.handle_scanning,  # Start scanning, but don't detect markers (Needs to figure out bounds first)
            'SCANNING': self.handle_scanning,
            'PUSHING': self.handle_pushing,
            'ROAMING': self.handle_roaming,
            'RETURNING': self.handle_returning,
            'FINISHED': self.handle_finished
        }
        handler = state_handlers.get(self.state)
        if handler:
            handler()

    # === SCANNING: Rotate in place to find boxes and markers ===
    def handle_scanning(self):        
        if self.phase == "INIT":
            self.scan_start_angle = self.current_angle
            self.phase = "In-Progress"

        elif self.phase == "In-Progress" or self.state == "STARTUP":
            self.twist.linear.x = 0.0
            self.twist.angular.z = 1.0
            self.velocity_publisher.publish(self.twist)

            # Check if the robot has turned half way around, preventing premature has_started
            if angle_turned := (self.current_angle - self.scan_start_angle + 2 * math.pi) % (2 * math.pi) >= math.radians(180):
                self.has_Spun = True            

            angle_turned = (self.current_angle - self.scan_start_angle + 2 * math.pi) % (2 * math.pi)
            if angle_turned >= math.radians(350):
                self.phase = "INIT"  # Reset for next time

                if self.state == "STARTUP" and self.has_Spun:
                    self.state = "SCANNING"  # Start scanning for boxes and markers
                    self.has_started = True
                else:
                    self.stop()
                    self.state = "ROAMING" # Start roaming to find boxes

                pair = self.get_closest_pair()
                if pair:
                    self.stop()
                    self.target_box = pair[0]
                    self.target_marker = pair[1]
                    self.state = 'PUSHING' # Start pushing the box
        else:
            self.phase = "INIT"  # Reset if phase is not recognized

    # === PUSHING: Move behind box and push to marker ===
    def handle_pushing(self):
        if self.phase == "INIT":
            bx, by = self.target_box.position
            sx, sy = self.target_marker.position

            # Vector from box to marker (push direction)
            push_vec = np.array([sx - bx*1.1, sy - by*1.1])
            norm = np.linalg.norm(push_vec)

            if norm == 0:
                self.state = "SCANNING"
                return

            # Normalize the push direction
            push_dir = push_vec / norm

            # Set a fixed distance to stand behind the box
            stand_back_distance = 0.5  # meters behind the box

            # Move target is behind the box along the negative push direction
            move_x = bx - push_dir[0] * stand_back_distance
            move_y = by - push_dir[1] * stand_back_distance

            self.behind_target = (move_x, move_y)
            self.phase = "BEHIND_TARGET_BOX"
            self.arrived = False

        elif self.phase == "BEHIND_TARGET_BOX":
            if not self.arrived:
                self.arrived = self.move_to_target(self.behind_target)
            else:
                self.phase = "PUSH_FORWARD"
                self.arrived = False

        elif self.phase == "PUSH_FORWARD":
            if not self.arrived:
                self.arrived = self.move_to_target(self.target_marker.position)
            else:
                self.stop()
                self.state = "RETURNING"
                self.phase = "INIT"
                self.boxes.remove(self.target_box) # Remove the box after pushing

    # === RETURNING: Move back to the starting position at 0,0 ===
    def handle_returning(self):
        if self.phase == "INIT":
            self.behind_target = (0.0, 0.0)
            self.phase = "MOVE_BACK"
            self.arrived = False

        elif self.phase == "MOVE_BACK":
            if not self.arrived:
                self.arrived = self.move_to_target(self.behind_target)
            else:
                self.stop()
                if self.has_started:
                    self.state = "SCANNING"
                else:
                    self.state = "STARTUP"
                self.phase = "INIT"


    # === ROAMING: Random motion used to ensure nothing is hidden ===
    def handle_roaming(self):
        # Randomly pick -1 or 1 to turn left or right
        turn_direction = np.random.choice([-3, 3])
        self.twist.linear.x = 1.2
        self.twist.angular.z = float(turn_direction)
        self.velocity_publisher.publish(self.twist)
        time.sleep(0.5)  # Sleep for a bit to simulate random motion
        self.stop()
        self.state = 'SCANNING'

    # === FINISHED: Shutdown after all work ===
    def handle_finished(self):
        self.stop()
        self.destroy_node()
        rclpy.shutdown()

    # Function to move the robot towards a target position
    def move_to_target(self, target):
        target_x, target_y = target
        robot_x, robot_y = self.position

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

        # If pushing phase is push forward, check if the box is close to the walls, instead of just the midpoint
        if self.phase == "PUSH_FORWARD":
            # Only check if the x or y coordinate is close
            if self.target_marker.wall_direction in ['North', 'South']:
                distance_to_target = abs(robot_y - target_y)
            elif self.target_marker.wall_direction in ['East', 'West']:
                distance_to_target = abs(robot_x - target_x)

            # If the box is close to the wall, stop
            if distance_to_target < 0.4:
                self.stop()
                return True
        else:
            # Check if the robot is close enough to the target
            if distance_to_target < 0.05:
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
