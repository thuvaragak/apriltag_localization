import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, PointCloud2, Image
import sensor_msgs.msg as sensor_msgs
import sensor_msgs_py.point_cloud2 as pc2
import std_msgs.msg
import numpy as np
import cv2
from cv_bridge import CvBridge
from pupil_apriltags import Detector
from nav_msgs.msg import Odometry

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import tf_transformations
from geometry_msgs.msg import Twist

class CameraRobotDistanceCalculator(Node):
    def __init__(self):
        super().__init__('camera_robot_distance_calculator')

        self.turn = 0
        self.prev_ang = 0.0
        self.count = 1 
        # Intrinsic calibration matrix
        self.K = []
        self.bridge = CvBridge()
        self.bbox = []
        self.img = None
        self.detector = Detector(families='tag36h11')
        self.latest_image = None
        self.latest_odom = None

        self.twist_sub = self.create_subscription(Twist, '/bcr_bot/cmd_vel', self.twist_callback, 10)

        # Subscribers
        self.camera = self.create_subscription(
            CameraInfo,
            '/bcr_bot/kinect_camera/camera_info',  # Topic name
            self.camera_info_callback,
            10
        )

        self.subscription = self.create_subscription(
            Image, '/bcr_bot/kinect_camera/image_raw', self.image_callback, 10)

        self.velodyne = self.create_subscription(
            PointCloud2,
            "/velodyne_points",
            self.lidar_callback,
            10,
        )

        # Publisher for transformed point cloud
        self.publisher = self.create_publisher(
            PointCloud2,
            "/transformed_lidar_points",
            10,
        )

        # Top-view image parameters
        self.world_size = (10.0, 20.0)  # Width and height of sensor field in meters
        self.image_size = (500, 1000)  # Corresponding top-view image in pixels

        # Calibration matrices
        self.P_rect_00 = np.zeros((3, 4), dtype=np.float64)  # Projection matrix
        self.R_rect_00 = np.zeros((4, 4), dtype=np.float64)  # Rectifying rotation matrix
        self.RT = np.zeros((4, 4), dtype=np.float64)  # Transformation matrix

        # Load calibration data
        self.load_calibration_data()

    def twist_callback(self, msg):
        # print("Velocity", msg.angular.z)
        if msg.angular.z != 0 and self.turn == 0:
            self.turn = 1
            self.count +=1
        elif msg.angular.z == 0 and self.turn != 0:
            self.turn = 0
            self.count +=1

    def load_calibration_data(self):
        """
        Load calibration data into the matrices.
        """
        # Projection matrix (P_rect_00): Intrinsic parameters for the Kinect camera
        self.P_rect_00[0, 0] = 554.26; self.P_rect_00[0, 1] = 0.0; self.P_rect_00[0, 2] = 320.0; self.P_rect_00[0, 3] = 0.0
        self.P_rect_00[1, 0] = 0.0; self.P_rect_00[1, 1] = 579.41; self.P_rect_00[1, 2] = 240.0; self.P_rect_00[1, 3] = 0.0
        self.P_rect_00[2, 0] = 0.0; self.P_rect_00[2, 1] = 0.0; self.P_rect_00[2, 2] = 1.0; self.P_rect_00[2, 3] = 0.0

        # Rectifying rotation matrix (R_rect_00): Identity matrix for monocular camera
        self.R_rect_00[0, 0] = 1.0; self.R_rect_00[0, 1] = 0.0; self.R_rect_00[0, 2] = 0.0; self.R_rect_00[0, 3] = 0.0
        self.R_rect_00[1, 0] = 0.0; self.R_rect_00[1, 1] = 1.0; self.R_rect_00[1, 2] = 0.0; self.R_rect_00[1, 3] = 0.0
        self.R_rect_00[2, 0] = 0.0; self.R_rect_00[2, 1] = 0.0; self.R_rect_00[2, 2] = 1.0; self.R_rect_00[2, 3] = 0.0
        self.R_rect_00[3, 0] = 0.0; self.R_rect_00[3, 1] = 0.0; self.R_rect_00[3, 2] = 0.0; self.R_rect_00[3, 3] = 1.0

        # Transformation matrix (RT): LiDAR to camera
        self.RT[0, 0] = -0.45; self.RT[0, 1] = 0.89; self.RT[0, 2] = 0.0; self.RT[0, 3] = 0.45
        self.RT[1, 0] = 0.0; self.RT[1, 1] = 0.0; self.RT[1, 2] = 0.2025; self.RT[1, 3] = 0
        self.RT[2, 0] = 0.801; self.RT[2, 1] = -0.5896; self.RT[2, 2] = 0.0; self.RT[2, 3] = -0.3729
        self.RT[3, 0] = 0.0; self.RT[3, 1] = 0.0; self.RT[3, 2] = 0.0; self.RT[3, 3] = 1.0

    def lidar_callback(self, msg):
        if self.latest_image is None:
            self.get_logger().warn("No image received yet!")
            return 

        # Extract LiDAR points
        points_list = np.array([
            [p[0], p[1], p[2]] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        ], dtype=np.float32)

        if points_list.shape[0] == 0:
            self.get_logger().info("No points received from Velodyne")
            return
        else:
            pass
            # self.get_logger().warn("Received LiDAR points")

        # Create top-view image
        topview_img = self.generate_topview(points_list)

        # Display the top-view image
        cv2.imshow("LiDAR Top-View", topview_img)
        cv2.waitKey(1)

        # Project LiDAR points onto the camera image
        self.project_lidar_to_camera2(points_list)

    def generate_topview(self, points_list):
        # Create a blank top-view image
        topview_img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)

        # Plot LiDAR points into the image
        for point in points_list:
            xw = point[0]  # World position in meters (x facing forward)
            yw = point[1]  # World position in meters (y facing left)

            # Convert world coordinates to image coordinates
            y = int(-xw * self.image_size[1] / self.world_size[1] + self.image_size[1])
            x = int(-yw * self.image_size[0] / self.world_size[0] + self.image_size[0] / 2)

            # Draw the point on the top-view image
            if 0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]:
                cv2.circle(topview_img, (x, y), 3, (0, 255, 0), -1)  # Draw green circles

        # Plot distance markers
        line_spacing = 2.0  # Gap between distance markers
        n_markers = int(self.world_size[1] // line_spacing)
        for i in range(n_markers):
            y = int(-(i * line_spacing) * self.image_size[1] / self.world_size[1] + self.image_size[1])
            cv2.line(topview_img, (0, y), (self.image_size[0], y), (255, 0, 0), 1)  # Draw blue lines

        return topview_img

    # def project_lidar_to_camera2(self, points_list):
    #     if self.latest_image is None:
    #         return

    #     # Create a copy of the image for visualization
    #     vis_img = self.latest_image.copy()
    #     overlay = vis_img.copy()

    #     # Remove points that will be erroneously projected onto the camera plane
    #     max_x = 25.0
    #     max_y = 6.0
    #     min_z = -1.4
    #     points_list_filtered = [point for point in points_list if
    #                            not (point[0] > max_x or point[0] < 0.0 or abs(point[1]) > max_y or point[2] < min_z)]

    #     # Project each LiDAR point onto the image plane
    #     for point in points_list_filtered:
    #         # Convert current Lidar point into homogeneous coordinates
    #         X = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64).reshape(4, 1)

    #         # Apply the projection equation: Y = P_rect_00 * R_rect_00 * RT * X
    #         Y = self.P_rect_00 @ self.R_rect_00 @ self.RT @ X

    #         # Convert Y back into Euclidean coordinates
    #         pt = (int(Y[0, 0] / Y[2, 0]), int(Y[1, 0] / Y[2, 0]))

    #         # Color coding based on distance
    #         val = point[0]
    #         max_val = 20.0
    #         red = min(255, int(255 * abs((val - max_val) / max_val)))
    #         green = min(255, int(255 * (1 - abs((val - max_val) / max_val))))

    #         if 0 <= pt[0] < self.latest_image.shape[1] and 0 <= pt[1] < self.latest_image.shape[0]:
    #             cv2.circle(overlay, pt, 5, (0, green, red), -1)

    #     # Blend the overlay with the original image
    #     opacity = 0.6
    #     cv2.addWeighted(overlay, opacity, vis_img, 1 - opacity, 0, vis_img)

    #     # Display the result
    #     cv2.imshow("LiDAR Points on Camera Image", vis_img)
    #     cv2.waitKey(1)

    # Filtering lidar inside BBOX
    # def project_lidar_to_camera2(self, points_list):
    #     if self.latest_image is None:
    #         return

    #     # Create a copy of the image for visualization
    #     vis_img = self.latest_image.copy()
    #     overlay = vis_img.copy()

    #     # Remove points that will be erroneously projected onto the camera plane
    #     max_x = 25.0
    #     max_y = 6.0
    #     min_z = -1.4
    #     points_list_filtered = [point for point in points_list if
    #                         not (point[0] > max_x or point[0] < 0.0 or abs(point[1]) > max_y or point[2] < min_z)]

    #     # Filter LiDAR points inside the AprilTag bounding box
    #     if len(self.bbox) == 4:  # Ensure bounding box coordinates are available
    #         x_min = min(self.bbox[0], self.bbox[2])
    #         x_max = max(self.bbox[0], self.bbox[2])
    #         y_min = min(self.bbox[1], self.bbox[3])
    #         y_max = max(self.bbox[1], self.bbox[3])

    #         filtered_points = []
    #         for point in points_list_filtered:
    #             # Convert current LiDAR point into homogeneous coordinates
    #             X = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64).reshape(4, 1)

    #             # Apply the projection equation: Y = P_rect_00 * R_rect_00 * RT * X
    #             Y = self.P_rect_00 @ self.R_rect_00 @ self.RT @ X

    #             # Convert Y back into Euclidean coordinates
    #             pt = (int(Y[0, 0] / Y[2, 0]), int(Y[1, 0] / Y[2, 0]))

    #             # Check if the projected point lies within the AprilTag bounding box
    #             if x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max:
    #                 filtered_points.append(point)

    #                 # Color coding based on distance
    #                 val = point[0]
    #                 max_val = 20.0
    #                 red = min(255, int(255 * abs((val - max_val) / max_val)))
    #                 green = min(255, int(255 * (1 - abs((val - max_val) / max_val))))

    #                 if 0 <= pt[0] < self.latest_image.shape[1] and 0 <= pt[1] < self.latest_image.shape[0]:
    #                     cv2.circle(overlay, pt, 5, (0, green, red), -1)

    #         # Log the number of filtered points
    #         self.get_logger().info(f"Number of LiDAR points inside AprilTag region: {len(filtered_points)}")

    #     # Blend the overlay with the original image
    #     opacity = 0.6
    #     cv2.addWeighted(overlay, opacity, vis_img, 1 - opacity, 0, vis_img)

    #     # Display the result
    #     cv2.imshow("LiDAR Points on Camera Image", vis_img)
    #     cv2.waitKey(1)
    
    # Calculate Euclidean distance
    def project_lidar_to_camera2(self, points_list):
        if self.latest_image is None:
            return

        # Create a copy of the image for visualization
        vis_img = self.latest_image.copy()
        overlay = vis_img.copy()

        # Remove points that will be erroneously projected onto the camera plane
        max_x = 10.0
        max_y = 10.0
        min_z = 0.0
        points_list_filtered = [point for point in points_list if
                            not (point[0] > max_x or point[0] < 0.0 or abs(point[1]) > max_y or point[2] < min_z)]

        # Filter LiDAR points inside the AprilTag bounding box
        if len(self.bbox) == 4:  # Ensure bounding box coordinates are available
            x_min = min(self.bbox[0], self.bbox[2])
            x_max = max(self.bbox[0], self.bbox[2])
            y_min = min(self.bbox[1], self.bbox[3])
            y_max = max(self.bbox[1], self.bbox[3])
            filtered_points = []

            for point in points_list_filtered:
                # Convert current LiDAR point into homogeneous coordinates
                X = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64).reshape(4, 1)

                # Apply the projection equation: Y = P_rect_00 * R_rect_00 * RT * X
                Y = self.P_rect_00 @ self.R_rect_00 @ self.RT @ X

                # Convert Y back into Euclidean coordinates
                pt = (int(Y[0, 0] / Y[2, 0]), int(Y[1, 0] / Y[2, 0]))

                # Check if the projected point lies within the AprilTag bounding box
                if x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max:
                    filtered_points.append(point)

                    # Color coding based on distance
                    val = point[0]
                    max_val = 20.0
                    red = min(255, int(255 * abs((val - max_val) / max_val)))
                    green = min(255, int(255 * (1 - abs((val - max_val) / max_val))))

                    if 0 <= pt[0] < self.latest_image.shape[1] and 0 <= pt[1] < self.latest_image.shape[0]:
                        cv2.circle(overlay, pt, 5, (0, green, red), -1)

            # Log the number of filtered points
            # self.get_logger().info(f"Number of LiDAR points inside AprilTag region: {len(filtered_points)}")

            # Calculate the minimum position (closest point) in X, Y, Z coordinates
            if len(filtered_points) > 0:
                # Find the point with the minimum Euclidean distance from the robot
                min_distance_point = min(filtered_points, key=lambda p: np.sqrt(p[0]**2 + p[1]**2 + p[2]**2))

                # Extract the X, Y, Z coordinates of the closest point
                min_x, min_y, min_z = min_distance_point

                # Log the minimum position (distance in X, Y, Z coordinates)
                if self.count%2 != 0:
                    dist_x = min_y
                    dist_y = min_x
                else:
                    dist_x = min_x
                    dist_y = min_y

                self.get_logger().info(f"AprilTag location respective to: (X, Y): ({dist_x:.2f}, {dist_y:.2f}) meters")

        # Blend the overlay with the original image
        opacity = 0.6
        cv2.addWeighted(overlay, opacity, vis_img, 1 - opacity, 0, vis_img)

        # Display the result
        cv2.imshow("LiDAR Points on Camera Image", vis_img)
        cv2.waitKey(1)

    def camera_info_callback(self, msg):
        self.K = np.array(msg.k).reshape(3, 3) 

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image = cv2.resize(self.latest_image, (500, 1000))

            # Detect AprilTags
            gray = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2GRAY)
            tags = self.detector.detect(gray)
            count = 0
            a = 0
            self.bbox = []
            for tag in tags:
                # Draw bounding box
                count += 1
                for i in range(4):
                    a += 1
                    p1 = (int(tag.corners[i][0]), int(tag.corners[i][1]))
                    p2 = (int(tag.corners[(i + 1) % 4][0]), int(tag.corners[(i + 1) % 4][1]))
                    if count == 1:
                        if a % 2 == 0:
                            self.bbox.append(tag.corners[i][1])
                        else:
                            self.bbox.append(tag.corners[i][0])
                    cv2.line(self.latest_image, p1, p2, (0, 255, 0), 2)

                # Draw center and print tag ID
                center = (int(tag.center[0]), int(tag.center[1]))
                cv2.circle(self.latest_image, center, 5, (0, 0, 255), -1)
                cv2.putText(self.latest_image, f"ID: {tag.tag_id}", (center[0] - 20, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraRobotDistanceCalculator()
    rclpy.spin(node)

if __name__ == '__main__':
    main()


