# 🚀 Camera–3D LiDAR Sensor Fusion to Estimate Distance Between Robot & AprilTag in ROS 2 & Gazebo!

1. Install ROS2 gazebo & apriltag dependencies and clone https://github.com/blackcoffeerobotics/bcr_bot in your ros2 workspace.

2. Clone this repo and replace the world & bcr_bot urdf files with respect to the original files in the bcr_bot package  & build it. 

3. Launch the robot and run:
```
ros2 run fusing_sensors robot_tag_dist 
```
## Solution Approach:
 🔹 Visualized the 3D LiDAR point cloud from the Velodyne sensor in a top-down view using Gazebo to better understand the spatial distribution of objects.
 🔹 Projected LiDAR points into the camera frame using homogeneous coordinate transformation and camera intrinsics/extrinsics.
   ➤ Projection equation: Y = P_rect_00 * R_rect_00 * RT * X
    • P_rect_00: Camera intrinsic matrix
    • R_rect_00: Rectification matrix (identity)
    • RT: Transformation from LiDAR to camera
    • X: LiDAR point in homogeneous coordinates
 🔹 Detected AprilTags in the image using the pupil_apriltag Python library and bound them.
 🔹 Identified LiDAR points falling within the AprilTag’s bounding box and calculated the closest distance in (x, y) coordinates.
 🔹 Used Euclidean distance to determine the nearest point to the robot.
 🔹 Overlaid results on the camera image for visual interpretation with color-coded feedback.

 ##### It is part of my robotics perception project focused on AprilTag-based localization using ROS 2, OpenCV, and LiDAR–Camera calibration.

###### If any doubts, contact me at thuvaraga.krishnarajah@gmail.com


