# Tips: 
1. When first opening the project in vscode, reopen project in DevContainers
2. Store workshops in folders under src. Run files via arrow, no commands needed
3. You can run `ros2 topic list` to see all the nodes

# Start Simulation:
1. Create a new terminal
2. run: `ros2 launch uol_tidybot tidybot.launch.py`
2. or a more complex environment preset: `ros2 launch uol_tidybot tidybot.launch.py world:=level_2_3.world`

# Commands for altering the environment
1. Spawn a ton of red boxes: `ros2 run  uol_tidybot generate_objects --ros-args -p red:=true -p n_objects:=10`

# Control Robot:
1. Create a new terminal (Do not stop the other terminals)
2. run: `ros2 run teleop_twist_keyboard teleop_twist_keyboard`

# Packaging:
1. Once code is complete, run:
`ros2 pkg create --build-type ament_python PACKAGENAME`
2. Drag folder under relevant workshop folder
3. Drag python scripts under **PACKAGENAME** folder

HELP: https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html

# Other:
1. Use `ros2 topic echo /limo/depth_camera_link/camera_info` to find the camera info