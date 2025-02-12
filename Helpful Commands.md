# Tips: 
1. When first opening the project in vscode, reopen project in DevContainers
2. Store workshops in folders under src. Run files via arrow, no commands needed

# Start Simulation:
1. Create a new terminal
2. run: `ros2 launch uol_tidybot tidybot.launch.py`

# Control Robot:
1. Create a new terminal (Do not stop the other terminals)
2. run: `ros2 run teleop_twist_keyboard teleop_twist_keyboard`

# Packaging:
1. Once code is complete, run:
`ros2 pkg create --build-type ament_python PACKAGENAME`
2. Drag folder under relevant workshop folder
3. Drag python scripts under **PACKAGENAME** folder

HELP: https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html