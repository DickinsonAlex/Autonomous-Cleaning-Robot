# Start Simulation:
1. Create a new terminal
2. run: `ros2 launch uol_tidybot tidybot.launch.py`
3. Optionally run: `ros2 run  uol_tidybot generate_objects --ros-args -p red:=true -p n_objects:=3` to spawn 3 red boxes
4. Then simply run main.py


# About my project
1. It sorts all the cubes by color
2. It has basic collision avoidance, but more advanced environments can cause it to get stuck
