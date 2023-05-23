"""Simple Spot ROS2 Example in Predicators Repo."""

import rclpy
from simple_walk_forward.walk_forward import WalkForward

rclpy.init()
goto = WalkForward()
goto.initialize_robot()
goto.walk_forward_with_world_frame_goal()
goto.shutdown()
