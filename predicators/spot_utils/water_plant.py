"""Special script for Andi CoRL experiments"""

from predicators import utils
from predicators.settings import CFG
from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME
from typing import List, Tuple
import numpy as np
from pathlib import Path

from predicators.settings import CFG
from predicators.spot_utils.perception.perception_structs import \
    LanguageObjectDetectionID, ObjectDetectionID
from predicators.spot_utils.utils import get_graph_nav_dir
from bosdyn.client import create_standard_sdk
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.util import authenticate
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot
import time
import math

from predicators import utils
from predicators.spot_utils.perception.spot_cameras import capture_images
from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.utils import verify_estop, get_pixel_from_user
from predicators.spot_utils.perception.object_detection import detect_objects, get_grasp_pixel, visualize_all_artifacts
from predicators.spot_utils.skills.spot_hand_move import move_hand_to_relative_pose
from predicators.spot_utils.utils import get_allowed_map_regions, \
    sample_move_offset_from_target, spot_pose_to_geom2d
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.skills.spot_navigation import navigate_to_absolute_pose
from predicators.spot_utils.skills.spot_grasp import grasp_at_pixel
from predicators.spot_utils.utils import get_spot_home_pose

TEST_CAMERAS = [
        "hand_color_image",
        "frontleft_fisheye_image",
        "left_fisheye_image",
        "right_fisheye_image",
        "frontright_fisheye_image",
    ]
TEST_LANGUAGE_DESCRIPTIONS = [
    "potted plant",
    "green apple/tennis ball",
    "elephant watering can"
]

args = utils.parse_args(env_required=False,
                        seed_required=False,
                        approach_required=False)
utils.update_config(args)

# Get constants.
hostname = CFG.spot_robot_ip
path = get_graph_nav_dir()
# Setup.
sdk = create_standard_sdk('SpotCameraTestClient')
robot = sdk.create_robot(hostname)
authenticate(robot)
verify_estop(robot)
lease_client = robot.ensure_client(LeaseClient.default_service_name)
lease_client.take()
lease_client = robot.ensure_client(LeaseClient.default_service_name)
lease_client.take()
lease_keepalive = LeaseKeepAlive(lease_client,
                                    must_acquire=True,
                                    return_at_exit=True)

assert path.exists()
# Creating a localizer so the robot knows its position in a map.
localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)
localizer.localize()
robot_pose = localizer.get_last_robot_pose()

# Pouring skill.
def pour_at_relative_pose(robot: Robot, rel_pos: math_helpers.Vec3) -> None:
    """Assuming the robot is holding something, execute a pour."""
    # First move the hand to the target pose.
    init_rot = math_helpers.Quat.from_pitch(np.pi/2)
    init_pose = math_helpers.SE3Pose(x=rel_pos.x,
                                           y=rel_pos.y,
                                           z=rel_pos.z,
                                           rot=init_rot)
    move_hand_to_relative_pose(robot, init_pose)
    time.sleep(0.5)
    pouring_rot = init_rot * math_helpers.Quat.from_yaw(np.pi / 4.0)
    pouring_pose = math_helpers.SE3Pose(x=rel_pos.x,
                                           y=rel_pos.y,
                                           z=rel_pos.z,
                                           rot=pouring_rot)
    move_hand_to_relative_pose(robot, pouring_pose)
    time.sleep(0.5)
    move_hand_to_relative_pose(robot, init_pose)

# Simple reward function example.
def reward_function(input_traj: List[Tuple[float, float]]) -> float:
    assert len(input_traj) == 3
    desired_trajectory = [(3.5, 0.45), (3.0, 0.45), (3.0, 0.0)]
    reward = 0.0
    for i, waypoint in enumerate(input_traj):
        reward += -np.linalg.norm(np.array([desired_trajectory[i][0], desired_trajectory[i][1]]) - np.array([waypoint[0], waypoint[1]]))
    return reward

# ANDI to modify
#def reward_network(input_traj: List[Tuple[float, float]]) -> float:
#    self.cost_nn = MLP(input_dim, 1, [], output_activation=None).to(self.device)
#    self.optimizer = optim.Adam(self.cost_nn.parameters(), lr=params["lr"])

#def calc_cost(self, traj):
#    traj = torch.tensor(traj, dtype=torch.float32).to(self.device)
#    return self.cost_nn(traj).item()

# Example sampler.
# spot_home_pose = get_spot_home_pose()
# max_reward = -np.inf
# max_reward_traj = None
rng = np.random.default_rng(0)
# for _ in range(10000):
#     curr_traj = []
#     for _ in range(3):
#         distance, angle, _ = sample_move_offset_from_target((spot_home_pose.x, spot_home_pose.y), spot_pose_to_geom2d(robot_pose), [], rng, 0.0, 2.5, get_allowed_map_regions())
#         x, y = spot_home_pose.x + np.cos(angle) * distance, spot_home_pose.y + np.sin(angle) * distance
#         curr_traj.append((x, y))
#     if reward_function(curr_traj) > max_reward:
#         max_reward = reward_function(curr_traj)
#         max_reward_traj = curr_traj
#     # ANDI TO MODIFY + ADD
#     # if calc_cost(curr_traj) < max_cost:
#     #    min_cost = calc_cost(curr_traj)
#     #    min_cost_traj = curr_traj
# assert max_reward_traj is not None

# print(max_reward_traj)

# for waypoint in max_reward_traj:
#     navigate_to_absolute_pose(robot, localizer, math_helpers.SE2Pose(waypoint[0], waypoint[1], 0.0))
#     time.sleep(0.5)

# stow_arm(robot)
# time.sleep(0.5)
# navigate_to_absolute_pose(robot, localizer, math_helpers.SE2Pose(2.163, 0.360, math.radians(-74.0)))
# time.sleep(0.5)
# navigate_to_absolute_pose(robot, localizer, math_helpers.SE2Pose(3.271, -0.077, math.radians(-78.6)))
# time.sleep(0.5)
# downward_angle = np.pi / 2.5
# target_pos = math_helpers.Vec3(0.8, 0.0, 0.25)
# rot = math_helpers.Quat.from_pitch(downward_angle)
# body_tform_goal = math_helpers.SE3Pose(x=target_pos.x, y=target_pos.y, z=target_pos.z, rot=rot)
# move_hand_to_relative_pose(robot, body_tform_goal)
# time.sleep(0.5)
# rgbds = capture_images(robot, localizer, TEST_CAMERAS)
# hand_camera = "hand_color_image"
# pixel = get_pixel_from_user(rgbds[hand_camera].rgb)
# grasp_at_pixel(robot, rgbds[hand_camera], pixel)
# time.sleep(0.5)
# #move_hand_to_relative_pose(robot, math_helpers.SE3Pose(0.753, 0.018, 0.89, math_helpers.Quat(0.6870, 0.0664, 0.7200, -0.0722)))
# move_hand_to_relative_pose(robot, math_helpers.SE3Pose(0.644, -0.008, 0.734, math_helpers.Quat(0.8861, 0.1572, 0.4129, -0.1398)))
# time.sleep(0.5)
# navigate_to_absolute_pose(robot, localizer, math_helpers.SE2Pose(2.808, -0.908, math.radians(-78.6)))
# time.sleep(0.5)
# pour_at_relative_pose(robot, math_helpers.Vec3(1.086, 0.083, 0.481))
move_hand_to_relative_pose(robot, math_helpers.SE3Pose(0.833, 0.150, 0.499, math_helpers.Quat(0.6156, -0.4840, 0.4909, 0.3817)))
time.sleep(0.5)
move_hand_to_relative_pose(robot, math_helpers.SE3Pose(0.644, -0.325, 0.250, math_helpers.Quat(0.7085, -0.2046, 0.6750, 0.0231)))
