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
from predicators.spot_utils.perception.spot_cameras import capture_images, get_last_captured_images
from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.utils import verify_estop, get_pixel_from_user
from predicators.spot_utils.perception.object_detection import detect_objects, get_grasp_pixel, visualize_all_artifacts, get_last_detected_objects
from predicators.spot_utils.skills.spot_hand_move import move_hand_to_relative_pose, open_gripper
from predicators.spot_utils.utils import get_allowed_map_regions, \
    sample_move_offset_from_target, spot_pose_to_geom2d
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.skills.spot_navigation import navigate_to_absolute_pose
from predicators.spot_utils.skills.spot_grasp import grasp_at_pixel
from predicators.spot_utils.skills.spot_place import place_at_relative_position
from predicators.spot_utils.utils import get_spot_home_pose


TEST_CAMERAS = [
        "hand_color_image",
        "frontleft_fisheye_image",
        "left_fisheye_image",
        "right_fisheye_image",
        "frontright_fisheye_image",
    ]
# DETECTION OBJECTS
TEST_LANGUAGE_DESCRIPTIONS = [
        "plant",
        "elephant watering can",
        "Spam can",
        "green apple",
        "green block",
        "book"
    ]

# HELPFUL CONSTANTS
DOWNWARD_ANGLE = np.pi / 2.5
DOWNWARD_ARM_ROT = math_helpers.Quat.from_pitch(DOWNWARD_ANGLE)
DOWNWARD_ARM_POSE = math_helpers.SE3Pose(x=0.8, y=0.0, z=0.25, rot=DOWNWARD_ARM_ROT)

HAND_CAMERA = "hand_color_image"

LANGUAGE_IDS: List[ObjectDetectionID] = [LanguageObjectDetectionID(d) for d in TEST_LANGUAGE_DESCRIPTIONS]

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

# RESET ROBOT
def reset_robot(robot: Robot, localizer: SpotLocalizer) -> None:
    stow_arm(robot)
    navigate_to_absolute_pose(robot, localizer, math_helpers.SE2Pose(3.943, 0.074, math.radians(-163.3)))
    time.sleep(0.5)

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
    pouring_rot = init_rot * math_helpers.Quat.from_yaw(np.pi / 2.0) # rotation for pour (if need greater rotation, make angle of pour higher)
    pouring_pose = math_helpers.SE3Pose(x=rel_pos.x,
                                           y=rel_pos.y,
                                           z=rel_pos.z,
                                           rot=pouring_rot)
    move_hand_to_relative_pose(robot, pouring_pose)
    time.sleep(1.5)
    move_hand_to_relative_pose(robot, init_pose)

# Simple reward function example.
def reward_function(input_traj: List[Tuple[float, float]]) -> float:
    assert len(input_traj) == 3
    desired_trajectory = [(3.5, 0.45), (3.0, 0.45), (3.0, 0.0)]
    reward = 0.0
    for i, waypoint in enumerate(input_traj):
        reward += -np.linalg.norm(np.array([desired_trajectory[i][0], desired_trajectory[i][1]]) - np.array([waypoint[0], waypoint[1]]))
    return reward

# Example sampler.
# spot_home_pose = get_spot_home_pose()
# max_reward = -np.inf
# max_reward_traj = None
rng = np.random.default_rng(0)
#reset_robot(robot, localizer)
rgbds = capture_images(robot, localizer, TEST_CAMERAS)

# PICK UP CAN SEQUENCE
#detections, artifacts = detect_objects(LANGUAGE_IDS, rgbds)
reset_robot(robot, localizer)
navigate_to_absolute_pose(robot, localizer, math_helpers.SE2Pose(3.629, 0.361, math.radians(164.7)))
time.sleep(0.5)
move_hand_to_relative_pose(robot, DOWNWARD_ARM_POSE)
time.sleep(0.5)
rgbds = capture_images(robot, localizer, TEST_CAMERAS)
#pixel = get_pixel_from_user(rgbds[HAND_CAMERA].rgb)
#target_detection_id = "elephant watering can"
#_, artifacts = get_last_detected_objects()
#pixel, _ = get_grasp_pixel(rgbds, artifacts, target_detection_id, HAND_CAMERA, rng)
grasp_at_pixel(robot, rgbds[HAND_CAMERA], (459, 306))
time.sleep(0.5)
move_hand_to_relative_pose(robot, math_helpers.SE3Pose(x=0.858, y=-0.075, z=0.394, rot=DOWNWARD_ARM_ROT))
time.sleep(0.5)
navigate_to_absolute_pose(robot, localizer, math_helpers.SE2Pose(3.069, -0.031, math.radians(-136.8)))

# WATER PLANT SEQUENCE
# reset_robot(robot, localizer)
# navigate_to_absolute_pose(robot, localizer, math_helpers.SE2Pose(3.295, 0.082, math.radians(-139.1)))
# time.sleep(0.5)
# move_hand_to_relative_pose(robot, DOWNWARD_ARM_POSE)
# rgbds = capture_images(robot, localizer, TEST_CAMERAS)
# #pixel = get_pixel_from_user(rgbds[HAND_CAMERA].rgb)
# #print(pixel)
# time.sleep(0.5)
# grasp_at_pixel(robot, rgbds[HAND_CAMERA], (184, 207))
# time.sleep(0.5)
# move_hand_to_relative_pose(robot, math_helpers.SE3Pose(x=0.497, y=0.461, z=0.740, rot=DOWNWARD_ARM_ROT))
# time.sleep(0.5)
# navigate_to_absolute_pose(robot, localizer, math_helpers.SE2Pose(2.879, -0.166, math.radians(-167.1))) # sampled location 
# time.sleep(0.5)
# move_hand_to_relative_pose(robot, math_helpers.SE3Pose(0.814, 0.015, 0.887, math_helpers.Quat(0.7648, -0.0013, 0.6442, -0.0044)))
# time.sleep(0.5)
# pour_at_relative_pose(robot, math_helpers.Vec3(0.814, 0.015, 0.887))

# WATER PLANT (FULL) SEQUENCE
# stow_arm(robot)
# time.sleep(0.5)
# navigate_to_absolute_pose(robot, localizer, math_helpers.SE2Pose(2.163, 0.360, math.radians(-74.0)))
# time.sleep(0.5)
# navigate_to_absolute_pose(robot, localizer, math_helpers.SE2Pose(3.271, -0.077, math.radians(-78.6)))
# time.sleep(0.5)
# rgbds = capture_images(robot, localizer, TEST_CAMERAS)
# hand_camera = "hand_color_image"
# pixel = get_pixel_from_user(rgbds[hand_camera].rgb)
# grasp_at_pixel(robot, rgbds[hand_camera], pixel)
# time.sleep(0.5)
# #move_hand_to_relative_pose(robot, math_helpers.SE3Pose(0.753, 0.018, 0.89, math_helpers.Quat(0.6870, 0.0664, 0.7200, -0.0722)))
# move_hand_to_relative_pose(robot, math_helpers.SE3Pose(0.834, 0.103, 0.565, math_helpers.Quat(0.6142, -0.0020, 0.7859, 0.0716)))
# time.sleep(0.5)
# navigate_to_absolute_pose(robot, localizer, math_helpers.SE2Pose(2.808, -0.908, math.radians(-78.6)))
# time.sleep(0.5)
# pour_at_relative_pose(robot, math_helpers.Vec3(1.086, 0.083, 0.481))
# # move_hand_to_relative_pose(robot, math_helpers.SE3Pose(0.833, 0.150, 0.499, math_helpers.Quat(0.6156, -0.4840, 0.4909, 0.3817)))
# # time.sleep(0.5)
# # move_hand_to_relative_pose(robot, math_helpers.SE3Pose(0.644, -0.325, 0.250, math_helpers.Quat(0.7085, -0.2046, 0.6750, 0.0231)))

# THROW AWAY RAG SEQUENCE
#reset_robot(robot, localizer)
# time.sleep(0.5)
# navigate_to_absolute_pose(robot, localizer, math_helpers.SE2Pose(3.350, 0.060, math.radians(-159.1)))
# time.sleep(0.5)
# DOWNWARD_ANGLE = np.pi / 2.5
# move_hand_to_relative_pose(robot, math_helpers.SE3Pose(x=0.629, y=-0.341, z=0.500, rot=DOWNWARD_ARM_ROT))
# time.sleep(0.5)
# move_hand_to_relative_pose(robot, math_helpers.SE3Pose(x=0.630, y=0.028, z=0.568, rot=DOWNWARD_ARM_ROT))
# time.sleep(0.5)
# rgbds = capture_images(robot, localizer, TEST_CAMERAS)
# hand_camera = "hand_color_image"
# pixel = get_pixel_from_user(rgbds[hand_camera].rgb)
# grasp_at_pixel(robot, rgbds[hand_camera], pixel)
# move_hand_to_relative_pose(robot, math_helpers.SE3Pose(x=0.858, y=-0.075, z=0.494, rot=DOWNWARD_ARM_ROT))
# time.sleep(0.5)
# # move_hand_to_relative_pose(robot, math_helpers.SE3Pose(x=0.530, y=-0.004, z=0.651, rot=DOWNWARD_ARM_ROT))
# # time.sleep(0.5)
# move_hand_to_relative_pose(robot, math_helpers.SE3Pose(x=0.588, y=0.605, z=0.462, rot=DOWNWARD_ARM_ROT))
# open_gripper(robot)
# time.sleep(0.5) 