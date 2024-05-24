"""Special script for Andi CoRL experiments"""

from predicators import utils
from predicators.settings import CFG
from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME
from typing import List, Tuple
import numpy as np

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

from predicators import utils
from predicators.spot_utils.perception.spot_cameras import capture_images
from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.utils import verify_estop
from predicators.spot_utils.perception.object_detection import detect_objects
from predicators.spot_utils.skills.spot_hand_move import move_hand_to_relative_pose
from predicators.spot_utils.utils import get_allowed_map_regions, \
    get_collision_geoms_for_nav, load_spot_metadata, object_to_top_down_geom, \
    sample_move_offset_from_target, spot_pose_to_geom2d, get_relative_se2_from_se3
from predicators.spot_utils.skills.spot_navigation import navigate_to_relative_pose, go_home, navigate_to_absolute_pose
from predicators.spot_utils.utils import get_robot_state, get_spot_home_pose



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

# Simple reward function example.
def reward_function(proposed_pose: math_helpers.SE2Pose) -> float:
    spot_home = get_spot_home_pose()
    return 1/np.linalg.norm(np.array([spot_home.x, spot_home.y]) - np.array([proposed_pose.x, proposed_pose.y]))

# Example sampler.
spot_home_pose = get_spot_home_pose()
max_reward = -np.inf
max_reward_pose = None
rng = np.random.default_rng(0)
for _ in range(1000):    
    distance, angle, _ = sample_move_offset_from_target((spot_home_pose.x, spot_home_pose.y), spot_pose_to_geom2d(robot_pose), [], rng, 0.0, 2.5, get_allowed_map_regions())
    x, y = spot_home_pose.x + np.cos(angle) * distance, spot_home_pose.y + np.sin(angle) * distance
    candidate_pose = math_helpers.SE2Pose(x, y, 0.0)
    if reward_function(candidate_pose) > max_reward:
        max_reward = reward_function(candidate_pose)
        max_reward_pose = candidate_pose
assert max_reward_pose is not None
print(max_reward_pose)
print(spot_home_pose)
navigate_to_absolute_pose(robot, localizer, max_reward_pose)

