"""Special script for Andi CoRL experiments"""

from predicators import utils
from predicators.settings import CFG
from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME
from typing import List
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



def move_and_hand_reach_relative_transform(robot: Robot, localizer: SpotLocalizer,
    abs_hand_world_pose: math_helpers.SE3Pose, rng: np.random.Generator) -> None:
    """Attempts to get the hand to a specific pose in the world frame.
    If this is too far away to reach, the robot will sample a point
    closeby to move the body to, then move the hand."""
    # First, compute distance between current robot pose and desired
    # pose.
    # snapshot = robot.get_frame_tree_snapshot()
    # hand_in_body = get_a_tform_b(snapshot, BODY_FRAME_NAME, "hand")
    # localizer.localize()
    # curr_hand_pose = hand_in_body.mult(localizer.get_last_robot_pose())
    # dist_to_goal = np.linalg.norm(curr_hand_pose.get_translation() - abs_hand_world_pose.get_translation())
    # print(f"Dist to goal: {dist_to_goal}")
    # if dist_to_goal > 0.8:
        # localizer.localize()
        # desired_hand_pos = abs_hand_world_pose.get_translation()
        # distance, angle, _ = sample_move_offset_from_target(tuple([desired_hand_pos[0], desired_hand_pos[1]]), spot_pose_to_geom2d(localizer.get_last_robot_pose()), [], rng, 0.25, 0.35, get_allowed_map_regions())
        # print(f"Distance, angle: {(distance, angle)}")
        # relative_se2_move = get_relative_se2_from_se3(localizer.get_last_robot_pose(), abs_hand_world_pose, distance,
        #                                  angle)
        # relative_se2_move = get_relative_se2_from_se3(curr_hand_pose, abs_hand_world_pose, 0.0,
        #                                  0.0)
        # navigate_to_relative_pose(robot, relative_se2_move)
        # navigate_to_absolute_pose(robot, localizer, abs_hand_world_pose.get_closest_se2_transform())
        # time.sleep(0.5)
    localizer.localize()
    curr_robot_pose = localizer.get_last_robot_pose()
    desired_hand_in_body = abs_hand_world_pose.mult(curr_robot_pose.inverse())
    print(desired_hand_in_body)
    move_hand_to_relative_pose(robot, desired_hand_in_body)
    time.sleep(0.5)


    snapshot = robot.get_frame_tree_snapshot()
    hand_in_body = get_a_tform_b(snapshot, BODY_FRAME_NAME, "hand")
    localizer.localize()
    hand_in_world = hand_in_body.mult(localizer.get_last_robot_pose())
    # print(abs_hand_world_pose)
    # print(hand_in_world)



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
rng = np.random.default_rng(0)

assert path.exists()
# Creating a localizer so the robot knows its position in a map.
localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)
rgbds = capture_images(robot, localizer, TEST_CAMERAS)
language_ids: List[ObjectDetectionID] = [
    LanguageObjectDetectionID(d) for d in TEST_LANGUAGE_DESCRIPTIONS
]
# detections, artifacts = detect_objects(language_ids, rgbds)
snapshot = robot.get_frame_tree_snapshot()
hand_in_body = get_a_tform_b(snapshot, BODY_FRAME_NAME, "hand")
localizer.localize()
hand_in_world = hand_in_body.mult(localizer.get_last_robot_pose())

# for obj_id, detection in detections.items():
#     print(f"Detected {obj_id} at {detection}")
# print(f"Robot pose: {localizer.get_last_robot_pose()}")
# print(f"Hand pose: {hand_in_world}")

# TESTING.
test_hand_pose = math_helpers.SE3Pose(2.691, -0.848, 0.509, math_helpers.Quat(-0.0994, 0.0791, -0.0041, 0.9919))
move_and_hand_reach_relative_transform(robot, localizer, test_hand_pose, rng)
# move_and_hand_reach_relative_transform(robot, localizer, hand_in_world, rng)

# Probably easier to just move the body to some absolute pose in the world, and then move the hand to a relative pose....
# TODO: sample from map.

# Simple reward function example.
def reward_function(proposed_pose: math_helpers.SE2Pose) -> float:
    spot_home = get_spot_home_pose()
    return 1/np.linalg.norm(np.array([spot_home.x, spot_home.y]) - np.array([proposed_pose.x, proposed_pose.y]))
