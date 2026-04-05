"""Standalone test: object search + navigate to table + wipe.

Skips the full planning pipeline. Useful for iterating on the
mapless wipe sampler and skill.

Usage:
    python scripts/test_mapless_wipe.py \
        --spot_robot_ip 192.168.80.3 \
        --spot_mapless_mode True \
        --spot_run_dry False
"""
import logging
import numpy as np

from bosdyn.client import math_helpers

from predicators import utils
from predicators.settings import CFG
from predicators.envs.spot_env import get_robot
from predicators.spot_utils.spot_localization import OdomLocalizer
from predicators.spot_utils.perception.object_detection import detect_objects
from predicators.spot_utils.perception.perception_structs import (
    KnownStaticObjectDetectionID,
    LanguageObjectDetectionID,
)
from predicators.spot_utils.skills.spot_find_objects import (
    init_search_for_objects,
)
from predicators.spot_utils.skills.spot_navigation import (
    navigate_to_absolute_pose,
)
from predicators.spot_utils.skills.spot_hand_move import (
    move_hand_to_relative_pose,
)

logging.basicConfig(level=logging.INFO)


def main() -> None:
    # Parse args and update config.
    args = utils.parse_args(env_required=False, seed_required=False,
                            approach_required=False)
    utils.update_config(args)
    assert CFG.spot_mapless_mode, "This script requires --spot_mapless_mode True"

    # Connect to robot.
    robot, localizer, lease_client = get_robot()
    assert robot is not None and localizer is not None
    print("[1/4] Connected to robot.")

    # Set up detection IDs (same as the wiping env).
    table_det_id = LanguageObjectDetectionID("childs_play_table")
    eraser_det_id = LanguageObjectDetectionID("green_and_blue_furry_eraser")
    search_ids = {table_det_id, eraser_det_id}

    # Move hand to a good search pose.
    hand_pose = math_helpers.SE3Pose(
        x=0.80, y=0.0, z=0.75,
        rot=math_helpers.Quat.from_pitch(np.pi / 3))
    move_hand_to_relative_pose(robot, hand_pose)

    # Run object search (spin in place).
    print("[2/4] Searching for objects...")
    detections, artifacts = init_search_for_objects(
        robot, localizer, search_ids)

    # Print what we found.
    for det_id, pose in detections.items():
        print(f"  Found {det_id}: ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f})")

    if table_det_id not in detections:
        print("ERROR: Table not detected! Exiting.")
        return

    # Compute approach pose (same logic as the sampler).
    localizer.localize()
    robot_pose = localizer.get_last_robot_pose()
    robot_x = robot_pose.x
    robot_y = robot_pose.y
    table_pose = detections[table_det_id]
    table_x = table_pose.x
    table_y = table_pose.y

    angle_table_to_robot = np.arctan2(robot_y - table_y,
                                      robot_x - table_x)
    standoff = 0.85
    stand_x = table_x + standoff * np.cos(angle_table_to_robot)
    stand_y = table_y + standoff * np.sin(angle_table_to_robot)
    facing_angle = angle_table_to_robot + np.pi

    print(f"  Table at:  ({table_x:.3f}, {table_y:.3f})")
    print(f"  Robot at:  ({robot_x:.3f}, {robot_y:.3f})")
    print(f"  Will navigate to: ({stand_x:.3f}, {stand_y:.3f}, "
          f"angle={np.degrees(facing_angle):.1f}°)")

    input("\nPress Enter to navigate to the table approach pose...")

    # Navigate.
    target_se2 = math_helpers.SE2Pose(stand_x, stand_y, facing_angle)
    print("[3/4] Navigating to approach pose...")
    navigate_to_absolute_pose(robot, localizer, target_se2)
    print("  Arrived.")

    input("\nPress Enter to run the wipe skill...")

    # Wipe (using iPhone-based online wiping).
    print("[4/4] Running wipe skill...")
    from predicators.spot_utils.skills.spot_wipe_online_iphone import \
        wipe_online
    wipe_online(robot)
    print("Done!")


if __name__ == "__main__":
    main()
