"""Interface for spot sweeping skill."""

import time
from typing import Tuple

import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot

from predicators.spot_utils.skills.spot_hand_move import \
    move_hand_to_relative_pose, move_hand_to_relative_pose_with_velocity


def wipe_one_stroke(robot: Robot, wipe_start_pose: math_helpers.SE3Pose,
                    move_dx: float, move_dy: float, duration: float) -> None:
    """Wipe a table surface in the xy plane.

    The robot starts at a start pose, and then moves forward and back by
    dx and dy.
    """
    # Move first in the yz direction (perpendicular to robot body) to avoid
    # knocking the target object over.
    move_hand_to_relative_pose(robot, wipe_start_pose)
    first_move_pose = math_helpers.SE3Pose(
        x=wipe_start_pose.x + move_dx,  # sensible default
        y=wipe_start_pose.y + move_dy,
        z=wipe_start_pose.z,
        rot=wipe_start_pose.rot,
    )
    move_hand_to_relative_pose_with_velocity(robot, wipe_start_pose,
                                             first_move_pose, duration)
    time.sleep(0.1)
    # Move back to the start pose.
    move_hand_to_relative_pose_with_velocity(robot, first_move_pose,
                                             wipe_start_pose, duration)


def wipe_multiple_strokes(robot: Robot, wipe_start_pose: math_helpers.SE3Pose,
                          end_look_pose: math_helpers.SE3Pose,
                          stroke_dx: float, stroke_dy: float,
                          delta_x_y_between_strokes: Tuple[float, float],
                          num_strokes: int,
                          duration_per_stroke: float) -> None:
    """Wipe a table surface in the xy plane.

    The robot starts at a start pose, and then moves forward and back by
    dx and dy.
    """
    curr_stroke_start_pose = wipe_start_pose
    for i in range(num_strokes):
        move_hand_to_relative_pose(robot, curr_stroke_start_pose)
        first_move_pose = math_helpers.SE3Pose(
            x=curr_stroke_start_pose.x + stroke_dx,
            y=curr_stroke_start_pose.y + stroke_dy,
            z=curr_stroke_start_pose.z,
            rot=curr_stroke_start_pose.rot,
        )
        move_hand_to_relative_pose_with_velocity(robot, curr_stroke_start_pose,
                                                 first_move_pose,
                                                 duration_per_stroke)
        time.sleep(0.1)
        # Move back to the start pose.
        move_hand_to_relative_pose_with_velocity(robot, first_move_pose,
                                                 curr_stroke_start_pose,
                                                 duration_per_stroke)
        # Move to the next stroke position.
        curr_stroke_start_pose = math_helpers.SE3Pose(
            x=curr_stroke_start_pose.x + delta_x_y_between_strokes[0],
            y=curr_stroke_start_pose.y + delta_x_y_between_strokes[1],
            z=curr_stroke_start_pose.z,
            rot=curr_stroke_start_pose.rot,
        )
    # Move to the end look pose.
    move_hand_to_relative_pose(robot, end_look_pose)


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # NOTE: this test assumes that the robot is standing in front of a table
    # that has a train_toy on it. The test starts by running object detection to
    # get the pose of the train_toy. Then the robot opens its gripper and pauses
    # until a brush is put in the gripper, with the bristles facing down and
    # forward. The robot should then brush the train_toy to the right.

    # pylint: disable=ungrouped-imports
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.util import authenticate

    from predicators import utils
    from predicators.settings import CFG
    from predicators.spot_utils.perception.perception_structs import \
        LanguageObjectDetectionID
    from predicators.spot_utils.skills.spot_find_objects import \
        init_search_for_objects
    from predicators.spot_utils.skills.spot_hand_move import close_gripper, \
        open_gripper
    from predicators.spot_utils.skills.spot_navigation import go_home
    from predicators.spot_utils.spot_localization import SpotLocalizer
    from predicators.spot_utils.utils import get_graph_nav_dir, \
        get_relative_se2_from_se3, verify_estop

    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)

        # Get constants.
        hostname = CFG.spot_robot_ip

        sdk = create_standard_sdk('WipeSkillTestClient')
        robot = sdk.create_robot(hostname)
        authenticate(robot)
        verify_estop(robot)
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        robot.time_sync.wait_for_sync()

        # Move the hand to the side.
        hand_side_pose = math_helpers.SE3Pose(x=0.80,
                                              y=0.0,
                                              z=0.25,
                                              rot=math_helpers.Quat.from_yaw(
                                                  -np.pi / 2))
        move_hand_to_relative_pose(robot, hand_side_pose)
        # Ask for the eraser.
        open_gripper(robot)
        # Press any key, instead of just enter. Useful for remote control.
        msg = "Put the brush in the robot's gripper, then press any key"
        utils.wait_for_any_button_press(msg)
        close_gripper(robot)

        # NOTE: these parameters hardcoded for a particular child_play_table
        # object njk is experimenting with. Please swap out depending on the
        # actual object you have
        start_pose = math_helpers.SE3Pose(x=0.85,
                                          y=-0.2,
                                          z=-0.08,
                                          rot=math_helpers.Quat.from_pitch(
                                              np.pi / 2))
        end_pose = math_helpers.SE3Pose(x=0.65,
                                        y=0.0,
                                        z=0.4,
                                        rot=math_helpers.Quat.from_pitch(
                                            np.pi / 2.5))
        # Execute the sweep.
        wipe_multiple_strokes(robot, start_pose, end_pose, 0.0, 0.4,
                              (0.05, 0.0), 5, 1.0)

    _run_manual_test()