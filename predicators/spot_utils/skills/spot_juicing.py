"""Interface for spot dumping skill."""

import time

import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot

from predicators.spot_utils.skills.spot_hand_move import \
    move_hand_to_relative_pose, open_gripper, close_gripper, move_hand_to_relative_pose_with_velocity
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.skills.spot_place import place_at_relative_position
from predicators.spot_utils.skills.spot_hand_move import open_gripper


def close_juicer_lid(robot: Robot) -> None:
    """Close the lid of the Ecoself juicer.

    Assumes that the robot is lined up such that it's facing the
    juicer, and its arm is stowed.
    """
    # Pre-recorded joint angles for the arm to move through such that
    # it properly closes the lid of the juicer.
    pose0 = math_helpers.SE3Pose(
            x=0.7730178236961365,
            y=-0.02702152170240879,
            z=0.23073684692382812,
            rot=math_helpers.Quat(
                w=0.9922443628311157,
                x=-0.014565691351890564,
                y=0.116593137383461,
                z=-0.040559105575084686
            )
        )
    pose1 = math_helpers.SE3Pose(
                x=0.796045184135437,
                y=-0.028232159093022346,
                z=0.37765690207481384,
                rot=math_helpers.Quat(
                    w=0.9883142709732056,
                    x=-0.017500856891274452,
                    y=0.14659591019153595,
                    z=-0.037925124168395996
                )
            )
    pose2 = math_helpers.SE3Pose(
            x=0.9593419432640076,
            y=-0.030203117057681084,
            z=0.34951632738113403,
            rot=math_helpers.Quat(
                w=0.9974877834320068,
                x=-0.025175603106617928,
                y=0.0591338574886322,
                z=-0.02979166992008686
            )
        )
    ready_to_whack_lid_pose = math_helpers.SE3Pose(
            x=1.0326576232910156,
            y=-0.03433791548013687,
            z=0.4801491451263428,
            rot=math_helpers.Quat(
                w=0.9980011582374573,
                x=-0.023644166067242622,
                y=0.05118606984615326,
                z=-0.028539864346385002
            )
        )
    lid_whacking_down_pose = math_helpers.SE3Pose(
            x=1.0537371635437012,
            y=-0.03710928559303284,
            z=0.33800581693649292,
            rot=math_helpers.Quat(
                w=0.9979082345962524,
                x=-0.02833721600472927,
                y=0.049695659428834915,
                z=-0.030107460916042328
            )
        )
    # Finally, a pose to move the arm to look and
    # check if the lid is closed.
    lid_check_look_pose = math_helpers.SE3Pose(
        x=0.5151424407958984,
        y=-0.009533177129924297,
        z=0.5031274557113647,
        rot=math_helpers.Quat(
            w=0.8306255340576172,
            x=-0.009793010540306568,
            y=0.5553292036056519,
            z=-0.039684370160102844
        )
    )

    # Move the hand through the poses in a
    # trajectory that should closethe lid.
    move_hand_to_relative_pose(robot, pose0)
    time.sleep(0.1)
    move_hand_to_relative_pose(robot, pose1)
    time.sleep(0.1)
    move_hand_to_relative_pose(robot, pose2)
    time.sleep(0.1)
    # Whack the lid twice to make sure it's
    # locked and shut.
    for i in range(2):
        move_hand_to_relative_pose(robot, ready_to_whack_lid_pose)
        move_hand_to_relative_pose_with_velocity(robot, ready_to_whack_lid_pose, lid_whacking_down_pose,duration=0.1)
        time.sleep(0.1)
    # Finally, move the arm to a pose
    # where we can check if the lid is closed.
    move_hand_to_relative_pose(robot, lid_check_look_pose)
    time.sleep(0.1)


def turn_juicer_on(robot: Robot) -> None:
    """Turn on the Ecoself juicer.
    Assumes that the robot is lined up such that it's facing the
    juicer head-on and centered."""

    # Define the poses for the arm to move through.
    pre_grasp_pose = math_helpers.SE3Pose(
                x=0.7174609899520874,
                y=-0.004277821630239487,
                z=-0.04757143884897232,
                rot=math_helpers.Quat(
                    w=0.9982805252075195,
                    x=0.05800527706742287,
                    y=-0.00844526756554842,
                    z=-0.00026528292801231146
                )
            )
    grasp_pose = math_helpers.SE3Pose(
        x=0.809599199295044,
        y=-0.009432027116417885,
        z=-0.04757143884897232,
        rot=math_helpers.Quat(
            w=0.997207760810852,
            x=0.0580286830663681,
            y=-0.04694018512964249,
            z=-0.002454422414302826
        )
    )
    switch_rotated_pose = math_helpers.SE3Pose(
        x=0.809599199295044,
        y=-0.009432027116417885,
        z=-0.04757143884897232,
        rot=math_helpers.Quat(
            w=0.6473594903945923,
            x=0.760860502719879,
            y=-0.030967185273766518,
            z=0.032526180148124695
        )
    )
    juicer_look_pose = math_helpers.SE3Pose(
        x=0.5151424407958984,
        y=-0.009533177129924297,
        z=0.5031274557113647,
        rot=math_helpers.Quat(
            w=0.8306255340576172,
            x=-0.009793010540306568,
            y=0.5553292036056519,
            z=-0.039684370160102844
        )
    )
    # Now, move the arm through the motions.
    open_gripper(robot)
    move_hand_to_relative_pose(robot, pre_grasp_pose)
    time.sleep(0.1)
    move_hand_to_relative_pose(robot, grasp_pose)
    time.sleep(0.1)
    close_gripper(robot)
    time.sleep(0.1)
    move_hand_to_relative_pose(robot, switch_rotated_pose)
    time.sleep(0.1)
    open_gripper(robot)
    stow_arm(robot)
    move_hand_to_relative_pose(robot, juicer_look_pose)


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # NOTE: this test assumes that the robot is right in front of the juicer.

    # pylint: disable=ungrouped-imports
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient
    from bosdyn.client.util import authenticate

    from predicators import utils
    from predicators.settings import CFG
    from predicators.spot_utils.utils import verify_estop

    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)
        hostname = CFG.spot_robot_ip
        sdk = create_standard_sdk('DumpSkillTestClient')
        robot = sdk.create_robot(hostname)
        authenticate(robot)
        verify_estop(robot)
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        robot.time_sync.wait_for_sync()
        close_juicer_lid(robot)
        turn_juicer_on(robot)

    _run_manual_test()
