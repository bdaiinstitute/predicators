"""Interface for spot dumping skill."""

import time

import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot

from predicators.spot_utils.skills.spot_hand_move import close_gripper, \
    move_hand_to_relative_pose, move_hand_to_relative_pose_with_velocity, \
    open_gripper, get_end_effector_state
from predicators.spot_utils.skills.spot_navigation import \
    navigate_to_absolute_pose_precise
from predicators.spot_utils.skills.spot_place import place_at_relative_position
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.utils import get_robot_gripper_open_percentage


def place_in_waste_valve_region(robot: Robot) -> None:
    """Place an object in the waste valve region of the Ecoself juicer.

    Assumes that the robot is lined up such that it's facing the juicer,
    and it is holding the cup to be placed.
    """
    # Pre-recorded poses for the arm to move through such that
    # it properly places the object in the waste valve region.
    curr_robot_orn = get_end_effector_state(robot).rot
    pose0 = math_helpers.SE3Pose(x=0.9818210005760193, y=0.3118826746940613, z=-0.04188335061073303, rot=curr_robot_orn)
    pose1 = math_helpers.SE3Pose(x=0.9785254001617432, y=0.12745216488838196, z=-0.04429827332496643, rot=curr_robot_orn)
    pose2 = math_helpers.SE3Pose(x=0.6208083033561707, y=0.05527550354599953, z=-0.0448486977815628, rot=curr_robot_orn)
    # Conformant push poses
    pose3 = math_helpers.SE3Pose(x=1.11581552028656, y=0.37607961893081665, z=-0.022341951727867126, rot=math_helpers.Quat(x=-0.012795763090252876, y=0.019999774172902107, z=0.2269088327884674, w=0.9736265540122986))
    pose4 = math_helpers.SE3Pose(x=1.1671946048736572, y=0.20484676957130432, z=-0.029089663177728653, rot=math_helpers.Quat(x=-0.012298588640987873, y=0.020446956157684326, z=0.12993541359901428, w=0.9912353754043579))
    # Place the cup and retract, then push the cup to align
    # with the juicer!
    move_hand_to_relative_pose(robot, pose0)
    time.sleep(0.1)
    move_hand_to_relative_pose(robot, pose1)
    time.sleep(0.1)
    open_gripper(robot)
    time.sleep(0.5)
    move_hand_to_relative_pose(robot, pose2)
    time.sleep(0.1)
    close_gripper(robot)
    stow_arm(robot)
    move_hand_to_relative_pose(robot, pose3)
    time.sleep(0.1)
    move_hand_to_relative_pose_with_velocity(robot, pose3, pose4,
                                             duration=1.5)
    time.sleep(0.1)
    move_hand_to_relative_pose(robot, pose3)
    stow_arm(robot)
    time.sleep(0.1)


def place_in_juice_valve_region(robot: Robot) -> None:
    """Place an object in the juice valve region of the Ecoself juicer.

    Assumes that the robot is lined up such that it's facing the juicer,
    and it is holding the cup to be placed.
    """
    # Pre-recorded poses for the arm to move through such that
    # it properly places the object in the juice valve region.
    curr_robot_orn = get_end_effector_state(robot).rot
    pose0 = math_helpers.SE3Pose(x=0.9258418679237366, y=-0.2896403670310974, z=0.00937592267990112, rot=curr_robot_orn)
    pose1 = math_helpers.SE3Pose(x=0.945264995098114, y=-0.11749177426099777, z=-0.01859173953533173, rot=curr_robot_orn)
    pose2 = math_helpers.SE3Pose(x=0.745264995098114, y=-0.11749177426099777, z=-0.01859173953533173, rot=curr_robot_orn)
    # Conformant push poses
    pose3 = math_helpers.SE3Pose(x=0.98499596118927, y=-0.3708963990211487, z=-0.01537534177303314, rot=math_helpers.Quat(x=0.00928519293665886, y=0.010129101574420929, z=-0.2520125210285187, w=0.9676263928413391))
    pose4 = math_helpers.SE3Pose(x=1.1113766431808472, y=-0.2160855382680893, z=0.014770278707146645, rot=math_helpers.Quat(x=0.002047186717391014, y=0.02063177339732647, z=-0.14226560294628143, w=0.9896113276481628))
    # Place the cup and retract, then push the cup to align
    # with the juicer!
    move_hand_to_relative_pose(robot, pose0)
    time.sleep(0.1)
    move_hand_to_relative_pose(robot, pose1)
    time.sleep(0.1)
    open_gripper(robot)
    time.sleep(1)
    move_hand_to_relative_pose(robot, pose2)
    time.sleep(0.1)
    close_gripper(robot)
    stow_arm(robot)
    move_hand_to_relative_pose(robot, pose3)
    time.sleep(0.1)
    move_hand_to_relative_pose_with_velocity(robot, pose3, pose4,
                                                duration=1.5)
    time.sleep(0.1)
    move_hand_to_relative_pose(robot, pose3)
    stow_arm(robot)


def drop_inside_juicer(robot: Robot) -> None:
    """Place an object inside the juicer.

    Assumes that the robot is lined up such that it's facing the juicer,
    and its arm is stowed. Also assumes the robot is holding the object
    to be juiced.
    """
    pre_place_pose = math_helpers.SE3Pose(x=0.6108426666259766,
                                          y=-0.00424979580566287,
                                          z=0.5442570447921753,
                                          rot=math_helpers.Quat(
                                              x=0.0010119372745975852,
                                              y=0.564949631690979,
                                              z=-0.015587270259857178,
                                              w=0.8249775171279907))
    place_pose = math_helpers.SE3Pose(x=0.9380689964294434,
                                      y=0.026553533487021923,
                                      z=0.42345791816711426,
                                      rot=math_helpers.Quat(
                                          x=0.0010119372745975852,
                                          y=0.564949631690979,
                                          z=-0.015587270259857178,
                                          w=0.8249775171279907))
    look_inside_juicer_pose = math_helpers.SE3Pose(x=0.8897091150283813,
                                                   y=-0.004638137761503458,
                                                   z=0.4537152338027954,
                                                   rot=math_helpers.Quat(
                                                       x=0.0017602762673050165,
                                                       y=0.6850596070289612,
                                                       z=-0.017489491030573845,
                                                       w=0.728274941444397))
    # Move the arm to the right location and
    # drop.
    move_hand_to_relative_pose(robot, pre_place_pose)
    time.sleep(0.1)
    move_hand_to_relative_pose(robot, place_pose)
    time.sleep(0.1)
    open_gripper(robot)
    time.sleep(0.8)
    close_gripper(robot)
    # Move the arm to a position where we can see inside
    # the juicer.
    move_hand_to_relative_pose(robot, look_inside_juicer_pose)
    time.sleep(0.1)


def close_juicer_lid(robot: Robot) -> None:
    """Close the lid of the Ecoself juicer.

    Assumes that the robot is lined up such that it's facing the juicer.
    """
    # Pre-recorded joint angles for the arm to move through such that
    # it properly closes the lid of the juicer.
    pose0 = math_helpers.SE3Pose(x=0.8552572727203369,
                                 y=0.012051388621330261,
                                 z=0.0961974561214447,
                                 rot=math_helpers.Quat(
                                     x=-0.005542381200939417,
                                     y=0.016635911539196968,
                                     z=-0.0014844289980828762,
                                     w=0.9998451471328735))
    pose1 = math_helpers.SE3Pose(x=0.856045184135437,
                                 y=0.012051388621330261,
                                 z=0.37765690207481384,
                                 rot=math_helpers.Quat(
                                     w=0.9883142709732056,
                                     x=-0.017500856891274452,
                                     y=0.14659591019153595,
                                     z=-0.037925124168395996))
    pose2 = math_helpers.SE3Pose(x=0.9593419432640076,
                                 y=-0.030203117057681084,
                                 z=0.34951632738113403,
                                 rot=math_helpers.Quat(w=0.9974877834320068,
                                                       x=-0.025175603106617928,
                                                       y=0.0591338574886322,
                                                       z=-0.02979166992008686))
    ready_to_whack_lid_pose = math_helpers.SE3Pose(
        x=1.0326576232910156,
        y=-0.03433791548013687,
        z=0.4801491451263428,
        rot=math_helpers.Quat(w=0.9980011582374573,
                              x=-0.023644166067242622,
                              y=0.05118606984615326,
                              z=-0.028539864346385002))
    lid_whacking_down_pose = math_helpers.SE3Pose(x=1.0537371635437012,
                                                  y=-0.01710928559303284,
                                                  z=0.33800581693649292,
                                                  rot=math_helpers.Quat(
                                                      w=0.9979082345962524,
                                                      x=-0.02833721600472927,
                                                      y=0.049695659428834915,
                                                      z=-0.030107460916042328))
    # Finally, a pose to move the arm to look and
    # check if the lid is closed.
    lid_check_look_pose = math_helpers.SE3Pose(x=0.5151424407958984,
                                               y=-0.009533177129924297,
                                               z=0.5031274557113647,
                                               rot=math_helpers.Quat(
                                                   w=0.8306255340576172,
                                                   x=-0.009793010540306568,
                                                   y=0.5553292036056519,
                                                   z=-0.039684370160102844))

    # Move the hand through the poses in a
    # trajectory that should closethe lid.
    stow_arm(robot)
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
        move_hand_to_relative_pose_with_velocity(robot,
                                                 ready_to_whack_lid_pose,
                                                 lid_whacking_down_pose,
                                                 duration=0.1)
        time.sleep(0.1)
    # Finally, move the arm to a pose
    # where we can check if the lid is closed.
    move_hand_to_relative_pose(robot, lid_check_look_pose)
    time.sleep(0.1)


def turn_juicer_on(robot: Robot) -> None:
    """Turn on the Ecoself juicer.

    Assumes that the robot is lined up such that it's facing the juicer
    head-on and centered.
    """
    # Define the poses for the arm to move through.
    pre_grasp_pose = math_helpers.SE3Pose(x=0.7174609899520874,
                                          y=0.014277821630239487,
                                          z=-0.04757143884897232,
                                          rot=math_helpers.Quat(
                                              w=0.9982805252075195,
                                              x=0.05800527706742287,
                                              y=-0.00844526756554842,
                                              z=-0.00026528292801231146))
    grasp_pose = math_helpers.SE3Pose(x=0.822599199295044,
                                      y=0.020432027116417885,
                                      z=-0.04757143884897232,
                                      rot=math_helpers.Quat(
                                          w=0.997207760810852,
                                          x=0.0580286830663681,
                                          y=-0.04694018512964249,
                                          z=-0.002454422414302826))
    rotated_quat = math_helpers.Quat(x=0.6800637245178223, 
                                         y=0.028225712478160858, 
                                         z=-0.050544172525405884, 
                                         w=0.7308638095855713)
    juicer_look_pose = math_helpers.SE3Pose(x=0.5151424407958984,
                                            y=-0.009533177129924297,
                                            z=0.5031274557113647,
                                            rot=math_helpers.Quat(
                                                w=0.8306255340576172,
                                                x=-0.009793010540306568,
                                                y=0.5553292036056519,
                                                z=-0.039684370160102844))
    # Now, move the arm through the motions.
    open_gripper(robot)
    move_hand_to_relative_pose(robot, pre_grasp_pose)
    time.sleep(0.1)
    # Keep attempting to move the arm incrementally
    # until the gripper closes around the juicer button.
    max_num_tries = 8
    curr_pose_to_try = grasp_pose
    for _ in range(max_num_tries):
        move_hand_to_relative_pose(robot, curr_pose_to_try)
        time.sleep(0.1)
        close_gripper(robot)
        # NOTE: we use 15 here, because sometimes we get a "partial"
        # grip on the button, and we want to make sure that
        # we have a good grip on it.
        if get_robot_gripper_open_percentage(robot) < 15:
            # We didn't grip the button; move slightly
            # forward and try again.
            open_gripper(robot)
            curr_pose_to_try = math_helpers.SE3Pose(x=curr_pose_to_try.x +
                                                    0.025,
                                                    y=curr_pose_to_try.y,
                                                    z=curr_pose_to_try.z,
                                                    rot=curr_pose_to_try.rot)
        else:
            break
    time.sleep(0.1)
    # Now turn the knob.
    rotated_pose = math_helpers.SE3Pose(x=curr_pose_to_try.x,
                                        y=curr_pose_to_try.y,
                                        z=curr_pose_to_try.z,
                                        rot=rotated_quat)
    move_hand_to_relative_pose(robot, rotated_pose)
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
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.util import authenticate

    from predicators import utils
    from predicators.settings import CFG
    from predicators.spot_utils.spot_localization import SpotLocalizer
    from predicators.spot_utils.utils import get_graph_nav_dir, \
        load_spot_metadata, verify_estop

    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)
        hostname = CFG.spot_robot_ip
        path = get_graph_nav_dir()
        sdk = create_standard_sdk('DumpSkillTestClient')
        robot = sdk.create_robot(hostname)
        authenticate(robot)
        verify_estop(robot)
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        lease_keepalive = LeaseKeepAlive(lease_client,
                                         must_acquire=True,
                                         return_at_exit=True)
        robot.time_sync.wait_for_sync()
        localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)
        juicer_pose_params = load_spot_metadata(
        )["juicing_location"]["juice_machine"]
        juicing_pose = math_helpers.SE2Pose(x=juicer_pose_params[0],
                                            y=juicer_pose_params[1],
                                            angle=juicer_pose_params[2])
        # Testing a skill sequence!
        navigate_to_absolute_pose_precise(robot,
                                          localizer,
                                          juicing_pose,
                                          tolerance=0.025,
                                          max_num_tries=10)
        place_in_juice_valve_region(robot)
        # place_in_waste_valve_region(robot)
        # drop_inside_juicer(robot)
        # close_juicer_lid(robot)
        # turn_juicer_on(robot)

    _run_manual_test()
