"""Interface for spot dumping skill."""

import time

import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot

from predicators.spot_utils.skills.spot_hand_move import \
    move_hand_to_relative_pose
from predicators.spot_utils.skills.spot_place import place_at_relative_position


def sweep(robot: Robot,
          sweep_target: math_helpers.Vec3,
          sweep_start_distance: float,
          sweep_yaw: float,
          sweep_magnitude: float) -> None:
    """TODO document
    """
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # NOTE: this test assumes that the robot is standing in front of a table
    # that has a soda can on it. The test starts by running object detection to
    # get the pose of the soda can. Then the robot opens its gripper and pauses
    # until a brush is put in the gripper, with the bristles facing down and
    # forward. The robot should then brush the soda can to the right.

    # pylint: disable=ungrouped-imports
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.util import authenticate

    from predicators import utils
    from predicators.settings import CFG
    from predicators.spot_utils.perception.perception_structs import \
        LanguageObjectDetectionID
    from predicators.spot_utils.spot_localization import SpotLocalizer
    from predicators.spot_utils.utils import get_graph_nav_dir, verify_estop
    from predicators.spot_utils.skills.spot_navigation import go_home
    from predicators.spot_utils.skills.spot_find_objects import init_search_for_objects
    from predicators.spot_utils.skills.spot_hand_move import open_gripper, close_gripper


    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)

        # Get constants.
        hostname = CFG.spot_robot_ip
        path = get_graph_nav_dir()

        sdk = create_standard_sdk('SweepSkillTestClient')
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

        # Go home.
        go_home(robot, localizer)

        # Find the soda can.
        soda_detection_id = LanguageObjectDetectionID("soda can")
        detections, _ = init_search_for_objects(robot, localizer, {soda_detection_id})
        soda_pose = detections[soda_detection_id]

        # Ask for the brush.
        open_gripper(robot)
        input("Put the brush in the robot's gripper, then press enter")
        close_gripper(robot)

        # Sweep.
        

    _run_manual_test()
