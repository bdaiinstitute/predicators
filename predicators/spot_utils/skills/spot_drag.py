"""Interface for spot dragging skill."""

import time

import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot

from predicators.spot_utils.skills.spot_navigation import \
    navigate_to_relative_pose


def drag_object(robot: Robot, relative_move: math_helpers.SE2Pose) -> None:
    """Drag a grasped object in some direction.

    Assumes that the object is already grasped and can be moved.
    """
    navigate_to_relative_pose(robot, relative_move)


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # NOTE: this test assumes that the robot is facing the chair.

    # pylint: disable=ungrouped-imports
    from pathlib import Path

    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.util import authenticate

    from predicators import utils
    from predicators.settings import CFG
    from predicators.spot_utils.perception.object_detection import \
        detect_objects, get_object_center_pixel_from_artifacts
    from predicators.spot_utils.perception.perception_structs import \
        LanguageObjectDetectionID
    from predicators.spot_utils.perception.spot_cameras import capture_images
    from predicators.spot_utils.skills.spot_grasp import grasp_at_pixel
    from predicators.spot_utils.spot_localization import SpotLocalizer
    from predicators.spot_utils.utils import get_graph_nav_dir, verify_estop

    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)

        # Get constants.
        hostname = CFG.spot_robot_ip
        path = get_graph_nav_dir()

        sdk = create_standard_sdk('DragSkillTestClient')
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
        # Capture an image.
        camera = "hand_color_image"
        rgbds = capture_images(robot, localizer, [camera])
        rgbd = rgbds[camera]

        # Run detection to find the bucket.
        # Detect the april tag and brush.
        chair_id = LanguageObjectDetectionID("chair arm")
        _, artifacts = detect_objects([chair_id], rgbds)

        pixel = get_object_center_pixel_from_artifacts(artifacts, chair_id,
                                                       camera)

        # Grasp the chair somewhere.
        top_down_rot = math_helpers.Quat.from_pitch(np.pi / 2)
        grasp_at_pixel(robot, rgbd, pixel, grasp_rot=top_down_rot)

        # Drag backwards and to the right.
        drag_object(robot, math_helpers.SE2Pose(x=-1.0, y=-0.8, angle=0.0))

    _run_manual_test()
