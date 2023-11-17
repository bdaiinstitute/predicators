"""Interface for spot dragging skill."""

import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot

from predicators.spot_utils.perception.object_detection import \
    detect_objects, get_grasp_pixel
from predicators.spot_utils.perception.spot_cameras import capture_images
from predicators.spot_utils.skills.spot_grasp import grasp_at_pixel
from predicators.spot_utils.skills.spot_hand_move import \
    move_hand_to_relative_pose, open_gripper
from predicators.spot_utils.skills.spot_navigation import \
    navigate_to_relative_pose
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.utils import DEFAULT_HAND_LOOK_DOWN_POSE, lock_arm


def drag(
    robot: Robot,
    drag_rel_pose: math_helpers.SE2Pose,
    max_xytheta_vel=(0.25, 0.25, 0.1),
    min_xytheta_vel=(-0.25, -0.25, -0.1)
) -> None:
    """Drag in the body frame, assuming that the dragged object is already
    grasped."""
    # Lock the arm.
    lock_arm(robot)

    # Move.
    navigate_to_relative_pose(robot,
                              drag_rel_pose,
                              max_xytheta_vel=max_xytheta_vel,
                              min_xytheta_vel=min_xytheta_vel)

    # Open the gripper and stow the arm to finish.
    open_gripper(robot)
    stow_arm(robot)


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # NOTE: this test assumes that the robot is standing in front of the
    # platform. The platform will be dragged diagonally back and right.

    # pylint: disable=ungrouped-imports
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.util import authenticate

    from predicators import utils
    from predicators.settings import CFG
    from predicators.spot_utils.perception.perception_structs import \
        AprilTagObjectDetectionID
    from predicators.spot_utils.skills.spot_find_objects import \
        init_search_for_objects
    from predicators.spot_utils.spot_localization import SpotLocalizer
    from predicators.spot_utils.utils import get_graph_nav_dir, \
        get_relative_se2_from_se3, get_spot_home_pose, verify_estop

    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)

        # Set up the robot and localizer.
        hostname = CFG.spot_robot_ip
        path = get_graph_nav_dir()
        sdk = create_standard_sdk("DragTestClient")
        robot = sdk.create_robot(hostname)
        authenticate(robot)
        verify_estop(robot)
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        lease_keepalive = LeaseKeepAlive(lease_client,
                                         must_acquire=True,
                                         return_at_exit=True)
        assert path.exists()
        localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)

        platform = AprilTagObjectDetectionID(411)

        # Test assumes that the platform is in front of the robot.
        localizer.localize()

        home_pose = get_spot_home_pose()
        pre_pick_nav_angle = home_pose.angle - np.pi
        pre_pick_nav_distance = 1.25

        # Find the platform.
        detections, _ = init_search_for_objects(robot, localizer, [platform])

        # Navigate to in front of the platform.
        localizer.localize()
        robot_pose = localizer.get_last_robot_pose()
        rel_pose = get_relative_se2_from_se3(robot_pose, detections[platform],
                                             pre_pick_nav_distance,
                                             pre_pick_nav_angle)
        navigate_to_relative_pose(robot, rel_pose)
        localizer.localize()

        # Look down at the surface.
        move_hand_to_relative_pose(robot, DEFAULT_HAND_LOOK_DOWN_POSE)
        open_gripper(robot)

        # Capture an image from the hand camera.
        hand_camera = "hand_color_image"
        rgbds = capture_images(robot, localizer, [hand_camera])
        rgbd = rgbds[hand_camera]

        # Run detection to get a pixel for grasping.
        _, artifacts = detect_objects([platform], rgbds)
        pixel = get_grasp_pixel(rgbds, artifacts, platform, hand_camera)

        # Pick at the pixel with a top-down and rotated grasp.
        top_down_rot = math_helpers.Quat.from_pitch(np.pi / 2)
        side_rot = math_helpers.Quat.from_yaw(np.pi / 2)
        grasp_rot = side_rot * top_down_rot
        grasp_at_pixel(robot, rgbd, pixel, grasp_rot=grasp_rot)
        localizer.localize()

        # Drag!
        drag_rel_pose = math_helpers.SE2Pose(-0.5, 0.5, 0.0)
        drag(robot, drag_rel_pose)

    _run_manual_test()
