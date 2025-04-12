"""Interface for spot dumping skill."""

import time

import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot

from predicators.spot_utils.skills.spot_hand_move import \
    move_hand_to_relative_pose
from predicators.spot_utils.skills.spot_place import place_at_relative_position


def dump_container(robot: Robot,
                   place_z: float,
                   place_angle: float = np.pi / 3,
                   dump_x: float = 0.8,
                   dump_y: float = -0.4,
                   dump_z: float = 0.5,
                   place_y: float = 0.0) -> None:
    """Turn over and dump out a container.

    Assumes that the container is grasped with a top-down grasp on the
    side of the container, and with the fingers pointed inward.
    """
    # # Construct the desired hand pose for dumping.
    # yaw = math_helpers.Quat.from_yaw(np.pi / 3)
    # roll2 = math_helpers.Quat.from_roll(np.pi / 3)
    # roll1 = math_helpers.Quat.from_roll(np.pi / 3)
    # rot = roll1 * yaw * roll2
    dump_quat = math_helpers.Quat(w=0.865, x=-0.201, y=-0.448, z=0.102)
    hand_dump_pose = math_helpers.SE3Pose(x=dump_x,
                                          y=dump_y,
                                          z=dump_z,
                                          rot=dump_quat)
    # Execute the move to the pose.
    move_hand_to_relative_pose(robot, hand_dump_pose)
    # Wait a few seconds for the object(s) to be dumped.
    time.sleep(2.0)
    # Place the container back down.
    body_to_position = math_helpers.Vec3(x=dump_x, y=place_y, z=place_z)
    place_at_relative_position(robot, body_to_position, place_angle)


# def get_end_effector_state(robot) -> None:
#     """Get the current position and orientation of the Spot arm's end effector."""
#     # Create a RobotStateClient to query the robot's state
#     from bosdyn.client.robot_state import RobotStateClient

#     state_client = robot.ensure_client(RobotStateClient.default_service_name)

#     # Get the robot's state
#     robot_state = state_client.get_robot_state()

#     # Access the kinematic state of the arm
#     arm_state = robot_state.kinematic_state

#     # Extract the end-effector pose (position and orientation)
#     if arm_state and arm_state.transforms_snapshot:
#         ee_transform = arm_state.transforms_snapshot.child_to_parent_edge_map.get("hand", None)
#         if ee_transform:
#             position = ee_transform.parent_tform_child.position
#             orientation = ee_transform.parent_tform_child.rotation
#             print(f"End Effector Position: x={position.x}, y={position.y}, z={position.z}")
#             print(f"End Effector Orientation (Quaternion): x={orientation.x}, y={orientation.y}, z={orientation.z}, w={orientation.w}")
#         else:
#             print("End effector transform not found.")
#     else:
#         print("Arm state or transforms snapshot not available.")

if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # NOTE: this test assumes that the robot is facing the bucket.

    # pylint: disable=ungrouped-imports
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.util import authenticate

    from predicators import utils
    from predicators.settings import CFG
    from predicators.spot_utils.perception.object_detection import \
        detect_objects, get_grasp_pixel
    from predicators.spot_utils.perception.perception_structs import \
        LanguageObjectDetectionID
    from predicators.spot_utils.perception.spot_cameras import capture_images
    from predicators.spot_utils.skills.spot_grasp import grasp_at_pixel
    from predicators.spot_utils.spot_localization import SpotLocalizer
    from predicators.spot_utils.utils import DEFAULT_HAND_LOOK_FLOOR_POSE, \
        get_graph_nav_dir, verify_estop

    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)

        place_height = 0.05  # taking into account the size of the bucket

        # Get constants.
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

        # Start by looking down and then grasping the red bucket.
        move_hand_to_relative_pose(robot, DEFAULT_HAND_LOOK_FLOOR_POSE)

        # Capture an image.
        camera = "hand_color_image"
        rgbds = capture_images(robot, localizer, [camera])
        rgbd = rgbds[camera]

        # Run detection to find the bucket.
        # Detect the april tag and brush.
        bucket_id = LanguageObjectDetectionID("large red bucket")
        _, artifacts = detect_objects([bucket_id], rgbds)
        rng = np.random.default_rng(CFG.seed)
        (r, c), _ = get_grasp_pixel(rgbds, artifacts, bucket_id, camera, rng)
        pixel = (r + 50, c)

        # Grasp at the pixel with a top-down grasp.
        top_down_rot = math_helpers.Quat.from_pitch(np.pi / 2)
        grasp_at_pixel(robot, rgbd, pixel, grasp_rot=top_down_rot)

        # Dump!
        dump_container(robot, place_height)
        # get_end_effector_state(robot)

    _run_manual_test()
