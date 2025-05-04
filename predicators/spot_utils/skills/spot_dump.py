"""Interface for spot dumping skill."""

import time

import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot

from predicators.spot_utils.skills.spot_hand_move import \
    move_hand_to_relative_pose, move_arm_to_joint_angles, open_gripper
from predicators.spot_utils.skills.spot_place import place_at_relative_position
from predicators.spot_utils.skills.spot_hand_move import open_gripper


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
    # dump_quat = math_helpers.Quat(w=0.193, x=0.846, y=0.037, z=0.495)
    # hand_dump_pose = math_helpers.SE3Pose(x=dump_x,
    #                                       y=dump_y,
    #                                       z=dump_z,
    #                                       rot=dump_quat)
   
    # dump_quat = math_helpers.Quat(x=-0.6664543747901917, y=0.6712515354156494, z=-0.19593559205532074, w=-0.25859081745147705)
    # # override dump x, y, z
    # dump_x=0.4373268187046051
    # dump_y=-0.49700379371643066
    # dump_z=0.28532472252845764
    # hand_dump_pose = math_helpers.SE3Pose(x=dump_x,
    #                                         y=dump_y,
    #                                         z=dump_z,
    #                                         rot=dump_quat)
    # # Execute the move to the pose.
    # move_hand_to_relative_pose(robot, hand_dump_pose)
    # Move the hand to the desired pose.
    down_quat = math_helpers.Quat.from_pitch(np.pi / 2.5)
    end_eff_pose = get_end_effector_state(robot)
    hand_place_back_pose = math_helpers.SE3Pose(x=end_eff_pose.x,
                                          y=end_eff_pose.y,
                                          z=end_eff_pose.z - 0.01,
                                          rot=down_quat)
    # Joint values for dumping:
    joint_values = [-1.5590229034423828, -0.754558265209198, 1.6554505825042725, -0.05442166328430176, -1.8044668436050415, 2.883525848388672]
    move_arm_to_joint_angles(robot, joint_values)
    # Wait a few seconds for the object(s) to be dumped.
    time.sleep(2.0)
    # Place the container back down.
    # body_to_position = math_helpers.Vec3(x=0.6, y=-0.3, z=-0.3)
    move_hand_to_relative_pose(robot, hand_place_back_pose)
    time.sleep(0.5)
    open_gripper(robot)


def get_end_effector_state(robot) -> None:
    """Get the current position and orientation of the Spot arm's end effector."""
    # Create a RobotStateClient to query the robot's state
    from bosdyn.client.robot_state import RobotStateClient

    state_client = robot.ensure_client(RobotStateClient.default_service_name)

    # Get the robot's state
    robot_state = state_client.get_robot_state()

    # Access the kinematic state of the arm
    arm_state = robot_state.kinematic_state

    # Extract the end-effector pose (position and orientation)
    if arm_state and arm_state.transforms_snapshot:
        ee_transform = arm_state.transforms_snapshot.child_to_parent_edge_map.get("hand", None)
        if ee_transform:
            position = ee_transform.parent_tform_child.position
            orientation = ee_transform.parent_tform_child.rotation
            print(f"End Effector Position: x={position.x}, y={position.y}, z={position.z}")
            print(f"End Effector Orientation (Quaternion): x={orientation.x}, y={orientation.y}, z={orientation.z}, w={orientation.w}")
            orientation = math_helpers.Quat(x=orientation.x, y=orientation.y, z=orientation.z, w=orientation.w)
            return math_helpers.SE3Pose(position.x, position.y, position.z, rot=orientation)
        else:
            print("End effector transform not found.")
    else:
        print("Arm state or transforms snapshot not available.")

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

        # # Start by looking down and then grasping the red bucket.
        # move_hand_to_relative_pose(robot, DEFAULT_HAND_LOOK_FLOOR_POSE)

        # # Capture an image.
        # camera = "hand_color_image"
        # rgbds = capture_images(robot, localizer, [camera])
        # rgbd = rgbds[camera]

        # # Run detection to find the bucket.
        # # Detect the april tag and brush.
        # bucket_id = LanguageObjectDetectionID("bottle/clear_cup/clear_trashcan")
        # _, artifacts = detect_objects([bucket_id], rgbds)
        # rng = np.random.default_rng(CFG.seed)
        # pixel, _ = get_grasp_pixel(rgbds, artifacts, bucket_id, camera, rng)

        # # Grasp at the pixel with a top-down grasp.
        # top_down_rot = math_helpers.Quat.from_pitch(np.pi / 2)
        # grasp_at_pixel(robot, rgbd, pixel, grasp_rot=top_down_rot)

        # Dump!
        dump_container(robot, place_height)

        # get_end_effector_state(robot)

    _run_manual_test()
