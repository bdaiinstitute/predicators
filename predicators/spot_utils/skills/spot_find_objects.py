"""Interface for finding objects by moving around and running detection."""
import time
from collections import defaultdict
from typing import Any, Collection, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.lease import LeaseClient
from bosdyn.client.math_helpers import SE3Pose
from bosdyn.client.sdk import Robot
from rich import print
from scipy.spatial import Delaunay

from predicators import utils
from predicators.settings import CFG
from predicators.spot_utils.perception.object_detection import detect_objects
from predicators.spot_utils.perception.object_perception import \
    get_vlm_atom_combinations, vlm_predicate_batch_classify
from predicators.spot_utils.perception.perception_structs import \
    ObjectDetectionID, RGBDImageWithContext
from predicators.spot_utils.perception.spot_cameras import capture_images
from predicators.spot_utils.skills.spot_hand_move import close_gripper, \
    move_hand_to_relative_pose, open_gripper
from predicators.spot_utils.skills.spot_navigation import \
    navigate_to_relative_pose
from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.utils import DEFAULT_HAND_LOOK_DOWN_POSE, \
    DEFAULT_HAND_LOOK_FLOOR_POSE, get_allowed_map_regions, \
    get_collision_geoms_for_nav, get_relative_se2_from_se3, \
    sample_random_nearby_point_to_move, spot_pose_to_geom2d
from predicators.structs import Object, State, VLMGroundAtom, VLMPredicate


def _find_objects_with_choreographed_moves(
    robot: Robot,
    localizer: SpotLocalizer,
    object_ids: Collection[ObjectDetectionID],
    relative_base_moves: Sequence[math_helpers.SE2Pose],
    relative_hand_moves: Optional[Sequence[math_helpers.SE3Pose]] = None,
    open_and_close_gripper: bool = True,
    allowed_regions: Optional[Collection[Delaunay]] = None,
    vlm_predicates: Optional[Set[VLMPredicate]] = None,
    id2object: Optional[Dict[ObjectDetectionID, Object]] = None,
) -> Tuple[Dict[ObjectDetectionID, SE3Pose], Dict[str, Any], Dict[
        VLMGroundAtom, bool or None]]:
    """Helper for object search with hard-coded relative moves."""

    if relative_hand_moves is not None:
        assert len(relative_hand_moves) == len(relative_base_moves)

    # Naively combine detections and artifacts using the most recent ones.
    all_detections: Dict[ObjectDetectionID, math_helpers.SE3Pose] = {}
    all_artifacts: Dict[str, Any] = {}
    # Save all RGBDs in case of failure so we can analyze them.
    all_rgbds: List[Dict[str, RGBDImageWithContext]] = []

    # Save VLMGroundAtoms from all poses
    # NOTE: overwrite if the same atom is found; to improve later
    all_vlm_atom_dict: Dict[VLMGroundAtom,
                            bool or None] = defaultdict(lambda: None)

    # Open the hand to mitigate possible occlusions.
    if open_and_close_gripper:
        open_gripper(robot)

    # Wait briefly for the hand to finish opening.
    time.sleep(0.5)

    # Run detection once to start before moving.
    rgbds = capture_images(robot, localizer)
    all_rgbds.append(rgbds)
    detections, artifacts = detect_objects(object_ids,
                                           rgbds,
                                           allowed_regions=allowed_regions)
    all_detections.update(detections)
    all_artifacts.update(artifacts)

    for i, relative_pose in enumerate(relative_base_moves):
        remaining_object_ids = set(object_ids) - set(all_detections)
        print(f"Found objects: {set(all_detections)}")
        print(f"Remaining objects: {remaining_object_ids}")

        # Get VLM queries + Send request
        # TODO: We may query objects only in current view's images.
        # Now we query all detected objects in all past views.
        if CFG.spot_vlm_eval_predicate and len(all_detections) > 0 and len(
                vlm_predicates) > 0:
            objects = [id2object[id_] for id_ in all_detections]
            vlm_atoms = get_vlm_atom_combinations(objects, vlm_predicates)
            vlm_atom_dict = vlm_predicate_batch_classify(
                vlm_atoms, rgbds, predicates=vlm_predicates, get_dict=True)
            # Update value if original is None while new is not None
            for atom, result in vlm_atom_dict.items():
                if all_vlm_atom_dict[atom] is None and result is not None:
                    all_vlm_atom_dict[atom] = result
            print(f"Calculated VLM atoms: {dict(vlm_atom_dict)}")
            print(
                f"True VLM atoms (with values as True): {dict(filter(lambda it: it[1], all_vlm_atom_dict.items()))}"
            )
        else:
            # No VLM predicates or no objects found yet
            pass

        # Success, finish.
        if not remaining_object_ids:
            break

        # Move and re-capture.
        navigate_to_relative_pose(robot, relative_pose)

        if relative_hand_moves is not None:
            hand_move = relative_hand_moves[i]
            move_hand_to_relative_pose(robot, hand_move)

        localizer.localize()
        rgbds = capture_images(robot, localizer)
        all_rgbds.append(rgbds)
        detections, artifacts = detect_objects(object_ids,
                                               rgbds,
                                               allowed_regions=allowed_regions)
        all_detections.update(detections)
        all_artifacts.update(artifacts)

    # Close the gripper.
    if open_and_close_gripper:
        close_gripper(robot)

    # Success, finish.
    remaining_object_ids = set(object_ids) - set(all_detections)
    if not remaining_object_ids:
        return all_detections, all_artifacts, all_vlm_atom_dict

    # Fail. Analyze the RGBDs if you want (by uncommenting here).
    # import imageio.v2 as iio
    # for i, rgbds in enumerate(all_rgbds):
    #     for camera, rgbd in rgbds.items():
    #         path = f"init_search_for_objects_angle{i}_{camera}.png"
    #         iio.imsave(path, rgbd.rgb)
    #         print(f"Wrote out to {path}.")

    remaining_object_ids = set(object_ids) - set(all_detections)
    raise RuntimeError(f"Could not find objects: {remaining_object_ids}")


def init_search_for_objects(
    robot: Robot,
    localizer: SpotLocalizer,
    object_ids: Collection[ObjectDetectionID],
    num_spins: int = 8,
    relative_hand_moves: Optional[List[math_helpers.SE3Pose]] = None,
    allowed_regions: Optional[Collection[Delaunay]] = None,
    vlm_predicates: Optional[Set[VLMPredicate]] = None,
    id2object: Optional[Dict[ObjectDetectionID, Object]] = None,
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any], Dict[
        VLMGroundAtom, bool or None]]:
    """Spin around in place looking for objects.

    Raise a RuntimeError if an object can't be found after spinning.
    """
    spin_amount = 2 * np.pi / (num_spins + 1)
    relative_pose = math_helpers.SE2Pose(0, 0, spin_amount)
    base_moves = [relative_pose] * num_spins
    return _find_objects_with_choreographed_moves(
        robot,
        localizer,
        object_ids,
        base_moves,
        relative_hand_moves=relative_hand_moves,
        allowed_regions=allowed_regions,
        vlm_predicates=vlm_predicates,
        id2object=id2object,
    )


def step_back_to_find_objects(
    robot: Robot,
    localizer: SpotLocalizer,
    object_ids: Collection[ObjectDetectionID],
    allowed_regions: Optional[Collection[Delaunay]] = None,
) -> None:
    """Execute a hard-coded sequence of movements and hope that one of them
    puts the lost object in view.

    This is very specifically designed for the case where an object has
    fallen in the immediate vicinity.
    """
    moves = [
        # First move way back and don't move the hand. This is useful when the
        # object has not actually fallen, but wasn't grasped.
        (math_helpers.SE2Pose(-0.75, 0.0, 0.0), DEFAULT_HAND_LOOK_DOWN_POSE),
        # Just look down at the floor.
        (math_helpers.SE2Pose(0.0, 0.0, 0.0), DEFAULT_HAND_LOOK_FLOOR_POSE),
        # Spin to the right and look at the floor.
        (math_helpers.SE2Pose(0.0, 0.0,
                              np.pi / 6), DEFAULT_HAND_LOOK_FLOOR_POSE),
        # Spin to the left and look at the floor.
        (math_helpers.SE2Pose(0.0, 0.0,
                              -np.pi / 6), DEFAULT_HAND_LOOK_FLOOR_POSE),
    ]
    base_moves, hand_moves = zip(*moves)
    # Don't open and close the gripper because we need the object to be
    # in view when the action has finished, and we can't leave the gripper
    # open because then HandEmpty will misfire.
    _find_objects_with_choreographed_moves(robot,
                                           localizer,
                                           object_ids,
                                           base_moves,
                                           hand_moves,
                                           open_and_close_gripper=False,
                                           allowed_regions=allowed_regions)


def find_objects(
    state: State,
    rng: np.random.Generator,
    robot: Robot,
    localizer: SpotLocalizer,
    lease_client: LeaseClient,
    object_ids: Collection[ObjectDetectionID],
    allowed_regions: Optional[Collection[Delaunay]] = None,
) -> None:
    """First try stepping back to find an object, and if that doesn't work,
    then try to either ask the user or keep sampling a random location to move
    to in order to find the lost object."""
    try:
        step_back_to_find_objects(robot,
                                  localizer,
                                  object_ids,
                                  allowed_regions=allowed_regions)
    except RuntimeError:
        prompt = ("Hit 'c' to have the robot try to find the object "
                  "by moving to a random pose, or "
                  "take control of the robot and make the object "
                  "become in its view. Hit the 'Enter' key when you're done!")
        user_pref = input(prompt)
        lease_client.take()
        if user_pref == "c":
            localizer.localize()
            spot_pose = localizer.get_last_robot_pose()
            robot_geom = spot_pose_to_geom2d(spot_pose)
            collision_geoms = get_collision_geoms_for_nav(state)
            allowed_regions = get_allowed_map_regions()
            dist, yaw, _ = sample_random_nearby_point_to_move(
                robot_geom, collision_geoms, rng, 2.5, allowed_regions)
            rel_pose = get_relative_se2_from_se3(spot_pose, spot_pose, dist,
                                                 yaw)
            navigate_to_relative_pose(robot, rel_pose)


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # This test assumes that the 408, 409, and 410 april tags can be found.

    # pylint: disable=ungrouped-imports
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseKeepAlive
    from bosdyn.client.util import authenticate

    from predicators.settings import CFG
    from predicators.spot_utils.perception.object_detection import \
        AprilTagObjectDetectionID
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
        sdk = create_standard_sdk('FindObjectsTestClient')
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

        object_ids = [
            # Table.
            AprilTagObjectDetectionID(408),
            # Table.
            AprilTagObjectDetectionID(409),
            # Cube.
            AprilTagObjectDetectionID(410),
        ]

        # Test running the initial search for objects.
        input("Set up initial object search test")
        init_search_for_objects(robot, localizer, object_ids)

        # Test finding a lost object.
        input("Set up finding lost object test")
        cube = object_ids[2]

        step_back_to_find_objects(robot, localizer, {cube})

    _run_manual_test()
