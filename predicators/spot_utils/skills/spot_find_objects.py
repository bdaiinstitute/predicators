"""Interface for finding objects by moving around and running detection."""
import logging
import time
from collections import defaultdict
from typing import Any, Collection, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.lease import LeaseClient
from bosdyn.client.math_helpers import SE3Pose
from bosdyn.client.sdk import Robot
from rich import print
from rich.table import Table
from scipy.spatial import Delaunay

from predicators import utils
from predicators.settings import CFG
from predicators.spot_utils.perception import vlm_pointing
from predicators.spot_utils.perception.object_detection import detect_objects
from predicators.spot_utils.perception.object_perception import \
    get_vlm_atom_combinations, vlm_predicate_batch_classify
from predicators.spot_utils.perception.perception_structs import \
    LanguageObjectDetectionID, ObjectDetectionID, RGBDImageWithContext
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
        VLMGroundAtom, Optional[bool]]]:
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
                            Optional[bool]] = defaultdict(lambda: None)

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
    
    # NOTE: For hardcode robot object
    from predicators.spot_utils.utils import _robot_type

    for i, relative_pose in enumerate(relative_base_moves):
        remaining_object_ids = set(object_ids) - set(all_detections)
        print(f"Found objects: {set(all_detections)}")
        print(f"Remaining objects: {remaining_object_ids}")

        # Get VLM queries + Send request
        # TODO: We may query objects only in current view's images.
        # Now we query all detected objects in all past views.
        if CFG.spot_vlm_eval_predicate and len(all_detections) > 0 and vlm_predicates and len(vlm_predicates) > 0:
            assert id2object is not None
            # NOTE: Get all objects; hardcode Spot robot object here
            objects = [id2object[id_] for id_ in all_detections]
            # Create constant robot object
            _robot_object = Object("robot", _robot_type)
            objects.append(_robot_object)
            
            vlm_atoms = get_vlm_atom_combinations(objects, vlm_predicates)
            vlm_atom_dict = vlm_predicate_batch_classify(
                vlm_atoms, rgbds, predicates=vlm_predicates, get_dict=True)
            # Update value if original is None while new is not None
            for atom, result in vlm_atom_dict.items():
                if all_vlm_atom_dict[atom] is None and result is not None:
                    all_vlm_atom_dict[atom] = result
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

    # Logging
    if CFG.vlm_eval_verbose:
        print(f"Calculated VLM atoms (in all views): {dict(sorted(all_vlm_atom_dict.items()))}")
        print(f"True VLM atoms (in all views; with values as True): "
            f"{dict(sorted(filter(lambda it: it[1], all_vlm_atom_dict.items())))}")

    table = Table(title="Evaluated VLM atoms (in all views)")
    table.add_column("Atom", style="cyan")
    table.add_column("Value", style="magenta")
    for atom, result in sorted(all_vlm_atom_dict.items(), key=lambda item: str(item[0])):
        table.add_row(str(atom), str(result))
    print(table)

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


def _teleop_search_for_objects(
    robot: Robot,
    localizer: SpotLocalizer,
    object_ids: Collection[ObjectDetectionID],
    allowed_regions: Optional[Collection[Delaunay]] = None,
    vlm_predicates: Optional[Set[VLMPredicate]] = None,
    id2object: Optional[Dict[ObjectDetectionID, Object]] = None,
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any], Dict[
        VLMGroundAtom, Optional[bool]]]:
    """Manual object search for demo mode.

    The operator is expected to move Spot (via tablet/teleop) until the objects
    are in view, then press Enter to trigger detection.
    """
    print("\n=== Demo Mode: Teleop object search ===")
    print("Use the tablet/teleop controls to position Spot so that the target")
    print("objects are visible. When ready, press Enter to run detection.")
    print("Type 'skip' to abort teleop search.\n")

    # Track detections and VLM atoms across attempts.
    all_vlm_atom_dict: Dict[VLMGroundAtom, Optional[bool]] = defaultdict(
        lambda: None)
    attempt = 0

    while True:
        attempt += 1
        user_input = input(
            f"[Teleop Search] Attempt {attempt}: press Enter to capture images "
            "or type 'skip' to abort: ").strip().lower()
        if user_input == "skip":
            raise RuntimeError("Teleop object search aborted by operator.")

        rgbds = capture_images(robot, localizer)

        language_ids: Set[ObjectDetectionID] = {
            obj_id for obj_id in object_ids
            if isinstance(obj_id, LanguageObjectDetectionID)
        }
        if CFG.spot_teleop_pointing_detection:
            detic_ids = set(object_ids) - language_ids
            if detic_ids:
                logging.info(
                    "[Teleop Search] Skipping DETIC/SAM for %d language IDs;"
                    " still running for %d non-language IDs.",
                    len(language_ids), len(detic_ids))
        else:
            detic_ids = set(object_ids)

        detections: Dict[ObjectDetectionID, math_helpers.SE3Pose] = {}
        artifacts: Dict[str, Any] = {
            "rgbds": rgbds,
            "language": {"object_id_to_img_detections": {}},
            "april": {}
        }
        if detic_ids:
            detic_detections, artifacts = detect_objects(detic_ids,
                                                         rgbds,
                                                         allowed_regions=
                                                         allowed_regions)
            detections.update(detic_detections)

        if CFG.spot_use_vlm_pointing and id2object is not None:
            if CFG.spot_teleop_pointing_detection:
                target_ids = language_ids
            else:
                target_ids = language_ids - set(detections)
            if target_ids:
                pointing_detections = _pointing_detect_language_objects(
                    target_ids, rgbds, allowed_regions, id2object)
                if pointing_detections:
                    detections.update(pointing_detections)
                else:
                    logging.warning("[Teleop Search] Gemini pointing returned no"
                                    " detections for %s.", target_ids)
                    if CFG.spot_teleop_pointing_detection:
                        logging.info("[Teleop Search] Falling back to DETIC/SAM "
                                     "for %s after Gemini failure.",
                                     target_ids)
                        fallback_detections, _ = detect_objects(
                            target_ids, rgbds,
                            allowed_regions=allowed_regions)
                        if fallback_detections:
                            detections.update(fallback_detections)

        remaining_object_ids = set(object_ids) - set(detections)
        print(f"[Teleop Search] Found: {set(detections)}")
        if not remaining_object_ids:
            # Optionally evaluate VLM predicates once objects are in view.
            if CFG.spot_vlm_eval_predicate and vlm_predicates and id2object:
                from predicators.spot_utils.utils import _robot_type

                objects = [id2object[obj_id] for obj_id in detections]
                objects.append(Object("robot", _robot_type))
                vlm_atoms = get_vlm_atom_combinations(objects, vlm_predicates)
                vlm_atom_dict = vlm_predicate_batch_classify(
                    vlm_atoms, rgbds, predicates=vlm_predicates, get_dict=True)
                for atom, result in vlm_atom_dict.items():
                    if all_vlm_atom_dict[atom] is None and result is not None:
                        all_vlm_atom_dict[atom] = result
            return detections, artifacts, all_vlm_atom_dict

        print(f"[Teleop Search] Remaining objects: {remaining_object_ids}")
        print("Adjust Spot's position and try again.\n")


def _pose_within_allowed_regions(
        pose: math_helpers.SE3Pose,
        allowed_regions: Optional[Collection[Delaunay]]) -> bool:
    if allowed_regions is None:
        return True
    pose_xy = np.array([pose.x, pose.y])
    for region in allowed_regions:
        if region.find_simplex(pose_xy).item() >= 0:
            return True
    return False


def _pose_from_pixel(rgbd: RGBDImageWithContext,
                     pixel: Tuple[int, int],
                     min_depth_value: float = 2.0
                    ) -> Optional[math_helpers.SE3Pose]:
    """Convert a pixel in an RGBD image into a world-frame pose."""
    x_pix, y_pix = pixel
    height, width = rgbd.depth.shape
    if not (0 <= x_pix < width and 0 <= y_pix < height):
        return None
    depth_value = rgbd.depth[y_pix, x_pix]
    if depth_value <= min_depth_value:
        logging.debug("Pointing fallback depth too small at pixel %s", pixel)
        return None
    fx = rgbd.camera_model.intrinsics.focal_length.x
    fy = rgbd.camera_model.intrinsics.focal_length.y
    cx = rgbd.camera_model.intrinsics.principal_point.x
    cy = rgbd.camera_model.intrinsics.principal_point.y
    depth_scale = rgbd.depth_scale
    camera_z = depth_value / depth_scale
    camera_x = np.multiply(camera_z, (x_pix - cx)) / fx
    camera_y = np.multiply(camera_z, (y_pix - cy)) / fy
    camera_pose = math_helpers.SE3Pose(
        float(camera_x),
        float(camera_y),
        float(camera_z),
        rot=math_helpers.Quat(),
    )
    return rgbd.world_tform_camera * camera_pose


def _pointing_detect_language_objects(
        remaining_ids: Collection[ObjectDetectionID],
        rgbds: Dict[str, RGBDImageWithContext],
        allowed_regions: Optional[Collection[Delaunay]],
        id2object: Dict[ObjectDetectionID, Object]
) -> Dict[ObjectDetectionID, math_helpers.SE3Pose]:
    """Use the VLM pointing service to approximate detections for teleop mode."""
    detections: Dict[ObjectDetectionID, math_helpers.SE3Pose] = {}
    if not remaining_ids:
        return detections

    camera_priority = [
        "hand_color_image",
        "frontleft_fisheye_image",
        "frontright_fisheye_image",
        "left_fisheye_image",
        "right_fisheye_image",
        "back_fisheye_image",
    ]
    other_cameras = [c for c in rgbds if c not in camera_priority]
    ordered_cameras = camera_priority + other_cameras

    for obj_id in remaining_ids:
        if not isinstance(obj_id, LanguageObjectDetectionID):
            continue
        target_obj = id2object.get(obj_id)
        if target_obj is None:
            continue
        for camera_name in ordered_cameras:
            if camera_name not in rgbds:
                continue
            result = vlm_pointing.compute_pointing_result(
                target_obj,
                rgbds,
                detection_artifacts=None,
                detection_id_to_obj={},
                rng=None,
                camera_name=camera_name)
            if result is None or result.pixel is None:
                continue
            pose = _pose_from_pixel(rgbds[camera_name], result.pixel)
            if pose is None:
                continue
            if not _pose_within_allowed_regions(pose, allowed_regions):
                logging.info(
                    "Pointing fallback pose for %s outside allowed region; skipping.",
                    obj_id.language_id)
                continue
            detections[obj_id] = pose
            logging.info("Pointing fallback succeeded for %s using %s camera.",
                         obj_id.language_id, camera_name)
            break
    return detections


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
    if CFG.spot_demo_teleop_find_objects:
        return _teleop_search_for_objects(
            robot,
            localizer,
            object_ids,
            allowed_regions=allowed_regions,
            vlm_predicates=vlm_predicates,
            id2object=id2object,
        )
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
    _find_objects_with_choreographed_moves(
        robot,
        localizer,
        object_ids,
        base_moves,
        hand_moves,
        open_and_close_gripper=False,
        allowed_regions=allowed_regions,
        # FIXME need to pass in VLM predicates and id2object
        vlm_predicates=None,
        id2object=None,
    )


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
                  "become in its view. Hit the 'Enter' key when you're done!\n")
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
