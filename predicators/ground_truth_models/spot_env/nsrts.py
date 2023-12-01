"""Ground-truth NSRTs for the PDDLEnv."""

from typing import Dict, List, Sequence, Set

import numpy as np
from bosdyn.client import math_helpers

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.envs.spot_env import SpotRearrangementEnv, \
    _movable_object_type, _object_to_top_down_geom, get_allowed_map_regions, \
    get_detection_id_for_object
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.settings import CFG
from predicators.spot_utils.perception.object_detection import \
    get_grasp_pixel, get_last_detected_objects
from predicators.spot_utils.perception.spot_cameras import \
    get_last_captured_images
from predicators.spot_utils.utils import _Geom2D, get_spot_home_pose, \
    sample_move_offset_from_target, spot_pose_to_geom2d
from predicators.structs import NSRT, Array, GroundAtom, NSRTSampler, Object, \
    ParameterizedOption, Predicate, State, Type


def _get_collision_geoms_for_nav(state: State) -> List[_Geom2D]:
    """Get all relevant collision geometries for navigating."""
    # We want to consider collisions with all objects that:
    # (1) aren't the robot
    # (2) aren't the floor
    # (3) aren't being currently held.
    collision_geoms = []
    for obj in set(state):
        if obj.type.name != "robot" and obj.name != "floor":
            if obj.type == _movable_object_type:
                if state.get(obj, "held") > 0.5:
                    continue
            collision_geoms.append(_object_to_top_down_geom(obj, state))
    return collision_geoms


def _move_offset_sampler(state: State, robot_obj: Object,
                         obj_to_nav_to: Object, rng: np.random.Generator,
                         min_dist: float, max_dist: float) -> Array:
    """Called by all the different movement samplers."""
    obj_to_nav_to_pos = (state.get(obj_to_nav_to,
                                   "x"), state.get(obj_to_nav_to, "y"))
    spot_pose = utils.get_se3_pose_from_state(state, robot_obj)
    robot_geom = spot_pose_to_geom2d(spot_pose)
    convex_hulls = get_allowed_map_regions()
    collision_geoms = _get_collision_geoms_for_nav(state)
    distance, angle, _ = sample_move_offset_from_target(
        obj_to_nav_to_pos,
        robot_geom,
        collision_geoms,
        rng,
        min_distance=min_dist,
        max_distance=max_dist,
        allowed_regions=convex_hulls,
    )
    return np.array([distance, angle])


def _move_to_body_view_object_sampler(state: State, goal: Set[GroundAtom],
                                      rng: np.random.Generator,
                                      objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, dyaw (to the object you're moving to).
    del goal

    min_dist = 1.7
    max_dist = 1.95

    robot_obj = objs[0]
    obj_to_nav_to = objs[1]
    return _move_offset_sampler(state, robot_obj, obj_to_nav_to, rng, min_dist,
                                max_dist)


def _move_to_hand_view_object_sampler(state: State, goal: Set[GroundAtom],
                                      rng: np.random.Generator,
                                      objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, dyaw (to the object you're moving to).
    del goal

    min_dist = 1.2
    max_dist = 1.5

    robot_obj = objs[0]
    obj_to_nav_to = objs[1]

    return _move_offset_sampler(state, robot_obj, obj_to_nav_to, rng, min_dist,
                                max_dist)


def _move_to_reach_object_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, dyaw (to the object you're moving to).
    del goal

    # NOTE: closer than move_to_view. Important for placing.
    min_dist = 0.0
    max_dist = 0.8

    robot_obj = objs[0]
    obj_to_nav_to = objs[1]
    return _move_offset_sampler(state, robot_obj, obj_to_nav_to, rng, min_dist,
                                max_dist)


def _pick_object_from_top_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
    del state, goal  # not used
    target_obj = objs[1]
    # Special case: if we're running dry, the image won't be used.
    # Randomly sample a pixel.
    if CFG.spot_run_dry:
        # Load the object mask.
        obj_mask_filename = f"grasp_maps/{target_obj.name}-object.npy"
        obj_mask_path = utils.get_env_asset_path(obj_mask_filename)
        obj_mask = np.load(obj_mask_path)
        pixel_choices = np.where(obj_mask)
        num_choices = len(pixel_choices[0])
        choice_idx = rng.choice(num_choices)
        pixel_r = pixel_choices[0][choice_idx]
        pixel_c = pixel_choices[1][choice_idx]
        assert obj_mask[pixel_r, pixel_c]
        params_tuple = (pixel_r, pixel_c, 0.0, 0.0, 0.0, 0.0)
    else:
        # Select the coordinates of a pixel within the image so that
        # we grasp at that pixel!
        target_detection_id = get_detection_id_for_object(target_obj)
        rgbds = get_last_captured_images()
        _, artifacts = get_last_detected_objects()
        hand_camera = "hand_color_image"
        pixel = get_grasp_pixel(rgbds, artifacts, target_detection_id,
                                hand_camera, rng)
        if target_obj.name == "ball":
            rot_quat = math_helpers.Quat.from_pitch(np.pi / 2)
            rot_quat_tuple = (rot_quat.w, rot_quat.x, rot_quat.y, rot_quat.z)
        else:
            rot_quat_tuple = (0.0, 0.0, 0.0, 0.0)
        params_tuple = pixel + rot_quat_tuple

    return np.array(params_tuple)


def _place_object_on_top_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
    # Parameters are relative dx, dy, dz (to surface objects center).
    del goal, rng, state  # randomization coming soon
    dx = 0.025
    dz = 0.05

    # If we're placing the cup, we want to place it further away from
    # the robot so it's solidly on the table in front of it.
    if len(objs) == 3 and objs[1].name == "cup":
        dx = 0.23
        dz = 0.0

    return np.array([dx, 0.0, dz])


def _drop_object_inside_sampler(state: State, goal: Set[GroundAtom],
                                rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
    # Parameters are relative dx, dy, dz to the center of the top of the
    # container.
    del state, goal, rng  # randomization coming soon

    drop_height = 0.5
    dx = 0.0
    dy = 0.0
    if len(objs) == 4 and objs[2].name == "cup":
        drop_height = 0.10
        # We're assuming here that the most recent detection of the container
        # was systematically wrong in the x direction, so we're compensating.
        # The reason for this systematic error is that when the cup is lying
        # flat on the table, the robot selects a point on its bottom as the
        # pose of the cup. We instead want to place at the center of the cup.
        dx = 0.10

    return np.array([dx, dy, drop_height])


def _drag_to_unblock_object_sampler(state: State, goal: Set[GroundAtom],
                                    rng: np.random.Generator,
                                    objs: Sequence[Object]) -> Array:
    # Parameters are relative dx, dy, dyaw to move while holding.
    del state, goal, objs, rng  # randomization coming soon
    return np.array([-1.25, 0.0, np.pi / 3])


def _sweep_into_container_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
    # Parameters are start dx, start dy.
    # NOTE: these parameters may change (need to experiment on robot).
    del state, goal, objs
    if CFG.spot_use_perfect_samplers:
        return np.array([0.0, 0.25])
    dx, dy = rng.uniform(-0.5, 0.5, size=2)
    return np.array([dx, dy])


def _prepare_sweeping_sampler(state: State, goal: Set[GroundAtom],
                              rng: np.random.Generator,
                              objs: Sequence[Object]) -> Array:
    # Parameters are dx, dy, yaw w.r.t. the target object.
    del state, goal, objs, rng  # randomization coming soon

    # Currently assume that the robot is facing the surface in its home pose.
    # Soon, we will change this to actually sample angles of approach and do
    # collision detection.
    home_pose = get_spot_home_pose()

    return np.array([-0.8, -0.4, home_pose.angle])


class SpotEnvsGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the Spot Env."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "spot_cube_env",
            "spot_soda_table_env",
            "spot_soda_bucket_env",
            "spot_soda_chair_env",
            "spot_soda_sweep_env",
            "spot_ball_and_cup_sticky_table_env",
            "spot_brush_shelf_env",
        }

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        env = get_or_create_env(env_name)
        assert isinstance(env, SpotRearrangementEnv)

        nsrts = set()

        operator_name_to_sampler: Dict[str, NSRTSampler] = {
            "MoveToHandViewObject": _move_to_hand_view_object_sampler,
            "MoveToBodyViewObject": _move_to_body_view_object_sampler,
            "MoveToReachObject": _move_to_reach_object_sampler,
            "PickObjectFromTop": _pick_object_from_top_sampler,
            "PickCupToDumpBall": _pick_object_from_top_sampler,
            "PlaceObjectOnTop": _place_object_on_top_sampler,
            "DropObjectInside": _drop_object_inside_sampler,
            "DropObjectInsideContainerOnTop": _drop_object_inside_sampler,
            "DragToUnblockObject": _drag_to_unblock_object_sampler,
            "SweepIntoContainer": _sweep_into_container_sampler,
            "PrepareContainerForSweeping": _prepare_sweeping_sampler,
        }

        for strips_op in env.strips_operators:
            sampler = operator_name_to_sampler[strips_op.name]
            option = options[strips_op.name]
            nsrt = strips_op.make_nsrt(
                option=option,
                option_vars=strips_op.parameters,
                sampler=sampler,
            )
            nsrts.add(nsrt)
        return nsrts
