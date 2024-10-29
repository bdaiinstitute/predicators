"""A perceiver specific to spot envs."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Collection
import json

import imageio.v2 as iio
import numpy as np
from bosdyn.client import math_helpers
from matplotlib import pyplot as plt
import PIL.Image

from predicators import utils
from predicators.envs import BaseEnv, get_or_create_env
from predicators.envs.spot_env import HANDEMPTY_GRIPPER_THRESHOLD, \
    SpotCubeEnv, SpotRearrangementEnv, _drafting_table_type, \
    _PartialPerceptionState, _SpotObservation, in_general_view_classifier, _ALL_TYPES
from predicators.perception.base_perceiver import BasePerceiver
from predicators.settings import CFG
from predicators.spot_utils.utils import _container_type, \
    _immovable_object_type, _movable_object_type, _robot_type, \
    get_allowed_map_regions, load_spot_metadata, object_to_top_down_geom
from predicators.structs import Action, DefaultState, EnvironmentTask, \
    GoalDescription, GroundAtom, Object, Observation, Predicate, \
    SpotActionExtraInfo, State, Task, Video
from predicators.spot_utils.perception.object_perception import vlm


class SpotPerceiver(BasePerceiver):
    """A perceiver specific to spot envs."""

    def __init__(self) -> None:
        super().__init__()
        self._known_object_poses: Dict[Object, math_helpers.SE3Pose] = {}
        self._objects_in_view: Set[Object] = set()
        self._objects_in_hand_view: Set[Object] = set()
        self._objects_in_any_view_except_back: Set[Object] = set()
        self._robot: Optional[Object] = None
        self._nonpercept_atoms: Set[GroundAtom] = set()
        self._nonpercept_predicates: Set[Predicate] = set()
        self._percept_predicates: Set[Predicate] = set()
        self._prev_action: Optional[Action] = None
        self._held_object: Optional[Object] = None
        self._gripper_open_percentage = 0.0
        self._robot_pos: math_helpers.SE3Pose = math_helpers.SE3Pose(
            0, 0, 0, math_helpers.Quat())
        self._lost_objects: Set[Object] = set()
        self._curr_env: Optional[BaseEnv] = None
        self._waiting_for_observation = True
        self._ordered_objects: List[Object] = []  # list of all known objects
        # Keep track of objects that are contained (out of view) in another
        # object, like a bag or bucket. This is important not only for gremlins
        # but also for small changes in the container's perceived pose.
        self._container_to_contained_objects: Dict[Object, Set[Object]] = {}
        # Load static, hard-coded features of objects, like their shapes.
        meta = load_spot_metadata()
        self._static_object_features = meta.get("static-object-features", {})

    @classmethod
    def get_name(cls) -> str:
        return "spot_perceiver"

    def reset(self, env_task: EnvironmentTask) -> Task:
        # Unless dry running, don't reset after the first time.
        if self._waiting_for_observation or CFG.spot_run_dry:
            self._waiting_for_observation = True
            self._curr_env = get_or_create_env(CFG.env)
            assert isinstance(self._curr_env, SpotRearrangementEnv)
            self._known_object_poses = {}
            self._objects_in_view = set()
            self._objects_in_hand_view = set()
            self._objects_in_any_view_except_back = set()
            self._robot = None
            self._nonpercept_atoms = set()
            self._nonpercept_predicates = set()
            self._percept_predicates = self._curr_env.percept_predicates
            self._held_object = None
            self._gripper_open_percentage = 0.0
            self._robot_pos = math_helpers.SE3Pose(0, 0, 0,
                                                   math_helpers.Quat())
            self._lost_objects = set()
            self._container_to_contained_objects = {}
        self._prev_action = None  # already processed at the end of the cycle
        init_state = self._create_state()
        goal = self._create_goal(init_state, env_task.goal_description)
        return Task(init_state, goal)

    def update_perceiver_with_action(self, action: Action) -> None:
        # NOTE: we need to keep track of the previous action
        # because the step function (where we need knowledge
        # of the previous action) occurs *after* the action
        # has already been taken.
        self._prev_action = action

    def step(self, observation: Observation) -> State:
        self._update_state_from_observation(observation)
        # Update the curr held item when applicable.
        assert self._curr_env is not None
        if self._prev_action is not None:
            assert isinstance(self._prev_action.extra_info,
                              SpotActionExtraInfo)
            controller_name = self._prev_action.extra_info.action_name
            objects = self._prev_action.extra_info.operator_objects
            logging.info(
                f"[Perceiver] Previous action was {controller_name}{objects}.")
            # The robot is always the 0th argument of an
            # operator!
            if "pick" in controller_name.lower():
                if self._held_object is not None:
                    assert CFG.spot_run_dry
                else:
                    # We know that the object that we attempted to grasp was
                    # the second argument to the controller.
                    object_attempted_to_grasp = objects[1]
                    # Remove from contained objects.
                    for contained in self.\
                        _container_to_contained_objects.values():
                        contained.discard(object_attempted_to_grasp)
                    # We only want to update the holding item id feature
                    # if we successfully picked something.
                    if self._gripper_open_percentage > \
                        HANDEMPTY_GRIPPER_THRESHOLD:
                        self._held_object = object_attempted_to_grasp
                    else:
                        # We lost the object!
                        logging.info("[Perceiver] Object was lost!")
                        self._lost_objects.add(object_attempted_to_grasp)
            elif any(n in controller_name.lower() for n in
                     ["place", "drop", "preparecontainerforsweeping", "drag"]):
                self._held_object = None
                # Check if the item we just placed is in view. It needs to
                # be in view to assess whether it was placed correctly.
                robot, obj = objects[:2]
                state = self._create_state()
                is_in_view = in_general_view_classifier(state, [robot, obj])
                if not is_in_view:
                    # We lost the object!
                    logging.info("[Perceiver] Object was lost!")
                    self._lost_objects.add(obj)
            elif any(n in controller_name.lower()
                     for n in ["sweepintocontainer", "sweeptwoobjects"]):
                robot = objects[0]
                state = self._create_state()
                if controller_name.lower() == "sweepintocontainer":
                    objs = {objects[2]}
                else:
                    assert controller_name.lower().startswith("sweeptwoobject")
                    objs = {objects[2], objects[3]}
                for o in objs:
                    is_in_view = in_general_view_classifier(state, [robot, o])
                    if not is_in_view:
                        # We lost the object!
                        logging.info("[Perceiver] Object was lost!")
                        self._lost_objects.add(o)
            else:
                # Ensure the held object is reset if the hand is empty.
                prev_held_object = self._held_object
                if self._gripper_open_percentage <= HANDEMPTY_GRIPPER_THRESHOLD:
                    self._held_object = None
                    # This can only happen if the item was dropped during
                    # something other than a place.
                    if prev_held_object is not None:
                        # We lost the object!
                        logging.info("[Perceiver] An object was lost: "
                                     f"{prev_held_object} was lost!")
                        self._lost_objects.add(prev_held_object)

        return self._create_state()

    def _update_state_from_observation(self, observation: Observation) -> None:
        assert isinstance(observation, _SpotObservation)
        # If a container is being updated, change the poses for contained
        # objects.
        for container in observation.objects_in_view:
            if container not in self._container_to_contained_objects:
                continue
            if container not in self._known_object_poses:
                continue
            last_container_pose = self._known_object_poses[container]
            new_container_pose = observation.objects_in_view[container]
            dx, dy, dz = np.subtract(new_container_pose, last_container_pose)
            for obj in self._container_to_contained_objects[container]:
                x, y, z = self._known_object_poses[
                    obj].x, self._known_object_poses[
                        obj].y, self._known_object_poses[obj].z
                new_obj_pose = (x + dx, y + dy, z + dz)
                self._known_object_poses[obj] = math_helpers.SE3Pose(
                    new_obj_pose[0], new_obj_pose[1], new_obj_pose[2],
                    self._known_object_poses[obj].rot)
        self._waiting_for_observation = False
        self._robot = observation.robot
        self._known_object_poses.update(observation.objects_in_view)
        self._objects_in_view = set(observation.objects_in_view)
        self._objects_in_hand_view = observation.objects_in_hand_view
        self._objects_in_any_view_except_back = \
            observation.objects_in_any_view_except_back
        self._nonpercept_atoms = observation.nonpercept_atoms
        self._nonpercept_predicates = observation.nonpercept_predicates
        self._gripper_open_percentage = observation.gripper_open_percentage
        self._robot_pos = observation.robot_pos
        for obj in observation.objects_in_view:
            self._lost_objects.discard(obj)

        # NOTE: This is only used when using VLM for predicate evaluation
        # NOTE: Performance aspect should be considered later
        if CFG.spot_vlm_eval_predicate:
            # Add current Spot images to the state if needed
            self._camera_images = observation.images
            self._vlm_atom_dict = observation.vlm_atom_dict
            self._vlm_predicates = observation.vlm_predicates
        else:
            self._camera_images = None
            self._vlm_atom_dict = None
            self._vlm_predicates = None

    def _create_state(self) -> State:
        if self._waiting_for_observation:
            return DefaultState
        # Build the continuous part of the state.
        assert self._robot is not None
        state_dict = {
            self._robot: {
                "gripper_open_percentage": self._gripper_open_percentage,
                "x": self._robot_pos.x,
                "y": self._robot_pos.y,
                "z": self._robot_pos.z,
                "qw": self._robot_pos.rot.w,
                "qx": self._robot_pos.rot.x,
                "qy": self._robot_pos.rot.y,
                "qz": self._robot_pos.rot.z,
            },
        }
        # Add new objects to the list of known objects.
        known_objs = set(self._ordered_objects)
        for obj in sorted(set(self._known_object_poses) - known_objs):
            self._ordered_objects.append(obj)
        for obj, pose in self._known_object_poses.items():
            object_id = self._ordered_objects.index(obj)
            state_dict[obj] = {
                "x": pose.x,
                "y": pose.y,
                "z": pose.z,
                "qw": pose.rot.w,
                "qx": pose.rot.x,
                "qy": pose.rot.y,
                "qz": pose.rot.z,
                "object_id": object_id,
            }
            # Add static object features.
            static_feats = self._static_object_features.get(obj.name, {})
            state_dict[obj].update(static_feats)
            # Add initial features for movable objects.
            if obj.is_instance(_movable_object_type):
                # Detect if the object is in (hand) view currently.
                if obj in self._objects_in_hand_view:
                    in_hand_view_val = 1.0
                else:
                    in_hand_view_val = 0.0
                state_dict[obj]["in_hand_view"] = in_hand_view_val
                if obj in self._objects_in_any_view_except_back:
                    in_view_val = 1.0
                else:
                    in_view_val = 0.0
                state_dict[obj]["in_view"] = in_view_val
                # Detect if we have lost the tool.
                if obj in self._lost_objects:
                    lost_val = 1.0
                else:
                    lost_val = 0.0
                state_dict[obj]["lost"] = lost_val
                if obj == self._held_object:
                    held_val = 1.0
                else:
                    held_val = 0.0
                state_dict[obj]["held"] = held_val
        # Construct a regular state before adding atoms.
        percept_state = utils.create_state_from_dict(state_dict)
        # Prepare the simulator state.
        simulator_state = {
            "predicates": self._nonpercept_predicates,
            "atoms": self._nonpercept_atoms,
        }

        # Uncomment for debugging.
        # logging.info("Percept state:")
        # logging.info(percept_state.pretty_str())
        # logging.info("Percept atoms:")
        # atom_str = "\n".join(
        #     map(
        #         str,
        #         sorted(utils.abstract(percept_state,
        #                               self._percept_predicates))))
        # logging.info(atom_str)
        # logging.info("Simulator state:")
        # logging.info(simulator_state)

        # Prepare the current images from observation
        camera_images = self._camera_images if CFG.spot_vlm_eval_predicate else None

        # Now finish the state.
        state = _PartialPerceptionState(
            percept_state.data,
            simulator_state=simulator_state,
            camera_images=camera_images,
            visible_objects=self._objects_in_view,
            vlm_atom_dict=self._vlm_atom_dict,
            vlm_predicates=self._vlm_predicates,
        )
        # DEBUG - look into dataclass field init - why warning

        return state

    def _create_goal(self, state: State,
                     goal_description: GoalDescription) -> Set[GroundAtom]:
        # Unfortunate hack to deal with the fact that the state is actually
        # not yet set. Hopefully one day other cleanups will enable cleaning.
        assert self._curr_env is not None
        pred_name_to_pred = {p.name: p for p in self._curr_env.predicates}
        if goal_description == "put the cube on the sticky table":
            assert isinstance(self._curr_env, SpotCubeEnv)
            cube = Object("cube", _movable_object_type)
            target = Object("sticky_table", _immovable_object_type)
            On = pred_name_to_pred["On"]
            return {GroundAtom(On, [cube, target])}
        if goal_description == "put the soda on the smooth table":
            can = Object("soda_can", _movable_object_type)
            smooth = Object("smooth_table", _immovable_object_type)
            On = pred_name_to_pred["On"]
            return {GroundAtom(On, [can, smooth])}
        if goal_description == "put the soda in the bucket":
            can = Object("soda_can", _movable_object_type)
            bucket = Object("bucket", _container_type)
            Inside = pred_name_to_pred["Inside"]
            return {GroundAtom(Inside, [can, bucket])}
        if goal_description == "pick up the soda can":
            robot = Object("robot", _robot_type)
            can = Object("soda_can", _movable_object_type)
            Holding = pred_name_to_pred["Holding"]
            return {GroundAtom(Holding, [robot, can])}
        if goal_description == "pick up the bucket":
            robot = Object("robot", _robot_type)
            bucket = Object("bucket", _container_type)
            Holding = pred_name_to_pred["Holding"]
            return {GroundAtom(Holding, [robot, bucket])}
        if goal_description == "put the soda in the bucket and hold the brush":
            robot = Object("robot", _robot_type)
            can = Object("soda_can", _movable_object_type)
            bucket = Object("bucket", _container_type)
            brush = Object("brush", _movable_object_type)
            Inside = pred_name_to_pred["Inside"]
            Holding = pred_name_to_pred["Holding"]
            return {
                GroundAtom(Inside, [can, bucket]),
                GroundAtom(Holding, [robot, brush])
            }
        if goal_description == "unblock the train_toy":
            train_toy = Object("train_toy", _movable_object_type)
            NotBlocked = pred_name_to_pred["NotBlocked"]
            return {
                GroundAtom(NotBlocked, [train_toy]),
            }
        if goal_description == "get the objects into the bucket":
            train_toy = Object("train_toy", _movable_object_type)
            football = Object("football", _movable_object_type)
            bucket = Object("bucket", _container_type)
            Inside = pred_name_to_pred["Inside"]
            return {
                GroundAtom(Inside, [train_toy, bucket]),
                GroundAtom(Inside, [football, bucket]),
            }
        if goal_description == "get the objects onto the table":
            train_toy = Object("train_toy", _movable_object_type)
            football = Object("football", _movable_object_type)
            black_table = Object("black_table", _immovable_object_type)
            On = pred_name_to_pred["On"]
            return {
                GroundAtom(On, [train_toy, black_table]),
                GroundAtom(On, [football, black_table]),
            }
        if goal_description == "get the objects out of the bucket":
            train_toy = Object("train_toy", _movable_object_type)
            football = Object("football", _movable_object_type)
            bucket = Object("bucket", _container_type)
            NotInsideAnyContainer = pred_name_to_pred["NotInsideAnyContainer"]
            return {
                GroundAtom(NotInsideAnyContainer, [train_toy]),
                GroundAtom(NotInsideAnyContainer, [football]),
            }
        if goal_description == "get the objects into the bucket and put " + \
            "the bucket on the shelf":
            train_toy = Object("train_toy", _movable_object_type)
            football = Object("football", _movable_object_type)
            bucket = Object("bucket", _container_type)
            shelf = Object("shelf1", _immovable_object_type)
            On = pred_name_to_pred["On"]
            Inside = pred_name_to_pred["Inside"]
            return {
                GroundAtom(Inside, [train_toy, bucket]),
                GroundAtom(Inside, [football, bucket]),
                GroundAtom(On, [bucket, shelf])
            }
        if goal_description == "get the train_toy and football onto the table":
            train_toy = Object("train_toy", _movable_object_type)
            football = Object("football", _movable_object_type)
            table = Object("black_table", _immovable_object_type)
            On = pred_name_to_pred["On"]
            return {
                GroundAtom(On, [train_toy, table]),
                GroundAtom(On, [football, table]),
            }
        if goal_description == "get the train_toy into the bucket":
            train_toy = Object("train_toy", _movable_object_type)
            bucket = Object("bucket", _container_type)
            Inside = pred_name_to_pred["Inside"]
            return {
                GroundAtom(Inside, [train_toy, bucket]),
            }
        if goal_description == "place bucket such that train_toy can be " + \
            "swept into it":
            bucket = Object("bucket", _container_type)
            train_toy = Object("train_toy", _movable_object_type)
            black_table = Object("black_table", _immovable_object_type)
            ContainerReadyForSweeping = pred_name_to_pred[
                "ContainerReadyForSweeping"]
            return {
                GroundAtom(ContainerReadyForSweeping, [bucket, black_table]),
            }
        if goal_description == "get the football into the bucket":
            football = Object("football", _movable_object_type)
            bucket = Object("bucket", _container_type)
            Inside = pred_name_to_pred["Inside"]
            return {
                GroundAtom(Inside, [football, bucket]),
            }
        if goal_description == "get the brush into the bucket":
            brush = Object("brush", _movable_object_type)
            bucket = Object("bucket", _container_type)
            Inside = pred_name_to_pred["Inside"]
            return {
                GroundAtom(Inside, [brush, bucket]),
            }
        if goal_description == "get the bucket on the table":
            bucket = Object("bucket", _container_type)
            table = Object("black_table", _immovable_object_type)
            On = pred_name_to_pred["On"]
            return {
                GroundAtom(On, [bucket, table]),
            }
        if goal_description == "get the brush on the table":
            brush = Object("brush", _movable_object_type)
            table = Object("black_table", _immovable_object_type)
            On = pred_name_to_pred["On"]
            return {
                GroundAtom(On, [brush, table]),
            }
        if goal_description == "put the ball on the table":
            ball = Object("ball", _movable_object_type)
            drafting_table = Object("drafting_table", _drafting_table_type)
            On = pred_name_to_pred["On"]
            return {
                GroundAtom(On, [ball, drafting_table]),
            }
        if goal_description == "put the brush in the second shelf":
            brush = Object("brush", _movable_object_type)
            shelf = Object("shelf1", _immovable_object_type)
            On = pred_name_to_pred["On"]
            return {
                GroundAtom(On, [brush, shelf]),
            }
        if goal_description == "put the bucket in the second shelf":
            bucket = Object("bucket", _container_type)
            shelf = Object("shelf1", _immovable_object_type)
            On = pred_name_to_pred["On"]
            return {
                GroundAtom(On, [bucket, shelf]),
            }
        if goal_description == "pick up the brush":
            robot = Object("robot", _robot_type)
            brush = Object("brush", _movable_object_type)
            Holding = pred_name_to_pred["Holding"]
            return {
                GroundAtom(Holding, [robot, brush]),
            }
        if goal_description == "pick up the red block":
            robot = Object("robot", _robot_type)
            block = Object("red_block", _movable_object_type)
            Holding = pred_name_to_pred["Holding"]
            return {GroundAtom(Holding, [robot, block])}
        if goal_description == "pick the red block into the green bowl":
            block = Object("red_block", _movable_object_type)
            bowl = Object("green_bowl", _container_type)
            Inside = pred_name_to_pred["Inside"]
            return {
                GroundAtom(Inside, [block, bowl]),
            }
        if goal_description == "put the red block into the plastic bin on floor":
            block = Object("red_block", _movable_object_type)
            bin = Object("plastic_bin", _container_type)
            Inside = pred_name_to_pred["Inside"]
            return {
                GroundAtom(Inside, [block, bin]),
            }
        if goal_description == "put the red block into the box on floor":
            block = Object("red_block", _movable_object_type)
            bin = Object("plastic_bin", _container_type)
            Inside = pred_name_to_pred["Inside"]
            return {
                GroundAtom(Inside, [block, bin]),
            }
        if goal_description == "put the red block into the plastic bin on floor":
            block = Object("red_block", _movable_object_type)
            box = Object("plastic_bin", _container_type)
            Inside = pred_name_to_pred["Inside"]
            return {
                GroundAtom(Inside, [block, box]),
            }
        if goal_description == "put the red block on table into the green bowl on floor":
            block = Object("red_block", _movable_object_type)
            bowl = Object("green_bowl", _container_type)
            Inside = pred_name_to_pred["Inside"]
            return {
                GroundAtom(Inside, [block, bowl]),
            }
        if goal_description == "put the red block on table into the plastic bin on floor":
            block = Object("red_block", _movable_object_type)
            bin = Object("plastic_bin", _container_type)
            Inside = pred_name_to_pred["Inside"]
            return {
                GroundAtom(Inside, [block, bin]),
            }
        if goal_description == "put the empty cup into the box, otherwise stay on floor":
            # TODO can we do: IMPLY AND OR?
            cup = Object("empty_cup", _movable_object_type)
            box = Object("box", _container_type)
            floor = Object("floor", _immovable_object_type)
            Inside = pred_name_to_pred["Inside"]
            On = pred_name_to_pred["On"]
            return {
                GroundAtom(Inside, [cup, box]),
                GroundAtom(On, [cup, floor])
            }
        if goal_description == "view the object from top":
            # TODO test
            robot = Object("robot", _robot_type)
            object_to_view = Object("red_block", _movable_object_type)
            InHandViewFromTop = pred_name_to_pred["InHandViewFromTop"]
            return {
                GroundAtom(InHandViewFromTop, [robot, object_to_view]),
            }
        if goal_description == "know container not as empty":
            # container = Object("container", _container_type)
            cup = Object("cup", _container_type)
            ContainingFoodKnown = pred_name_to_pred["ContainingFoodKnown"]
            ContainingFood = pred_name_to_pred["ContainingFood"]
            return {
                GroundAtom(ContainingFoodKnown, [cup]),
                GroundAtom(ContainingFood, [cup]),
            }
        if goal_description == "place empty cup into the box":
            cup = Object("cup", _container_type)
            plastic_bin = Object("plastic_bin", _container_type)
            ContainingFoodKnown = pred_name_to_pred["ContainingFoodKnown"]
            NotContainingFood = pred_name_to_pred["NotContainingFood"]
            Inside = pred_name_to_pred["Inside"]
            if state.data == {}:
                return {
                    GroundAtom(ContainingFoodKnown, [cup]),
                    GroundAtom(NotContainingFood, [cup]),
                    #GroundAtom(Inside, [cup, plastic_bin]),
                }
            object_name_to_object = {}
            return self._parse_vlm_goal_from_state(state, goal_description, object_name_to_object)
        if goal_description == "know container as empty":
            cup = Object("cup", _container_type)
            ContainingFoodKnown = pred_name_to_pred["ContainingFoodKnown"]
            NotContainingFood = pred_name_to_pred["NotContainingFood"]
            return {
                GroundAtom(ContainingFoodKnown, [cup]),
                GroundAtom(NotContainingFood, [cup]),
            }
        if goal_description == "put the cup into the plastic bin on floor":
            cup = Object("cup", _container_type)
            plastic_bin = Object("plastic_bin", _container_type)
            Inside = pred_name_to_pred["Inside"]
            return {
                GroundAtom(Inside, [cup, plastic_bin]),
            }
        if goal_description == "setup sweeping":
            robot = Object("robot", _robot_type)
            brush = Object("brush", _movable_object_type)
            bucket = Object("bucket", _container_type)
            black_table = Object("black_table", _immovable_object_type)
            football = Object("football", _movable_object_type)
            train_toy = Object("train_toy", _movable_object_type)
            On = pred_name_to_pred["On"]
            FitsInXY = pred_name_to_pred["FitsInXY"]
            IsSemanticallyGreaterThan = pred_name_to_pred[
                "IsSemanticallyGreaterThan"]
            IsPlaceable = pred_name_to_pred["IsPlaceable"]
            HasFlatTopSurface = pred_name_to_pred["HasFlatTopSurface"]
            RobotReadyForSweeping = pred_name_to_pred["RobotReadyForSweeping"]
            NotBlocked = pred_name_to_pred["NotBlocked"]
            TopAbove = pred_name_to_pred["TopAbove"]
            Holding = pred_name_to_pred["Holding"]
            ContainerReadyForSweeping = pred_name_to_pred[
                "ContainerReadyForSweeping"]
            IsSweeper = pred_name_to_pred["IsSweeper"]
            return {
                GroundAtom(On, [football, black_table]),
                GroundAtom(FitsInXY, [football, bucket]),
                GroundAtom(FitsInXY, [train_toy, bucket]),
                GroundAtom(IsSemanticallyGreaterThan, [train_toy, football]),
                GroundAtom(IsPlaceable, [train_toy]),
                GroundAtom(HasFlatTopSurface, [black_table]),
                GroundAtom(RobotReadyForSweeping, [robot, train_toy]),
                GroundAtom(NotBlocked, [train_toy]),
                GroundAtom(On, [train_toy, black_table]),
                GroundAtom(TopAbove, [black_table, bucket]),
                GroundAtom(IsPlaceable, [football]),
                GroundAtom(Holding, [robot, brush]),
                GroundAtom(ContainerReadyForSweeping, [bucket, black_table]),
                GroundAtom(IsSweeper, [brush])
            }
        raise NotImplementedError("Unrecognized goal description")

    def render_mental_images(self, observation: Observation,
                             env_task: EnvironmentTask) -> Video:
        if self._waiting_for_observation:
            return []
        state = self._create_state()

        assert isinstance(self._curr_env, SpotRearrangementEnv)
        x_lb = self._curr_env.render_x_lb
        x_ub = self._curr_env.render_x_ub
        y_lb = self._curr_env.render_y_lb
        y_ub = self._curr_env.render_y_ub
        figsize = (x_ub - x_lb, y_ub - y_lb)
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        # Draw the allowed regions.
        allowed_regions = get_allowed_map_regions()
        for region in allowed_regions:
            ax.triplot(region.points[:, 0],
                       region.points[:, 1],
                       region.simplices,
                       linestyle="--",
                       color="gray")
            ax.plot(region.points[:, 0],
                    region.points[:, 1],
                    'o',
                    color="gray")
        # Draw the robot as an arrow.
        assert self._robot is not None
        robot_pose = utils.get_se3_pose_from_state(state, self._robot)
        robot_x = robot_pose.x
        robot_y = robot_pose.y
        robot_yaw = robot_pose.rot.to_yaw()
        arrow_length = (x_ub - x_lb) / 20.0
        head_width = arrow_length / 3
        robot_dx = arrow_length * np.cos(robot_yaw)
        robot_dy = arrow_length * np.sin(robot_yaw)
        plt.arrow(robot_x,
                  robot_y,
                  robot_dx,
                  robot_dy,
                  color="red",
                  head_width=head_width)
        # Draw the other objects.
        for obj in state:
            if obj == self._robot:
                continue
            # Don't plot the floor because it's enormous.
            if obj.name == "floor":
                continue
            geom = object_to_top_down_geom(obj, state)
            geom.plot(ax,
                      label=obj.name,
                      facecolor=(0.0, 0.0, 0.0, 0.0),
                      edgecolor="black")
            assert isinstance(geom, (utils.Rectangle, utils.Circle))
            text_pos = (geom.x, geom.y)
            ax.text(text_pos[0],
                    text_pos[1],
                    obj.name,
                    color='white',
                    fontsize=12,
                    fontweight='bold',
                    bbox=dict(facecolor="gray", edgecolor="gray", alpha=0.5))
        ax.set_xlim(x_lb, x_ub)
        ax.set_ylim(y_lb, y_ub)
        plt.tight_layout()
        img = utils.fig2data(fig, CFG.render_state_dpi)
        # Save the most recent top-down view at every time step.
        outdir = Path(CFG.spot_perception_outdir)
        time_str = time.strftime("%Y%m%d-%H%M%S")
        outfile = outdir / f"mental_top_down_{time_str}.png"
        iio.imsave(outfile, img)
        logging.info(f"Wrote out to {outfile}")
        plt.close()
        return [img]

    def _get_language_goal_prompt_prefix(self,
                                         object_names: Collection[str]) -> str:
        # pylint:disable=line-too-long
        available_predicates = ", ".join([p for p in sorted([pred.pretty_str()[1] for pred in self._curr_env.goal_predicates])])
        available_object_types = ", ".join(sorted([t.name for t in _ALL_TYPES]))
        # We could extract the object names, but this is simpler.
        prompt = f"""# The available predicates are: {available_predicates}
# The available object types are: {available_object_types}
# Use the available predicates and object types to convert natural language goals into JSON goals.
        
# I want a sandwich with a patty, cheese, and lettuce, and get ready to give me a glass of milk.
{{"Holding": [["robot", "milk"]], "On": [["bread0", "board"], ["bread1", "lettuce0"], ["lettuce0", "cheese0"], ["cheese0", "patty0"], ["patty0", "bread0"]]}}
"""
        return prompt

    def _parse_vlm_goal_from_state(
            self, state: State, language_goal: str,
            id_to_obj: Dict[str, Object]) -> Set[GroundAtom]:
        """Helper for parsing language-based goals from JSON task specs."""
        ###
        language_goal = 'place empty cups into the plastic_bin'
        ###
        object_names = set(id_to_obj)
        prompt_prefix = self._get_language_goal_prompt_prefix(object_names)
        prompt = prompt_prefix + f"\n# {language_goal}"
        image_list = [
            PIL.Image.fromarray(v.rotated_rgb) for _, v in state.camera_images.items()
        ]
        responses = vlm.sample_completions(
                prompt=prompt,
                imgs=image_list,
                temperature=0.1,
                seed=int(time.time()),
                num_completions=1,
            )
        response = responses[0]
        # Currently assumes that the LLM is perfect. In the future, will need
        # to handle various errors and perhaps query the LLM for multiple
        # responses until we find one that can be parsed.
        try:
            goal_spec = json.loads(response)
        except json.JSONDecodeError as e:
            goal_spec = json.loads(response.replace('`', '').replace('json', ''))

        for pred, args in goal_spec.items():
            for arg in args:
                for obj_name in arg:
                    if 'robot' in obj_name:
                        id_to_obj[obj_name] = Object(obj_name, _robot_type)
                    elif any([name in obj_name for name in ['cup', 'box', 'bin']]):
                        id_to_obj[obj_name] = Object(obj_name, _container_type)
                    else:
                        id_to_obj[obj_name] = Object(obj_name, _movable_object_type)

        new_goal = self._curr_env._parse_goal_from_json(goal_spec, id_to_obj)
        return new_goal

