"""Mock environment for Spot robot.

This module provides a mock environment for the Spot robot that simulates:
- States and transitions for pick-and-place tasks
- RGB-D observations with object detections
- Gripper state and object tracking

The environment stores its data (graph, images, etc.) in a directory specified by CFG.mock_env_data_dir.
If not specified, it defaults to "mock_env_data". The data includes:
- graph.json: Contains state transitions and observations
- images/: Directory containing RGB-D images for each state

Configuration:
    mock_env_data_dir (str): Directory to store environment data (default: "mock_env_data")
"""

import logging
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator, Sequence, ClassVar, Union, Container, cast
from itertools import product

import numpy as np
from gym.spaces import Box
from rich.table import Table
from rich.logging import RichHandler
from rich.console import Console
from rich import print
import yaml

from predicators.envs import BaseEnv
from predicators.structs import Action, GoalDescription, State, Object, Type, EnvironmentTask, Video, Image
from predicators.structs import LiftedAtom, STRIPSOperator, Variable, Predicate, GroundAtom, GroundTruthPredicate, VLMPredicate, VLMGroundAtom, ParameterizedOption, NSRT, _GroundNSRT
from predicators.settings import CFG
from predicators.utils import get_object_combinations
from predicators.spot_utils.mock_env.mock_env_utils import (
    PREDICATES, PREDICATES_WITH_VLM, VLM_PREDICATES, TYPES, BELIEF_PREDICATES, GOAL_PREDICATES,
    _robot_type, _base_object_type, _movable_object_type, _container_type,
    _immovable_object_type, _SavedMockSpotObservation, _MockSpotObservation,
    _NotBlocked, _NotHolding, _Reachable, _InHandView, _InView, _RobotReadyForSweeping,
    _HandEmpty, _NotInsideAnyContainer, _Inside, _DrawerOpen, _DrawerClosed,
    _ContainingWaterKnown, _Known_ContainerEmpty, _NEq, _On, _TopAbove, _FitsInXY,
    _InHandViewFromTop, _Holding, _Blocking, _ContainerReadyForSweeping, _IsPlaceable,
    _IsNotPlaceable, _IsSweeper, _HasFlatTopSurface, _ContainingWaterUnknown,
    _ContainingWater, _NotContainingWater, _ContainerEmpty, _Unknown_ContainerEmpty,
    _BelieveTrue_ContainerEmpty, _BelieveFalse_ContainerEmpty
)


def get_vlm_atom_combinations_test(objects: Set[Object],
                         preds: Set[VLMPredicate]) -> Set[VLMGroundAtom]:
    """Get all possible combinations of objects for each predicate.
    
    Debug version of get_vlm_atom_combinations.
    """
    atoms = set()
    for pred in preds:
        param_objects = get_object_combinations(objects, pred.types)
        for objs in param_objects:
            atoms.add(VLMGroundAtom(pred, objs))
    return atoms


class MockSpotEnv(BaseEnv):
    """Mock environment for Spot robot.
    
    This environment is a POMDP where:
    - States are latent (we don't know actual states and don't need to know)
    - Observations are RGB-D images + gripper state + object detections
    - Actions can succeed or fail based on available images
    
    The environment stores its data in a directory specified by CFG.mock_env_data_dir.
    This includes:
    - State transition graph (graph.json)
    - RGB-D images for each state (images/)
    - Observation metadata (gripper state, objects in view/hand)
    
    Args:
        use_gui (bool): Whether to use GUI for visualization. Defaults to True.
    """
    @classmethod
    def get_name(cls) -> str:
        """Get the name of this environment."""
        return "mock_spot"
    
    preset_data_dir: Optional[str] = None

    def __init__(self, use_gui: bool = True) -> None:
        """Initialize the mock Spot environment."""
        super().__init__(use_gui)
        
        # Get data directory from config
        data_dir = self.preset_data_dir if self.preset_data_dir is not None else (CFG.mock_env_data_dir if hasattr(CFG, "mock_env_data_dir") else "mock_env_data")
        
        # Create data directories
        self._data_dir = Path(data_dir)
        self._images_dir = self._data_dir / "images"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Initialized MockSpotEnv with data_dir: %s", data_dir)
        
        # Initialize environment state
        self._current_state_id: Optional[str] = None
        self._gripper_open: bool = True
        self._objects_in_hand: Set[Object] = set()
        
        # Create constant objects
        self._spot_object = Object("robot", _robot_type)
        self._objects: Dict[str, Object] = {"robot": self._spot_object}
        
        # Load or initialize transition graph
        self._str_transitions: List[Tuple[str, Dict[str, Any], str]] = []  # (source_id, op_dict, dest_id)
        self._observations: Dict[str, _SavedMockSpotObservation] = {}  # state_id -> observation
        self._transition_metadata: Dict[str, Any] = {}
        
        # Load transitions and objects
        self._load_transitions()

        # Create operators
        self._operators = list(self._create_operators())
        
        # NOTE: to update this from language
        self.goal_atoms = None
        
    def _load_transitions(self) -> None:
        """Load the transition system.
        
        This loads:
        1. Objects and their types
        2. States and their atoms
        3. Transitions between states
        4. System metadata
        """
        try:
            with open(self._data_dir / "transition_system.yaml") as f:
                system_data = yaml.safe_load(f)
                
                # Load objects if not already loaded
                if not hasattr(self, '_loaded_objects'):
                    for obj_name, obj_data in system_data["objects"].items():
                        # Skip robot object as it's already created
                        if obj_name == "robot":
                            continue
                        # Get type from hierarchy
                        type_name = obj_data["type"]
                        type_obj = None
                        for t in TYPES:
                            if t.name == type_name:
                                type_obj = t
                                break
                        if type_obj is not None:
                            self._objects[obj_name] = Object(obj_name, type_obj)
                    self._loaded_objects = True
                
                # Load transitions in new format
                self._str_transitions = [
                    (t["source"], t["operator"], t["target"])
                    for t in system_data["transitions"]
                ]
                
                # Store metadata for potential use
                self._transition_metadata = system_data["metadata"]
                
        except FileNotFoundError:
            logging.warning("No transition system found at %s", self._data_dir / "transition_system.yaml")
            self._str_transitions = []
            self._transition_metadata = {}

    def _load_state(self, state_id: str) -> _SavedMockSpotObservation:
        """Load a state's observation."""
        return _SavedMockSpotObservation.load_state(state_id, self._images_dir, self._objects)
        
    def step(self, action: Action) -> _MockSpotObservation:
        """Take an action in the environment.
        
        The environment can transition in several ways:
        1. Normal manipulation operators: Load next state from transitions
        2. Observation operators: State doesn't change, only belief updates
        3. Invalid actions: Stay in current state
        4. States without images: Stay in current state
        
        Args:
            action: Action to take, containing operator name and objects
            
        Returns:
            Next observation
        """
        if self._current_observation is None:
            raise RuntimeError("Environment not reset - call reset() first")

        # Get current state ID and action info
        current_state_id = self._current_observation.state_id
        
        # Extract operator name and objects from action
        # Access through extra_info which is guaranteed to exist
        info = action.extra_info or {}
        op_name = info.get("operator_name")
        op_objects = info.get("objects", [])
        
        if op_name is None:
            # Invalid action format - stay in current state
            logging.warning(f"Invalid action format: {action}")
            return self._current_observation
            
        # Check if this is an observation operator
        is_observation_op = op_name in {
            "ObserveCupContent",
            "ObserveDrawerContentFindEmpty",
            "ObserveDrawerContentFindNotEmpty"
        } or op_name.startswith("Observe")
        
        if is_observation_op:
            # For observation operators, state doesn't change
            # Only update beliefs in current observation
            # The observation update should be handled by the agent's `perceiver`
            # when setting up the environment data
            return self._current_observation
            
        # Get next state ID from transitions
        next_state_id = None
        for source_id, op_dict, dest_id in self._str_transitions:
            # Compare operator name and objects
            if (source_id == current_state_id and
                isinstance(op_dict, dict) and
                op_dict.get("name") == op_name and
                op_dict.get("objects", []) == [obj.name for obj in op_objects]):
                next_state_id = dest_id
                break
                
        if next_state_id is None:
            # No valid transition found - stay in current state
            logging.warning(
                f"No valid transition found for action {op_name} with objects {op_objects} "
                f"from state {current_state_id}"
            )
            return self._current_observation
            
        # Try to load next observation
        try:
            # Convert objects set to dict for load_state
            objects_dict = {obj.name: obj for obj in self.objects}
            loaded_obs = _SavedMockSpotObservation.load_state(
                next_state_id,
                self._images_dir,
                objects_dict
            )
            self._current_observation = _MockSpotObservation.init_from_saved(
                loaded_obs,
                object_dict=objects_dict,
                vlm_atom_dict=None,  # Will be populated if needed
                vlm_predicates=VLM_PREDICATES if CFG.mock_env_vlm_eval_predicate else None
            )
            return self._current_observation
            
        except FileNotFoundError:
            # No observation data for next state - stay in current state
            logging.warning(f"No observation data found for state {next_state_id}")
            return self._current_observation

    def reset(self, train_or_test: str, task_idx: int) -> _MockSpotObservation:
        """Reset the environment to the initial state."""
        
        self._current_state_id = "0"
        # Create mock observation
        # Convert objects set to dict for load_state
        objects_dict = {obj.name: obj for obj in self.objects}
        loaded_obs = _SavedMockSpotObservation.load_state(
            "0",
            self._images_dir,
            objects_dict
        )
        obs = _MockSpotObservation.init_from_saved(
            loaded_obs,
            object_dict=objects_dict,
            vlm_atom_dict=None,  # Will be populated in Perceiver!
            vlm_predicates=VLM_PREDICATES if CFG.mock_env_vlm_eval_predicate else None,
        )
        
        # Set current task and observation
        if CFG.test_task_json_dir is not None and train_or_test == "test":
            self._current_task = self._test_tasks[task_idx]
        else:
            # Generate goal description and create task
            goal_description = self._generate_goal_description()
            self._current_task = EnvironmentTask(obs, goal_description)
        
        self._current_observation = obs
        self._current_task_goal_reached = False
        self._last_action = None
        
        return obs
    
    def _generate_goal_description(self) -> GoalDescription:
        """Generate a goal description for the current task."""
        # NOTE: to update this from language
        return self.goal_atoms

    def _create_operators(self) -> Iterator[STRIPSOperator]:
        """Create STRIPS operators for this environment.
        
        The operators are divided into two categories:
        1. Base Operators:
           - MoveToReachObject: Move robot to reach a movable object
           - MoveToHandViewObject: Move robot's hand to view an object
           - PickObjectFromTop: Pick up an object from a surface from above
           - PlaceObjectOnTop: Place a held object on a surface
           - DropObjectInside: Drop a held object inside a container
           
        2. Belief-Space Operators (enabled when use_belief_space_operators=True):
           - MoveToHandObserveObjectFromTop: Move to observe a container from above
           - ObserveCupContent: Observe if a cup has water
        
        Example Sequences:
        1. Basic Pick and Place:
           MoveToReachObject -> MoveToHandViewObject -> PickObjectFromTop -> 
           MoveToReachObject -> PlaceObjectOnTop
        
        2. Place in Container:
           MoveToReachObject -> MoveToHandViewObject -> PickObjectFromTop -> 
           MoveToReachObject -> DropObjectInside
           
        3. Check Container Contents:
           MoveToHandObserveObjectFromTop -> ObserveCupContent
           
        4. Pick After Checking:
           MoveToHandObserveObjectFromTop -> ObserveCupContent ->
           MoveToHandViewObject -> PickObjectFromTop
        """
        # First yield the base operators
        # MoveToReachObject: Move robot to a position where it can reach an object
        # Preconditions: Object not blocked, robot not holding it
        # Effects: Object becomes reachable
        robot = Variable("?robot", _robot_type)
        obj = Variable("?object", _movable_object_type)
        parameters = [robot, obj]
        preconds = {
            LiftedAtom(_NotBlocked, [obj]),
            LiftedAtom(_NotHolding, [robot, obj]),
        }
        add_effs = {LiftedAtom(_Reachable, [robot, obj])}
        del_effs: Set[LiftedAtom] = set()
        ignore_effs = {_Reachable, _InHandView, _InView, _RobotReadyForSweeping}
        yield STRIPSOperator("MoveToReachObject", parameters, preconds, add_effs,
                            del_effs, ignore_effs)

        # MoveToHandViewObject: Move robot's hand to view an object
        robot = Variable("?robot", _robot_type)
        obj = Variable("?object", _movable_object_type)
        parameters = [robot, obj]
        preconds = {
            LiftedAtom(_NotBlocked, [obj]),
            LiftedAtom(_HandEmpty, [robot]),
            LiftedAtom(_NotInsideAnyContainer, [obj])  # Object must not be in a container
        }
        add_effs = {LiftedAtom(_InHandView, [robot, obj])}
        del_effs = set()
        ignore_effs = {_InHandView, _InView, _RobotReadyForSweeping}
        yield STRIPSOperator("MoveToHandViewObject", parameters, preconds,
                            add_effs, del_effs, ignore_effs)

        # MoveToHandViewObjectFromTop: Move robot's hand to view an object from above
        robot = Variable("?robot", _robot_type)
        obj = Variable("?object", _movable_object_type)
        parameters = [robot, obj]
        preconds = {
            LiftedAtom(_NotBlocked, [obj]),
            LiftedAtom(_HandEmpty, [robot]),
            LiftedAtom(_NotInsideAnyContainer, [obj])  # Object must not be in a container
        }
        add_effs = {LiftedAtom(_InHandViewFromTop, [robot, obj])}
        del_effs = set()
        ignore_effs = {_InHandView, _InView, _RobotReadyForSweeping}
        yield STRIPSOperator("MoveToHandViewObjectFromTop", parameters, preconds,
                            add_effs, del_effs, ignore_effs)

        # MoveToHandViewObjectInContainer: Move robot's hand to view an object inside a container
        # Preconditions: Object not blocked, hand empty, object in container, container open
        # Effects: Object in hand's view
        robot = Variable("?robot", _robot_type)
        obj = Variable("?object", _movable_object_type)
        container = Variable("?container", _container_type)
        parameters = [robot, obj, container]
        preconds = {
            LiftedAtom(_NotBlocked, [obj]),
            LiftedAtom(_HandEmpty, [robot]),
            LiftedAtom(_Inside, [obj, container]),
            LiftedAtom(_DrawerOpen, [container])  # Container must be open to view object inside
        }
        add_effs = {LiftedAtom(_InHandView, [robot, obj])}
        del_effs = set()
        ignore_effs = {_InHandView, _InView, _RobotReadyForSweeping}
        yield STRIPSOperator("MoveToHandViewObjectInContainer", parameters, preconds,
                            add_effs, del_effs, ignore_effs)

        # PickObjectFromTop: Pick up an object from a surface from above
        # Preconditions: Object on surface, hand empty, object in view, not in container
        # Effects: Robot holding object, no longer on surface
        robot = Variable("?robot", _robot_type)
        obj = Variable("?object", _movable_object_type)
        surface = Variable("?surface", _immovable_object_type)
        parameters = [robot, obj, surface]
        preconds = {
            LiftedAtom(_On, [obj, surface]),
            LiftedAtom(_HandEmpty, [robot]),
            LiftedAtom(_InHandView, [robot, obj]),
            LiftedAtom(_NotInsideAnyContainer, [obj]),
            LiftedAtom(_IsPlaceable, [obj]),
            LiftedAtom(_HasFlatTopSurface, [surface]),
        }
        add_effs = {
            LiftedAtom(_Holding, [robot, obj]),
        }
        del_effs = {
            LiftedAtom(_On, [obj, surface]),
            LiftedAtom(_HandEmpty, [robot]),
            LiftedAtom(_InHandView, [robot, obj]),
            LiftedAtom(_NotHolding, [robot, obj]),
        }
        ignore_effs = set()
        yield STRIPSOperator("PickObjectFromTop", parameters, preconds, add_effs,
                            del_effs, ignore_effs)

        # PlaceObjectOnTop: Place a held object on a surface
        # Preconditions: Robot holding object, surface reachable and flat
        # Effects: Object on surface, hand empty
        robot = Variable("?robot", _robot_type)
        held = Variable("?held", _movable_object_type)
        surface = Variable("?surface", _immovable_object_type)
        parameters = [robot, held, surface]
        preconds = {
            LiftedAtom(_Holding, [robot, held]),
            LiftedAtom(_Reachable, [robot, surface]),
            LiftedAtom(_NEq, [held, surface]),
            LiftedAtom(_IsPlaceable, [held]),
            LiftedAtom(_HasFlatTopSurface, [surface]),
            LiftedAtom(_FitsInXY, [held, surface]),
        }
        add_effs = {
            LiftedAtom(_On, [held, surface]),
            LiftedAtom(_HandEmpty, [robot]),
            LiftedAtom(_NotHolding, [robot, held]),
        }
        del_effs = {
            LiftedAtom(_Holding, [robot, held]),
        }
        ignore_effs = set()
        yield STRIPSOperator("PlaceObjectOnTop", parameters, preconds, add_effs,
                            del_effs, ignore_effs)

        # DropObjectInside: Drop a held object inside a container
        robot = Variable("?robot", _robot_type)
        held = Variable("?held", _movable_object_type)
        container = Variable("?container", _container_type)
        parameters = [robot, held, container]
        preconds = {
            LiftedAtom(_Holding, [robot, held]),
            LiftedAtom(_Reachable, [robot, container]),
            LiftedAtom(_IsPlaceable, [held]),
            LiftedAtom(_FitsInXY, [held, container]),
        }
        add_effs = {
            LiftedAtom(_Inside, [held, container]),
            LiftedAtom(_HandEmpty, [robot]),
            LiftedAtom(_NotHolding, [robot, held]),
        }
        del_effs = {
            LiftedAtom(_Holding, [robot, held]),
            LiftedAtom(_NotInsideAnyContainer, [held])
        }
        ignore_effs = set()
        yield STRIPSOperator("DropObjectInside", parameters, preconds, add_effs,
                            del_effs, ignore_effs)

        # PickObjectFromContainer: Pick up an object from inside a container
        robot = Variable("?robot", _robot_type)
        obj = Variable("?object", _movable_object_type)
        container = Variable("?container", _container_type)
        parameters = [robot, obj, container]
        preconds = {
            LiftedAtom(_Inside, [obj, container]),
            LiftedAtom(_HandEmpty, [robot]),
            LiftedAtom(_InHandView, [robot, obj]),
            LiftedAtom(_IsPlaceable, [obj]),
            LiftedAtom(_DrawerOpen, [container]),  # Container must be open
            LiftedAtom(_Reachable, [robot, obj]),
        }
        add_effs = {
            LiftedAtom(_Holding, [robot, obj]),
            LiftedAtom(_NotInsideAnyContainer, [obj]),
        }
        del_effs = {
            LiftedAtom(_Inside, [obj, container]),
            LiftedAtom(_HandEmpty, [robot]),
            LiftedAtom(_InHandView, [robot, obj]),
            LiftedAtom(_NotHolding, [robot, obj]),
        }
        ignore_effs = set()
        yield STRIPSOperator("PickObjectFromContainer", parameters, preconds, add_effs,
                            del_effs, ignore_effs)

        # OpenDrawer: Open drawer without observation
        robot = Variable("?robot", _robot_type)
        drawer = Variable("?drawer", _container_type)
        parameters = [robot, drawer]
        preconds = {
            LiftedAtom(_HandEmpty, [robot]),
            LiftedAtom(_DrawerClosed, [drawer]),
            LiftedAtom(_Reachable, [robot, drawer]),
            LiftedAtom(_NotBlocked, [drawer]),
            LiftedAtom(_InHandView, [robot, drawer])  # Drawer must be in hand view
        }
        add_effs = {LiftedAtom(_DrawerOpen, [drawer])}
        del_effs = {LiftedAtom(_DrawerClosed, [drawer])}
        ignore_effs = set()
        yield STRIPSOperator("OpenDrawer", parameters, preconds, add_effs,
                            del_effs, ignore_effs)

        # CloseDrawer: Close drawer without observation
        robot = Variable("?robot", _robot_type)
        drawer = Variable("?drawer", _container_type)
        parameters = [robot, drawer]
        preconds = {
            LiftedAtom(_HandEmpty, [robot]),
            LiftedAtom(_DrawerOpen, [drawer]),
            LiftedAtom(_Reachable, [robot, drawer]),
            LiftedAtom(_NotBlocked, [drawer]),
            LiftedAtom(_InHandView, [robot, drawer])  # Drawer must be in hand view
        }
        add_effs = {LiftedAtom(_DrawerClosed, [drawer])}
        del_effs = {LiftedAtom(_DrawerOpen, [drawer])}
        ignore_effs = set()
        yield STRIPSOperator("CloseDrawer", parameters, preconds, add_effs,
                            del_effs, ignore_effs)
        
        # if not CFG.mock_env_use_belief_operators:
        #     return

        # ObserveCupContent: Observe if a cup has water (renamed from ObserveContainerContent)
        robot = Variable("?robot", _robot_type)
        container = Variable("?container", _container_type)
        parameters = [robot, container]
        preconds = {
            LiftedAtom(_InHandViewFromTop, [robot, container]),
            LiftedAtom(_ContainingWaterUnknown, [container]),
        }
        add_effs = {
            LiftedAtom(_ContainingWaterKnown, [container]),
        }
        del_effs = {
            LiftedAtom(_ContainingWaterUnknown, [container]),
        }
        ignore_effs = set()  # No effects to ignore for this operator
        yield STRIPSOperator("ObserveCupContent",
                            parameters,
                            preconds,
                            add_effs,
                            del_effs,
                            ignore_effs)

        # ObserveDrawerContentFindEmpty: Look in drawer and find it empty
        robot = Variable("?robot", _robot_type)
        container = Variable("?container", _container_type)
        parameters = [robot, container]
        preconds = {
            LiftedAtom(_Unknown_ContainerEmpty, [container]),
            LiftedAtom(_DrawerOpen, [container]),  # Drawer must be open to observe
            LiftedAtom(_Reachable, [robot, container]),  # Robot must be able to reach drawer
        }
        add_effs = {
            LiftedAtom(_Known_ContainerEmpty, [container]),  # We now know the drawer's state
            LiftedAtom(_BelieveTrue_ContainerEmpty, [container]),  # We believe it's empty
        }
        del_effs = {
            LiftedAtom(_Unknown_ContainerEmpty, [container]),
        }
        ignore_effs = set()
        yield STRIPSOperator("ObserveDrawerContentFindEmpty",
                            parameters,
                            preconds,
                            add_effs,
                            del_effs,
                            ignore_effs)

        # ObserveDrawerContentFindNotEmpty: Look in drawer and find objects
        robot = Variable("?robot", _robot_type)
        container = Variable("?container", _container_type)
        parameters = [robot, container]
        preconds = {
            LiftedAtom(_Unknown_ContainerEmpty, [container]),
            LiftedAtom(_DrawerOpen, [container]),  # Drawer must be open to observe
            LiftedAtom(_Reachable, [robot, container]),  # Robot must be able to reach drawer
        }
        add_effs = {
            LiftedAtom(_Known_ContainerEmpty, [container]),  # We now know the drawer's state
            LiftedAtom(_BelieveFalse_ContainerEmpty, [container]),  # We believe it's not empty
        }
        del_effs = {
            LiftedAtom(_Unknown_ContainerEmpty, [container]),
        }
        ignore_effs = set()
        yield STRIPSOperator("ObserveDrawerContentFindNotEmpty",
                            parameters,
                            preconds,
                            add_effs,
                            del_effs,
                            ignore_effs)

        if not CFG.mock_env_use_belief_operators:
            return
        
    def simulate(self, state: State, action: Action) -> State:
        """Simulate a state transition."""
        raise NotImplementedError("Simulate not implemented for mock environment.")

    def get_train_tasks(self) -> List[EnvironmentTask]:
        """Get list of training tasks."""
        return []  # No training tasks in mock environment

    def get_test_tasks(self) -> List[EnvironmentTask]:
        """Get list of test tasks."""
        if CFG.test_task_json_dir is not None:
            return self._test_tasks
        # Create a single test task with initial observation and goal
        obs = self.reset("test", 0)
        goal_description = self._generate_goal_description()
        task = EnvironmentTask(obs, goal_description)
        return [task]

    def get_task_from_params(self, params: Dict[str, Any]) -> EnvironmentTask:
        """Get a task from parameters."""
        raise NotImplementedError("Task creation not implemented for mock environment.")

    def render_state_plt(self, state: State, task: EnvironmentTask, action: Optional[Action] = None,
                        caption: Optional[str] = None) -> List[Image]:
        """Render state using matplotlib."""
        raise NotImplementedError("Matplotlib rendering not implemented for mock environment.")

    def render_state(self, state: State, task: EnvironmentTask, action: Optional[Action] = None,
                    caption: Optional[str] = None) -> List[Image]:
        """Render state using environment-specific renderer."""
        raise NotImplementedError("State rendering not implemented for mock environment.")

    def render_task(self, task: EnvironmentTask) -> Video:
        """Render task using environment-specific renderer."""
        raise NotImplementedError("Task rendering not implemented for mock environment.")

    @property
    def strips_operators(self) -> Set[STRIPSOperator]:
        """Get the STRIPSOperators for this environment."""
        return set(self._operators)
    
    @property
    def objects(self) -> Set[Object]:
        """Get all objects in the environment."""
        raise RuntimeError("Objects not specified for the Base environment.")

    @property
    def types(self) -> Set[Type]:
        """Get the types used in this environment."""
        return TYPES

    @property
    def predicates(self) -> Set[Predicate]:
        """Get the predicates used in this environment."""
        preds: Set[Predicate] = set(PREDICATES)  # Explicit type hint
        if CFG.mock_env_use_belief_operators:
            preds.update(BELIEF_PREDICATES)
        return preds

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """Get the goal predicates for this environment."""
        return set(GOAL_PREDICATES)  # Convert to set with proper type

    @property
    def action_space(self) -> Box:
        """Get the action space for this environment."""
        # Mock environment doesn't use continuous actions, but we need to define the space
        # Using a simple 3D space for position control
        return Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    @property
    def options(self) -> Set[ParameterizedOption]:
        """Get the options for this environment."""
        from predicators.ground_truth_models import get_gt_options
        return get_gt_options(self.get_name())
    
    @property
    def nsrts(self) -> Set[NSRT]:
        """Create NSRTs from the environment's predicates and options."""
        
        from predicators.utils import null_sampler
        
        named_options = {o.name: o for o in self.options}
        nsrts = set()
        
        for strips_op in self.strips_operators:
            if strips_op.name not in named_options:
                print(f"[blue]Skipping {strips_op.name} since it's not in named options[/blue]")
                continue
            option = named_options[strips_op.name]
            nsrt = strips_op.make_nsrt(
                option=option,
                option_vars=strips_op.parameters,
                sampler=null_sampler,  # Use dummy sampler for all operators
            )
            nsrts.add(nsrt)
        
        return nsrts

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        """Generate training tasks."""
        return []  # No training tasks in mock environment

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        """Generate test tasks."""
        return []  # No test tasks in mock environment


class MockSpotPickPlaceTwoCupEnv(MockSpotEnv):
    """A mock environment for testing pick and place with two cups."""
    
    # NOTE: This is a test transition system with manually created images
    preset_data_dir = os.path.join("mock_env_data", "test_mock_two_cup_pick_place_manual_images")

    @classmethod
    def get_name(cls) -> str:
        """Get the name of this environment."""
        return "mock_spot_pick_place_two_cup"

    def __init__(self, use_gui: bool = True) -> None:
        """Initialize the environment."""
        super().__init__(use_gui=use_gui)
        self.name = "mock_spot_pick_place_two_cup"
        
        # Create objects
        self.robot = Object("robot", _robot_type)
        self.cup1 = Object("cup1", _container_type)
        self.cup2 = Object("cup2", _container_type)
        self.table = Object("table", _immovable_object_type)
        self.target = Object("target", _container_type)
        
        # Set up initial state
        self._objects = {
            "robot": self.robot,
            "table": self.table,
            "target": self.target,
            "cup1": self.cup1,
            "cup2": self.cup2
        }
        self._set_initial_state_and_goal()
    
    def _set_initial_state_and_goal(self) -> None:
        """Set up initial state and goal atoms."""
        # Create initial and goal atoms
        self.initial_atoms = {
            # Robot state
            GroundAtom(_HandEmpty, [self.robot]),
            GroundAtom(_NotHolding, [self.robot, self.cup1]),
            GroundAtom(_NotHolding, [self.robot, self.cup2]),
            
            # Object positions
            GroundAtom(_On, [self.cup1, self.table]),
            GroundAtom(_On, [self.cup2, self.table]),
            GroundAtom(_On, [self.target, self.table]),
            
            # Object properties
            GroundAtom(_NotBlocked, [self.cup1]),
            GroundAtom(_NotBlocked, [self.cup2]),
            GroundAtom(_NotBlocked, [self.target]),
            GroundAtom(_IsPlaceable, [self.cup1]),
            GroundAtom(_IsPlaceable, [self.cup2]),
            
            # Surface properties
            GroundAtom(_HasFlatTopSurface, [self.table]),
            
            # Containment properties
            GroundAtom(_FitsInXY, [self.cup1, self.target]),
            GroundAtom(_FitsInXY, [self.cup2, self.target]),
            GroundAtom(_NotInsideAnyContainer, [self.cup1]),
            GroundAtom(_NotInsideAnyContainer, [self.cup2]),
            GroundAtom(_NotHolding, [self.robot, self.target]),
            
            # Reachability
            GroundAtom(_Reachable, [self.robot, self.cup1]),
            GroundAtom(_Reachable, [self.robot, self.cup2]),
            GroundAtom(_Reachable, [self.robot, self.target]),
            GroundAtom(_Reachable, [self.robot, self.table]),
            GroundAtom(_InHandView, [self.robot, self.cup1]),
            GroundAtom(_InHandView, [self.robot, self.cup2]),
            
            # Object relationships
            GroundAtom(_NEq, [self.cup1, self.table]),
            GroundAtom(_NEq, [self.cup1, self.target]),
            GroundAtom(_NEq, [self.cup2, self.table]),
            GroundAtom(_NEq, [self.cup2, self.target]),
            GroundAtom(_NEq, [self.cup1, self.cup2]),
            GroundAtom(_NEq, [self.target, self.table])
        }
        
        self.goal_atoms = {
            GroundAtom(_Inside, [self.cup1, self.target]),
            GroundAtom(_Inside, [self.cup2, self.target])
        }
    
    def _create_operators(self) -> Iterator[STRIPSOperator]:
        """Create STRIPS operators specific to pick-and-place tasks."""
        # Get all operators from parent class
        all_operators = list(super()._create_operators())
        
        # Define operators to keep
        op_names_to_keep = {
            "MoveToReachObject",
            "MoveToHandViewObject", 
            "PickObjectFromTop",
            # "PlaceObjectOnTop",
            "DropObjectInside"
        }
        
        # Filter operators
        for op in all_operators:
            if op.name in op_names_to_keep:
                yield op

    @property
    def objects(self) -> Set[Object]:
        """Get all objects in the environment."""
        return set(self._objects.values())

    def get_train_tasks(self) -> List[EnvironmentTask]:
        """Get list of training tasks."""
        # Reset environment to get initial observation
        obs = self.reset("train", 0)
        # Create task with initial observation and goal
        task = EnvironmentTask(obs, self.goal_atoms)
        return [task]

    def get_test_tasks(self) -> List[EnvironmentTask]:
        """Get list of test tasks."""
        # Reset environment to get initial observation
        obs = self.reset("test", 0)
        # Create task with initial observation and goal
        task = EnvironmentTask(obs, self.goal_atoms)
        return [task]
