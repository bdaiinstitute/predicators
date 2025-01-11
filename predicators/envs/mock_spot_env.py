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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator, Sequence, ClassVar

import numpy as np
from gym.spaces import Box

from predicators.envs import BaseEnv
from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext
from predicators.structs import Action, State, Object, Type, EnvironmentTask, Video, Image
from predicators.structs import LiftedAtom, STRIPSOperator, Variable, Predicate
from predicators.settings import CFG
from bosdyn.client import math_helpers


def _dummy_classifier(state: State, objects: Sequence[Object]) -> bool:
    """Dummy classifier that always returns True. Used for mock environment."""
    return True


# Types
_robot_type = Type("robot", ["x", "y", "z"])
_base_object_type = Type("base_object", ["x", "y", "z"])
_movable_object_type = Type("movable_object", ["x", "y", "z"], parent=_base_object_type)
_container_type = Type("container", ["x", "y", "z"], parent=_movable_object_type)
_immovable_object_type = Type("immovable_object", ["x", "y", "z"], parent=_base_object_type)

# Export all types
TYPES = {_robot_type, _base_object_type, _movable_object_type, _container_type, _immovable_object_type}

# Predicates
_NEq = Predicate("NEq", [_base_object_type, _base_object_type], _dummy_classifier)
_On = Predicate("On", [_movable_object_type, _base_object_type], _dummy_classifier)
_TopAbove = Predicate("TopAbove", [_base_object_type, _base_object_type], _dummy_classifier)
_Inside = Predicate("Inside", [_movable_object_type, _container_type], _dummy_classifier)
_NotInsideAnyContainer = Predicate("NotInsideAnyContainer", [_movable_object_type], _dummy_classifier)
_FitsInXY = Predicate("FitsInXY", [_movable_object_type, _base_object_type], _dummy_classifier)
_HandEmpty = Predicate("HandEmpty", [_robot_type], _dummy_classifier)
_Holding = Predicate("Holding", [_robot_type, _movable_object_type], _dummy_classifier)
_NotHolding = Predicate("NotHolding", [_robot_type, _movable_object_type], _dummy_classifier)
_InHandView = Predicate("InHandView", [_robot_type, _base_object_type], _dummy_classifier)
_InView = Predicate("InView", [_robot_type, _base_object_type], _dummy_classifier)
_Reachable = Predicate("Reachable", [_robot_type, _base_object_type], _dummy_classifier)
_Blocking = Predicate("Blocking", [_base_object_type, _base_object_type], _dummy_classifier)
_NotBlocked = Predicate("NotBlocked", [_base_object_type], _dummy_classifier)
_ContainerReadyForSweeping = Predicate("ContainerReadyForSweeping", [_container_type], _dummy_classifier)
_IsPlaceable = Predicate("IsPlaceable", [_movable_object_type], _dummy_classifier)
_IsNotPlaceable = Predicate("IsNotPlaceable", [_movable_object_type], _dummy_classifier)
_IsSweeper = Predicate("IsSweeper", [_movable_object_type], _dummy_classifier)
_HasFlatTopSurface = Predicate("HasFlatTopSurface", [_base_object_type], _dummy_classifier)
_RobotReadyForSweeping = Predicate("RobotReadyForSweeping", [_robot_type], _dummy_classifier)

# Add new predicates for cup emptiness
# TODO: Re-enable these predicates after fixing base pick-place functionality
# _ContainingWaterUnknown = Predicate("ContainingWaterUnknown", [_container_type], _dummy_classifier)
# _ContainingWaterKnown = Predicate("ContainingWaterKnown", [_container_type], _dummy_classifier)
# _ContainingWater = Predicate("ContainingWater", [_container_type], _dummy_classifier)
# _NotContainingWater = Predicate("NotContainingWater", [_container_type], _dummy_classifier)
# _InHandViewFromTop = Predicate("InHandViewFromTop", [_robot_type, _base_object_type], _dummy_classifier)

# Export all predicates
PREDICATES = {_NEq, _On, _TopAbove, _Inside, _NotInsideAnyContainer, _FitsInXY,
             _HandEmpty, _Holding, _NotHolding, _InHandView, _InView, _Reachable,
             _Blocking, _NotBlocked, _ContainerReadyForSweeping, _IsPlaceable,
             _IsNotPlaceable, _IsSweeper, _HasFlatTopSurface, _RobotReadyForSweeping}
             # TODO: Add these predicates back after fixing base pick-place functionality
             # _ContainingWaterUnknown, _ContainingWaterKnown, _ContainingWater,
             # _NotContainingWater, _InHandViewFromTop}

# Export goal predicates
GOAL_PREDICATES = {_On, _Inside}  # TODO: Add _ContainingWaterKnown back after fixing base pick-place functionality


@dataclass
class MockSpotObservation:
    """Observation for mock Spot environment."""
    rgbd: Optional[RGBDImageWithContext]  # RGB-D image with context
    gripper_open: bool  # Whether the gripper is open
    objects_in_view: Set[str]  # Names of objects visible in the image
    objects_in_hand: Set[str]  # Names of objects currently held
    state_id: str  # Unique ID for this observation's latent state


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

    def simulate(self, state: State, action: Action) -> State:
        """Simulate a state transition."""
        raise NotImplementedError("Simulate not implemented for mock environment.")

    def get_train_tasks(self) -> List[EnvironmentTask]:
        """Get list of training tasks."""
        return []  # No training tasks in mock environment

    def get_test_tasks(self) -> List[EnvironmentTask]:
        """Get list of test tasks."""
        return []  # No test tasks in mock environment

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
    def types(self) -> Set[Type]:
        """Get the types used in this environment."""
        return TYPES

    @property
    def predicates(self) -> Set[Predicate]:
        """Get the predicates used in this environment."""
        return PREDICATES

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """Get the goal predicates for this environment."""
        return GOAL_PREDICATES

    @property
    def action_space(self) -> Box:
        """Get the action space for this environment."""
        # Mock environment doesn't use continuous actions, but we need to define the space
        # Using a simple 3D space for position control
        return Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        """Generate training tasks."""
        return []  # No training tasks in mock environment

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        """Generate test tasks."""
        return []  # No test tasks in mock environment

    def __init__(self, use_gui: bool = True) -> None:
        """Initialize the mock Spot environment.
        
        Args:
            use_gui: Whether to use GUI for visualization
        """
        super().__init__(use_gui)
        
        # Get data directory from config
        data_dir = CFG.mock_env_data_dir if hasattr(CFG, "mock_env_data_dir") else "mock_env_data"
        
        # Create data directories
        self._data_dir = Path(data_dir)
        self._images_dir = self._data_dir / "images"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Initialized MockSpotEnv with data_dir: %s", data_dir)
        
        # Initialize environment state
        self._current_state_id: Optional[str] = None
        self._gripper_open: bool = True
        self._objects_in_hand: Set[str] = set()
        
        # Load or initialize transition graph
        self._transitions: Dict[str, Dict[str, str]] = {}  # state_id -> {action -> next_state_id}
        self._observations: Dict[str, MockSpotObservation] = {}  # state_id -> observation
        self._load_graph_data()

        # Create operators
        self._operators = list(self._create_operators())

    def _load_graph_data(self) -> None:
        """Load graph data from disk."""
        graph_file = self._data_dir / "graph.json"
        if not graph_file.exists():
            logging.info("No existing graph data found at %s", graph_file)
            return

        try:
            with open(graph_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._transitions = data["transitions"]
                
                # Convert observations back to MockSpotObservation objects
                self._observations = {}
                for state_id, obs_data in data["observations"].items():
                    self._observations[state_id] = MockSpotObservation(
                        rgbd=None,  # RGBD images are loaded separately
                        gripper_open=obs_data["gripper_open"],
                        objects_in_view=set(obs_data["objects_in_view"]),
                        objects_in_hand=set(obs_data["objects_in_hand"]),
                        state_id=state_id
                    )
            logging.info("Loaded graph data with %d states and %d transitions", 
                        len(self._observations), sum(len(t) for t in self._transitions.values()))
        except Exception as e:
            logging.error("Failed to load graph data: %s", e)
            self._transitions = {}
            self._observations = {}

    def _save_graph_data(self) -> None:
        """Save graph data to disk."""
        graph_file = self._data_dir / "graph.json"
        try:
            # Convert observations to serializable format
            observations_data = {}
            for state_id, obs in self._observations.items():
                observations_data[state_id] = {
                    "gripper_open": obs.gripper_open,
                    "objects_in_view": list(obs.objects_in_view),
                    "objects_in_hand": list(obs.objects_in_hand)
                }
            
            data = {
                "transitions": self._transitions,
                "observations": observations_data
            }
            
            with open(graph_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logging.debug("Saved graph data to %s", graph_file)
        except Exception as e:
            logging.error("Failed to save graph data: %s", e)

    def _create_operators(self) -> Iterator[STRIPSOperator]:
        """Create operators for the mock environment."""
        # MoveToReachObject
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

        # MoveToHandViewObject
        robot = Variable("?robot", _robot_type)
        obj = Variable("?object", _movable_object_type)
        parameters = [robot, obj]
        preconds = {
            LiftedAtom(_NotBlocked, [obj]),
            LiftedAtom(_HandEmpty, [robot])
        }
        add_effs = {LiftedAtom(_InHandView, [robot, obj])}
        del_effs = set()
        ignore_effs = {_Reachable, _InHandView, _InView, _RobotReadyForSweeping}
        yield STRIPSOperator("MoveToHandViewObject", parameters, preconds,
                            add_effs, del_effs, ignore_effs)

        # PickObjectFromTop
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

        # PlaceObjectOnTop
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

        # DropObjectInside
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

        # TODO: Re-enable these operators after fixing base pick-place functionality
        # # MoveToHandObserveObjectFromTop
        # robot = Variable("?robot", _robot_type)
        # obj = Variable("?object", _container_type)
        # parameters = [robot, obj]
        # preconds = {
        #     LiftedAtom(_NotBlocked, [obj]),
        #     LiftedAtom(_HandEmpty, [robot]),
        #     LiftedAtom(_ContainingWaterUnknown, [obj])
        # }
        # add_effs = {LiftedAtom(_InHandViewFromTop, [robot, obj])}
        # del_effs = set()
        # ignore_effs = {_Reachable, _InHandView, _InView, _RobotReadyForSweeping}
        # yield STRIPSOperator("MoveToHandObserveObjectFromTop", parameters, preconds,
        #                     add_effs, del_effs, ignore_effs)

        # # ObserveFromTop
        # robot = Variable("?robot", _robot_type)
        # obj = Variable("?object", _container_type)
        # parameters = [robot, obj]
        # preconds = {
        #     LiftedAtom(_InHandViewFromTop, [robot, obj]),
        #     LiftedAtom(_ContainingWaterUnknown, [obj])
        # }
        # add_effs = {
        #     LiftedAtom(_ContainingWaterKnown, [obj]),
        #     LiftedAtom(_ContainingWater, [obj])
        # }
        # del_effs = {
        #     LiftedAtom(_ContainingWaterUnknown, [obj]),
        #     LiftedAtom(_InHandViewFromTop, [robot, obj])
        # }
        # ignore_effs = set()
        # yield STRIPSOperator("ObserveFromTop", parameters, preconds, add_effs,
        #                     del_effs, ignore_effs)

    def add_state(self, 
                 rgbd: Optional[RGBDImageWithContext] = None,
                 gripper_open: bool = True,
                 objects_in_view: Optional[Set[str]] = None,
                 objects_in_hand: Optional[Set[str]] = None) -> str:
        """Add a new state to the environment."""
        # Generate unique state ID
        state_id = str(len(self._observations))
        
        # Create observation
        self._observations[state_id] = MockSpotObservation(
            rgbd=rgbd,
            gripper_open=gripper_open,
            objects_in_view=objects_in_view or set(),
            objects_in_hand=objects_in_hand or set(),
            state_id=state_id
        )
        logging.debug("Added state %s with data: %s", state_id, {
            "objects_in_view": objects_in_view or set(),
            "objects_in_hand": objects_in_hand or set(),
            "gripper_open": gripper_open,
            "has_rgbd": rgbd is not None
        })
        
        # Save updated data
        self._save_graph_data()
        
        return state_id

    def add_transition(self, 
                      from_state_id: str,
                      action_name: str,
                      to_state_id: str) -> None:
        """Add a transition between states."""
        if from_state_id not in self._observations:
            raise ValueError(f"Unknown state ID: {from_state_id}")
        if to_state_id not in self._observations:
            raise ValueError(f"Unknown state ID: {to_state_id}")
            
        # Verify action is a valid operator
        if not any(op.name == action_name for op in self._operators):
            raise ValueError(f"Unknown operator: {action_name}")
            
        # Initialize transitions dict for this state if needed
        if from_state_id not in self._transitions:
            self._transitions[from_state_id] = {}
            
        self._transitions[from_state_id][action_name] = to_state_id
        logging.debug("Added transition: %s -(%s)-> %s", from_state_id, action_name, to_state_id)
        self._save_graph_data()
