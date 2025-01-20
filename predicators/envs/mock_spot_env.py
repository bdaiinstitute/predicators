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
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator, Sequence, ClassVar, Union
from itertools import product

import numpy as np
from gym.spaces import Box

from predicators.envs import BaseEnv
from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext
from predicators.structs import Action, State, Object, Type, EnvironmentTask, Video, Image
from predicators.structs import LiftedAtom, STRIPSOperator, Variable, Predicate, GroundAtom, VLMPredicate, VLMGroundAtom
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
_ContainingWaterUnknown = Predicate("ContainingWaterUnknown", [_container_type], _dummy_classifier)
_ContainingWaterKnown = Predicate("ContainingWaterKnown", [_container_type], _dummy_classifier)
_ContainingWater = Predicate("ContainingWater", [_container_type], _dummy_classifier)
_NotContainingWater = Predicate("NotContainingWater", [_container_type], _dummy_classifier)
_InHandViewFromTop = Predicate("InHandViewFromTop", [_robot_type, _base_object_type], _dummy_classifier)
_ContainerEmpty = Predicate("ContainerEmpty", [_container_type], _dummy_classifier)

# Add new predicates for container emptiness
_Unknown_ContainerEmpty = Predicate("Unknown_ContainerEmpty", [_container_type], _dummy_classifier)
_Known_ContainerEmpty = Predicate("Known_ContainerEmpty", [_container_type], _dummy_classifier)
_BelieveTrue_ContainerEmpty = Predicate("BelieveTrue_ContainerEmpty", [_container_type], _dummy_classifier)
_BelieveFalse_ContainerEmpty = Predicate("BelieveFalse_ContainerEmpty", [_container_type], _dummy_classifier)

# Add predicates for drawer state
_DrawerClosed = Predicate("DrawerClosed", [_container_type], _dummy_classifier)
_DrawerOpen = Predicate("DrawerOpen", [_container_type], _dummy_classifier)

# Group belief-space predicates
BELIEF_PREDICATES = {
    _ContainingWaterUnknown,
    _ContainingWaterKnown,
    _ContainingWater,
    _NotContainingWater,
    _InHandViewFromTop,
    _Unknown_ContainerEmpty,
    _Known_ContainerEmpty,
    _BelieveTrue_ContainerEmpty,
    _BelieveFalse_ContainerEmpty
}

# Export all predicates
PREDICATES = {_NEq, _On, _TopAbove, _Inside, _NotInsideAnyContainer, _FitsInXY,
             _HandEmpty, _Holding, _NotHolding, _InHandView, _InView, _Reachable,
             _Blocking, _NotBlocked, _ContainerReadyForSweeping, _IsPlaceable,
             _IsNotPlaceable, _IsSweeper, _HasFlatTopSurface, _RobotReadyForSweeping,
             _DrawerClosed, _DrawerOpen, _Unknown_ContainerEmpty, _Known_ContainerEmpty,
             _BelieveTrue_ContainerEmpty, _BelieveFalse_ContainerEmpty, _InHandViewFromTop,
             _ContainingWaterUnknown, _ContainingWaterKnown, _ContainingWater, _NotContainingWater,
             _ContainerEmpty}
# Note: Now adding belief predicates

# Export goal predicates
GOAL_PREDICATES = {_On, _Inside, _ContainingWaterKnown, _Known_ContainerEmpty, _DrawerOpen}  # Add DrawerOpen to goal predicates


def get_vlm_predicates() -> Set[VLMPredicate]:
    """Get VLM predicates for mock spot environment."""
    _On = VLMPredicate(
        "On", [_movable_object_type, _base_object_type],
        prompt=
        "This predicate typically describes a movable object on a flat surface, so it's in conflict with the object being inside a container. Please check the image and confirm the object is on the surface."
    )
    _Inside = VLMPredicate(
        "Inside", [_movable_object_type, _container_type],
        prompt=
        "This typically describes an object (obj1, first arg) inside a container (obj2, second arg) (so it's overlapping), and it's in conflict with the object being on a surface. This is obj1 inside obj2, so obj1 should be smaller than obj2."
    )
    _Blocking = VLMPredicate(
        "Blocking", [_base_object_type, _base_object_type],
        prompt="This means if an object is blocking the Spot robot approaching another one."
    )
    _NotBlocked = VLMPredicate(
        "NotBlocked", [_base_object_type],
        prompt="The given object is not blocked by any other object.")

    _NotInsideAnyContainer = VLMPredicate(
        "NotInsideAnyContainer", [_movable_object_type],
        prompt="This predicate is true if the given object is not inside any container. Check the image and confirm the object is not inside any container."
    )
    
    return {_On, _Inside, _Blocking, _NotBlocked, _NotInsideAnyContainer}

# Export VLM predicates
VLM_PREDICATES = get_vlm_predicates()

# Export all predicates - use VLM or non-VLM based on config
def get_all_predicates() -> Set[Predicate]:
    """Get all predicates based on config."""
    if CFG.spot_vlm_eval_predicate:
        # Filter out non-VLM counterparts of VLM predicates
        vlm_names = {pred.name for pred in VLM_PREDICATES}
        non_vlm_preds = {pred for pred in PREDICATES if pred.name not in vlm_names}
        return non_vlm_preds | VLM_PREDICATES
    return PREDICATES

# Update PREDICATES_WITH_VLM to use function
PREDICATES_WITH_VLM = get_all_predicates() if CFG.spot_vlm_eval_predicate else None


@dataclass(frozen=True)
class _SavedMockSpotObservation:
    """Observation for mock Spot environment."""
    images: Optional[Dict[str, Any]]  # Dictionary of camera images
    gripper_open: bool  # Whether the gripper is open
    objects_in_view: Set[str]  # Names of objects visible in the image
    objects_in_hand: Set[str]  # Names of objects currently held
    state_id: str  # Unique ID for this observation's latent state
    atom_dict: Dict[str, bool]  # Dictionary of predicate atoms
    non_vlm_atom_dict: Optional[Set[GroundAtom]] = None
    
    
@dataclass(frozen=True)
class _MockSpotObservation(_SavedMockSpotObservation):
    """An observation from the mock Spot environment.
    
    Args:
        images: Dictionary of camera images
        gripper_open: Whether the gripper is open
        objects_in_view: Names of objects visible in the image
        objects_in_hand: Names of objects currently held
        vlm_atom_dict: Dictionary of VLM predicate atoms from previous step
        vlm_predicates: Set of VLM predicates from previous step
        non_vlm_atoms: Set of non-VLM predicate atoms from environment
        state_id: Unique ID for this observation's latent state
    """

    vlm_atom_dict: Optional[Dict[VLMGroundAtom, bool]] = None
    vlm_predicates: Optional[Set[VLMPredicate]] = None
    
    @classmethod
    def init_from_saved(cls, saved_obs: _SavedMockSpotObservation, vlm_atom_dict: Optional[Dict[VLMGroundAtom, bool]] = None,
                        vlm_predicates: Optional[Set[VLMPredicate]] = None) -> "_MockSpotObservation":
        """Initialize from a saved observation."""
        return cls(
            # Non-VLM fields, saved 
            images=saved_obs.images,
            gripper_open=saved_obs.gripper_open,
            objects_in_view=saved_obs.objects_in_view,
            objects_in_hand=saved_obs.objects_in_hand,
            state_id=saved_obs.state_id,
            atom_dict=saved_obs.atom_dict,
            non_vlm_atom_dict=saved_obs.non_vlm_atom_dict,
            # Fields decided by VLM online evaluation
            vlm_atom_dict=vlm_atom_dict,
            vlm_predicates=vlm_predicates
        )
        
        
# NOTE: keep these in mind:
# from predicators.spot_utils.perception.object_perception import \
#     get_vlm_atom_combinations, vlm_predicate_batch_classify

# Add utility functions for VLM predicates
def get_vlm_atom_combinations(objects: Set[Object], predicates: Set[VLMPredicate]) -> List[VLMGroundAtom]:
    """Get all possible VLM ground atom combinations.
    
    Args:
        objects: Dictionary mapping object names to Object instances
        predicates: Set of VLM predicates to generate atoms for
    """
    atoms = []
    for pred in predicates:
        # Get number of arguments needed
        num_args = len(pred.types)
        # Generate all possible combinations of objects
        for args in product(objects, repeat=num_args):
            # Check if object types match predicate types
            if all(isinstance(arg, Object) and 
                  (arg.type == type_ or 
                   (hasattr(arg.type, "parent") and arg.type.parent == type_))
                  for arg, type_ in zip(args, pred.types)):
                atoms.append(VLMGroundAtom(pred, list(args)))
    return atoms

def vlm_predicate_batch_classify(atoms: List[VLMGroundAtom], 
                               images: List[RGBDImageWithContext],
                               predicates: Set[VLMPredicate],
                               get_dict: bool = False) -> Union[Dict[VLMGroundAtom, bool], Set[VLMGroundAtom]]:
    """Mock VLM batch classification.
    
    In the mock environment, we'll return random classifications.
    In the real environment, this would call the VLM service.
    """
    if get_dict:
        return {atom: bool(np.random.randint(2)) for atom in atoms}
    return {atom for atom in atoms if bool(np.random.randint(2))}



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
        
        # Create constant objects.
        self._spot_object = Object("robot", _robot_type)
        
        # Load or initialize transition graph
        self._transitions: Dict[str, Dict[str, str]] = {}  # state_id -> {action -> next_state_id}
        self._observations: Dict[str, _SavedMockSpotObservation] = {}  # state_id -> observation
        self._load_graph_data()

        # Create operators
        self._operators = list(self._create_operators())
        
    def _build_observation(self, curr_obs: Optional[_MockSpotObservation]) -> _MockSpotObservation:
        """Get the current observation.
        
        Similar to Spot env, we need to use VLM to batch evaluate VLM predicates.
        For other properties, we need to get them from saved transition system.
        
        Args:
            curr_obs: Current observation for VLM atom comparison
        """
        # Get non-VLM atoms from saved data
        non_vlm_atoms = self._observations[self._current_state_id].non_vlm_atom_dict \
            if self._current_state_id is not None else set()
            
        # Get current observation data
        images = self._observations[self._current_state_id].images \
            if self._current_state_id is not None else None
        gripper_open = self._gripper_open
        objects_in_view = self._observations[self._current_state_id].objects_in_view \
            if self._current_state_id is not None else set()
        objects_in_hand = self._objects_in_hand
        
        # Initialize VLM-related fields
        vlm_predicates = VLM_PREDICATES if CFG.spot_vlm_eval_predicate else None
        vlm_atom_dict = {}
        
        # Handle VLM predicates if enabled
        # Prepare the VLM predicates and ground atoms
        if CFG.spot_vlm_eval_predicate:
            vlm_predicates = VLM_PREDICATES
            # Add robot object
            # Use currently visible objects to generate atom combinations
            # FIXME: check the objects here
            objects = list(objects_in_view)
            # NOTE: Also have the special robot object for robot-related predicates
            objects.append(self._spot_object)
            # Batch classify VLM predicates
            vlm_atom_new = vlm_predicate_batch_classify(
                vlm_atoms,
                [images["mock_camera"]],  # List of RGBD images
                predicates=VLM_PREDICATES,
                get_dict=True
            )
            assert isinstance(vlm_atom_new, dict)
            
            # Update VLM atom values
            if curr_obs is not None and curr_obs.vlm_atom_dict is not None:
                vlm_atom_dict = curr_obs.vlm_atom_dict.copy()
            for atom, result in vlm_atom_new.items():
                if result is not None:
                    vlm_atom_dict[atom] = result
            
        # Logging in Rich table
        # logging.info(f"Calculated VLM atoms (in current obs): {dict(vlm_atom_new)}")
        table = Table(title="Evaluated VLM atoms (in current obs)")
        table.add_column("Atom", style="cyan")
        table.add_column("Value", style="magenta")
        # Sort the atoms via key str for ordered printing
        for atom in sorted(vlm_atom_new.keys(), key=str):
            table.add_row(str(atom), str(vlm_atom_new[atom]))
        logging.info(log_rich_table(table))

        # Use Rich table to highlight how they change
        table_compare = Table(title="VLM atoms comparison")
        table_compare.add_column("Atom", style="cyan")
        table_compare.add_column("Value (Last)", style="blue")
        table_compare.add_column("Value (New)", style="magenta")
        vlm_atom_union = set(vlm_atom_new.keys()) | set(
            curr_obs.vlm_atom_dict.keys())
        # Sort the atoms via key str for ordered printing
        for atom in sorted(vlm_atom_union, key=str):
            table_compare.add_row(
                str(atom), str(curr_obs.vlm_atom_dict.get(atom, None)),
                str(vlm_atom_new.get(atom, None)))
        logging.info(log_rich_table(table_compare))

        logging.info(
            f"True VLM atoms (after updated with current obs): "
            f"{dict(filter(lambda it: it[1], vlm_atom_return.items()))}")
        
        return _MockSpotObservation(
            images=images,
            gripper_open=gripper_open,
            objects_in_view=objects_in_view,
            objects_in_hand=objects_in_hand,
            state_id=self._current_state_id or "init",
            atom_dict={},  # Empty since we use non_vlm_atom_dict
            non_vlm_atom_dict=non_vlm_atoms,
            vlm_atom_dict=vlm_atom_dict,
            vlm_predicates=vlm_predicates
        )

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
                    self._observations[state_id] = _SavedMockSpotObservation(
                        images=None,  # RGBD images are loaded separately
                        gripper_open=obs_data["gripper_open"],
                        objects_in_view=set(obs_data["objects_in_view"]),
                        objects_in_hand=set(obs_data["objects_in_hand"]),
                        state_id=state_id,
                        atom_dict={},
                        non_vlm_atom_dict=None,
                        # FIXME: need to update the saving (in env creator) and loading here
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

    def add_state(self, 
                 images: Optional[Dict[str, RGBDImageWithContext]] = None,
                 gripper_open: bool = True,
                 objects_in_view: Optional[Set[str]] = None,
                 objects_in_hand: Optional[Set[str]] = None) -> str:
        """Add a new state to the environment."""
        # Generate unique state ID
        state_id = str(len(self._observations))
        
        # Create observation
        self._observations[state_id] = _MockSpotObservation(
            images=images,
            gripper_open=gripper_open,
            objects_in_view=objects_in_view or set(),
            objects_in_hand=objects_in_hand or set(),
            state_id=state_id,
            atom_dict={},
            non_vlm_atom_dict=set(),
            vlm_atom_dict=None,
            vlm_predicates=None
        )
        logging.debug("Added state %s with data: %s", state_id, {
            "objects_in_view": objects_in_view or set(),
            "objects_in_hand": objects_in_hand or set(),
            "gripper_open": gripper_open,
            "has_rgbd": images is not None
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
        preds = PREDICATES.copy()
        if CFG.mock_env_use_belief_operators:
            preds.update(BELIEF_PREDICATES)
        return preds

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


class MockSpotPickPlaceTwoCupEnv(MockSpotEnv):
    """Mock environment specifically for pick-and-place two cuptasks."""
    
    def __init__(self, name: str = "pick_place") -> None:
        """Initialize pick-and-place environment."""
        super().__init__()
        self.name = name
        
        # Create objects
        self.robot = Object("robot", _robot_type)
        self.cup1 = Object("cup1", _container_type)
        self.cup2 = Object("cup2", _container_type)
        self.table = Object("table", _immovable_object_type)
        self.target = Object("target", _container_type)
        
        # Store objects
        self.objects = {self.robot, self.cup1, self.cup2, self.table, self.target}
        
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
            # "MoveToReachObject",
            # "MoveToHandViewObject", 
            "PickObjectFromTop",
            # "PlaceObjectOnTop",
            "DropObjectInside"
        }
        
        # Filter operators
        for op in all_operators:
            if op.name in op_names_to_keep:
                yield op
                
            