import logging
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
import matplotlib.pyplot as plt
import yaml

from predicators.envs import BaseEnv
from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext, UnposedImageWithContext
from predicators.structs import Action, State, Object, Type, EnvironmentTask, Video, Image
from predicators.structs import LiftedAtom, STRIPSOperator, Variable, Predicate, GroundAtom, VLMPredicate, GroundTruthPredicate, VLMGroundAtom
from predicators.settings import CFG
from bosdyn.client import math_helpers
from predicators.spot_utils.perception.object_perception import get_vlm_atom_combinations, vlm_predicate_batch_classify
from predicators.utils import get_object_combinations



# from predicators.structs import Type, GroundTruthPredicate, VLMPredicate, State, Object, Predicate
# from typing import Sequence, Set, Container, Optional
# from predicators.settings import CFG

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
_NEq = GroundTruthPredicate("NEq", [_base_object_type, _base_object_type], _dummy_classifier)
_On = GroundTruthPredicate("On", [_movable_object_type, _base_object_type], _dummy_classifier)
_TopAbove = GroundTruthPredicate("TopAbove", [_base_object_type, _base_object_type], _dummy_classifier)
_NotInsideAnyContainer = GroundTruthPredicate("NotInsideAnyContainer", [_movable_object_type], _dummy_classifier)
_FitsInXY = GroundTruthPredicate("FitsInXY", [_movable_object_type, _base_object_type], _dummy_classifier)
_HandEmpty = GroundTruthPredicate("HandEmpty", [_robot_type], _dummy_classifier)
_Holding = GroundTruthPredicate("Holding", [_robot_type, _movable_object_type], _dummy_classifier)
_NotHolding = GroundTruthPredicate("NotHolding", [_robot_type, _movable_object_type], _dummy_classifier)
_InHandView = GroundTruthPredicate("InHandView", [_robot_type, _base_object_type], _dummy_classifier)
_InView = GroundTruthPredicate("InView", [_robot_type, _base_object_type], _dummy_classifier)
_Reachable = GroundTruthPredicate("Reachable", [_robot_type, _base_object_type], _dummy_classifier)
_Blocking = GroundTruthPredicate("Blocking", [_base_object_type, _base_object_type], _dummy_classifier)
_NotBlocked = GroundTruthPredicate("NotBlocked", [_base_object_type], _dummy_classifier)
_ContainerReadyForSweeping = GroundTruthPredicate("ContainerReadyForSweeping", [_container_type], _dummy_classifier)
_IsPlaceable = GroundTruthPredicate("IsPlaceable", [_movable_object_type], _dummy_classifier)
_IsNotPlaceable = GroundTruthPredicate("IsNotPlaceable", [_movable_object_type], _dummy_classifier)
_IsSweeper = GroundTruthPredicate("IsSweeper", [_movable_object_type], _dummy_classifier)
_HasFlatTopSurface = GroundTruthPredicate("HasFlatTopSurface", [_base_object_type], _dummy_classifier)
_RobotReadyForSweeping = GroundTruthPredicate("RobotReadyForSweeping", [_robot_type], _dummy_classifier)
# Add predicates for drawer state
_DrawerClosed = GroundTruthPredicate("DrawerClosed", [_container_type], _dummy_classifier)
_DrawerOpen = GroundTruthPredicate("DrawerOpen", [_container_type], _dummy_classifier)

# Add predicates for Inside relation; Keep both world-state and belief-state version!
_Inside = GroundTruthPredicate("Inside", [_movable_object_type, _container_type], _dummy_classifier)
_Unknown_Inside = GroundTruthPredicate("Unknown_Inside", [_movable_object_type, _container_type], _dummy_classifier)
_Known_Inside = GroundTruthPredicate("Known_Inside", [_movable_object_type, _container_type], _dummy_classifier)
_BelieveTrue_Inside = GroundTruthPredicate("BelieveTrue_Inside", [_movable_object_type, _container_type], _dummy_classifier)
_BelieveFalse_Inside = GroundTruthPredicate("BelieveFalse_Inside", [_movable_object_type, _container_type], _dummy_classifier)

# Add new predicates for cup emptiness
_InHandViewFromTop = GroundTruthPredicate("InHandViewFromTop", [_robot_type, _base_object_type], _dummy_classifier)
_ContainerEmpty = GroundTruthPredicate("ContainerEmpty", [_container_type], _dummy_classifier)
# Belief predicates
_ContainingWaterUnknown = GroundTruthPredicate("ContainingWaterUnknown", [_container_type], _dummy_classifier)
_ContainingWaterKnown = GroundTruthPredicate("ContainingWaterKnown", [_container_type], _dummy_classifier)
_ContainingWater = GroundTruthPredicate("ContainingWater", [_container_type], _dummy_classifier)
_NotContainingWater = GroundTruthPredicate("NotContainingWater", [_container_type], _dummy_classifier)

# Add new belief predicates for container emptiness
_Unknown_ContainerEmpty = GroundTruthPredicate("Unknown_ContainerEmpty", [_container_type], _dummy_classifier)
_Known_ContainerEmpty = GroundTruthPredicate("Known_ContainerEmpty", [_container_type], _dummy_classifier)
_BelieveTrue_ContainerEmpty = GroundTruthPredicate("BelieveTrue_ContainerEmpty", [_container_type], _dummy_classifier)
_BelieveFalse_ContainerEmpty = GroundTruthPredicate("BelieveFalse_ContainerEmpty", [_container_type], _dummy_classifier)

# Add belief predicates for Inside
_Unknown_Inside = GroundTruthPredicate("Unknown_Inside", [_movable_object_type, _container_type], _dummy_classifier)
_Known_Inside = GroundTruthPredicate("Known_Inside", [_movable_object_type, _container_type], _dummy_classifier)
_BelieveTrue_Inside = GroundTruthPredicate("BelieveTrue_Inside", [_movable_object_type, _container_type], _dummy_classifier)
_BelieveFalse_Inside = GroundTruthPredicate("BelieveFalse_Inside", [_movable_object_type, _container_type], _dummy_classifier)

# Export all predicates
PREDICATES = {_NEq, _On, _TopAbove, _Inside, _NotInsideAnyContainer, _FitsInXY,
             _HandEmpty, _Holding, _NotHolding, _InHandView, _InView, _Reachable,
             _Blocking, _NotBlocked, _ContainerReadyForSweeping, _IsPlaceable,
             _IsNotPlaceable, _IsSweeper, _HasFlatTopSurface, _RobotReadyForSweeping,
             _DrawerClosed, _DrawerOpen, _InHandViewFromTop, _Inside}
# Note: Now adding belief predicates
BELIEF_PREDICATES = {_Unknown_Inside, _Known_Inside, _BelieveTrue_Inside, _BelieveFalse_Inside,
                     _Unknown_ContainerEmpty, _Known_ContainerEmpty, _BelieveTrue_ContainerEmpty,
                     _BelieveFalse_ContainerEmpty, _InHandViewFromTop, _ContainingWaterUnknown,
                     _ContainingWaterKnown, _ContainingWater, _NotContainingWater}
PREDICATES = PREDICATES | BELIEF_PREDICATES

# Export goal predicates
GOAL_PREDICATES = {_On, _Inside, _ContainingWaterKnown, _Known_ContainerEmpty, _DrawerOpen}  # Add DrawerOpen to goal predicates


def get_vlm_predicates() -> Tuple[Set[VLMPredicate], Set[VLMPredicate]]:
    """Get VLM predicates for mock spot environment."""
    
    # NOTE: VLM Ground Truth predicates
    _On = VLMPredicate(
        "On", [_movable_object_type, _base_object_type],
        prompt=
        "This predicate typically describes a movable object on a flat surface, so it's in conflict with the object being inside a container. Please check the image and confirm the object is on the surface."
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
    _InHandViewFromTop = VLMPredicate(
        "InHandViewFromTop", [_robot_type, _base_object_type],
        prompt="This predicate is true if the camera is viewing the given object (e.g., a container) from the top, so it could see e.g., if the container has anything in it."
    )
    
    # NOTE: let's put Inside also a belief predicate, but keep this one here
    _Inside = VLMPredicate(
        "Inside", [_movable_object_type, _container_type],
        prompt=
        "This typically describes an object (obj1, first arg) inside a container (obj2, second arg) (so it's overlapping), and it's in conflict with the object being on a surface. This is obj1 inside obj2, so obj1 should be smaller than obj2."
    )
    
    _Unknown_Inside = VLMPredicate(
        "Unknown_Inside", [_movable_object_type, _container_type],
        prompt="[Answer: yes/no only] This predicate is true (answer [yes]) if you cannot determine whether the first object is inside the second object (container). If you can tell whether it's inside or not, answer [no]."
    )
    
    _Known_Inside = VLMPredicate(
        "Known_Inside", [_movable_object_type, _container_type],
        prompt="[Answer: yes/no only] This predicate is true (answer [yes]) if you can determine whether the first object is inside the second object (container). If you cannot tell, answer [no]."
    )
    
    _BelieveTrue_Inside = VLMPredicate(
        "BelieveTrue_Inside", [_movable_object_type, _container_type],
        prompt="[Answer: yes/no only] This predicate is true (answer [yes]) if you believe the first object is inside the second object (container) based on what you can see. If you believe it's not inside, answer [no]."
    )
    
    _BelieveFalse_Inside = VLMPredicate(
        "BelieveFalse_Inside", [_movable_object_type, _container_type],
        prompt="[Answer: yes/no only] This predicate is true (answer [yes]) if you believe the first object is not inside the second object (container) based on what you can see. If you believe it is inside, answer [no]."
    )
    
    # NOTE: VLM Belief Predicates for container emptiness
    _ContainingWaterKnown = VLMPredicate(
        "ContainingWaterKnown", [_container_type],
        prompt="[Answer: yes/no only] This predicate is true (answer [yes]) if you know whether the container contains water or not. If you don't know, answer [no]."
    )
    
    _ContainingWaterUnknown = VLMPredicate(
        "ContainingWaterUnknown", [_container_type],
        prompt="[Answer: yes/no only] This predicate is true (answer [yes]) if you do not know whether the container contains water or not. If you know, answer [no]."
    )
    
    _ContainingWater = VLMPredicate(
        "ContainingWater", [_container_type],
        prompt="[Answer: yes/no only] This predicate is true (answer [yes]) if the container has water in it. If you know it doesn't have water, answer [no]."
    )
    
    _NotContainingWater = VLMPredicate(
        "NotContainingWater", [_container_type],
        prompt="[Answer: yes/no only] This predicate is true (answer [yes]) if the container does not have water in it. If it has water, answer [no]."
    )
    
    # NOTE: VLM Belief Predicates for container emptiness
    _Unknown_ContainerEmpty = VLMPredicate(
        "Unknown_ContainerEmpty", [_container_type],
        prompt="[Answer: yes/no only] This predicate is true (answer [yes]) if you do not know whether the container is empty or not. If you can tell whether it's empty or not, answer [no]."
    )
    
    _Known_ContainerEmpty = VLMPredicate(
        "Known_ContainerEmpty", [_container_type],
        prompt="[Answer: yes/no only] This predicate is true (answer [yes]) if you can determine whether the container is empty or contains objects inside. If you cannot tell, answer [no]."
    )
    
    _BelieveTrue_ContainerEmpty = VLMPredicate(
        "BelieveTrue_ContainerEmpty", [_container_type],
        prompt="[Answer: yes/no only] This predicate is true (answer [yes]) if you believe the container is empty based on what you can see. If you believe it contains objects, answer [no]."
    )
    
    _BelieveFalse_ContainerEmpty = VLMPredicate(
        "BelieveFalse_ContainerEmpty", [_container_type],
        prompt="[Answer: yes/no only] This predicate is true (answer [yes]) if you believe the container contains objects based on what you can see. If you believe it is empty, answer [no]."
    )
    
    # NOTE: VLM Predicates for drawer state
    _DrawerClosed = VLMPredicate(
        "DrawerClosed", [_container_type],
        prompt="[Answer: yes/no only] This predicate is true (answer [yes]) if the drawer is closed. If the drawer is open, answer [no]."
    )
    
    _DrawerOpen = VLMPredicate(
        "DrawerOpen", [_container_type],
        prompt="[Answer: yes/no only] This predicate is true (answer [yes]) if the drawer is open. If the drawer is closed, answer [no]."
    )
    
    vlm_predicates = {_On, _Blocking, _NotBlocked, _NotInsideAnyContainer, 
            _Inside, _Unknown_Inside, _Known_Inside, _BelieveTrue_Inside, _BelieveFalse_Inside,
            _ContainingWaterKnown, _ContainingWaterUnknown, _ContainingWater, _NotContainingWater,
            _InHandViewFromTop, _Unknown_ContainerEmpty, _Known_ContainerEmpty,
            _BelieveTrue_ContainerEmpty, _BelieveFalse_ContainerEmpty,
            _DrawerClosed, _DrawerOpen}
    
    belief_predicates = {
        _ContainingWaterUnknown,
        _ContainingWaterKnown,
        _ContainingWater,
        _NotContainingWater,
        _Unknown_ContainerEmpty,
        _Known_ContainerEmpty,
        _BelieveTrue_ContainerEmpty,
        _BelieveFalse_ContainerEmpty,
        _Inside,
        _Unknown_Inside,
        _Known_Inside,
        _BelieveTrue_Inside,
        _BelieveFalse_Inside
    }
    
    return vlm_predicates, belief_predicates

# Export VLM predicates
VLM_PREDICATES, VLM_BELIEF_PREDICATES = get_vlm_predicates()

# Export all predicates - use VLM or non-VLM based on config
def get_all_predicates() -> Set[Predicate]:
    """Get all predicates based on config."""
    if CFG.mock_env_vlm_eval_predicate:
        # Filter out non-VLM counterparts of VLM predicates
        vlm_names = {pred.name for pred in VLM_PREDICATES}
        non_vlm_preds = {pred for pred in PREDICATES if pred.name not in vlm_names}
        return non_vlm_preds | VLM_PREDICATES
    return PREDICATES

def get_fluent_predicates(operators: Set[STRIPSOperator]) -> Set[Predicate]:
    """Calculate fluent predicates by looking at operator effects.
    Similar to Fast Downward's get_fluents function.
    
    Returns:
        Set of Predicate objects that appear in operator effects.
    """
    fluent_predicates = set()
    # Look at all operators
    for op in operators:
        # Add predicates that appear in add or delete effects
        for effect in op.add_effects:
            fluent_predicates.add(effect.predicate)
        for effect in op.delete_effects:
            fluent_predicates.add(effect.predicate)
    return fluent_predicates

def get_active_predicates(operators: Set[STRIPSOperator]) -> Set[Predicate]:
    """Get active predicates by looking at operator preconditions and effects.
    """
    active_predicates = set()
    # Look at all operators
    for op in operators:
        # Add predicates that appear in preconditions
        for precondition in op.preconditions:
            active_predicates.add(precondition.predicate)
        # Add predicates that appear in add or delete effects
        for effect in op.add_effects:
            active_predicates.add(effect.predicate)
        for effect in op.delete_effects:
            active_predicates.add(effect.predicate)
    return active_predicates

# Update PREDICATES_WITH_VLM to use function
PREDICATES_WITH_VLM = get_all_predicates() if CFG.mock_env_vlm_eval_predicate else None


@dataclass(frozen=True)
class _SavedMockSpotObservation:
    """Observation for mock Spot environment.
    
    Contains:
    - Images (RGB/depth) with metadata (rotations, camera info)
    - Robot state (gripper, objects in view/hand)
    - Predicate atoms (VLM and non-VLM)
    - State metadata
    
    Directory structure for each state:
        state_0/
        ├── state_metadata.json  # Robot state, objects, atoms
        ├── image_metadata.json  # Image paths and transforms
        ├── view1_cam1_rgb.npy  # RGB image data
        └── view1_cam1_depth.npy  # Optional depth data
    """
    images: Optional[Dict[str, UnposedImageWithContext]]
    gripper_open: bool
    objects_in_view: Set[Object]
    objects_in_hand: Set[Object]
    state_id: str
    atom_dict: Dict[str, bool]
    non_vlm_atom_dict: Dict[GroundAtom, bool]
    metadata: Dict[str, Any]  # = field(default_factory=dict)

    def _serialize_ground_atom(self, atom: GroundAtom) -> str:
        """Convert a GroundAtom to a string representation."""
        pred_name = atom.predicate.name
        obj_names = [obj.name for obj in atom.objects]
        return f"{pred_name}({','.join(obj_names)})"

    def _serialize_atom_dict(self, atom_dict: Dict[GroundAtom, bool]) -> Dict[str, bool]:
        """Convert a dict with GroundAtom keys to string keys."""
        return {self._serialize_ground_atom(atom): value 
                for atom, value in atom_dict.items()}
        
    @staticmethod
    def _deserialize_ground_atom(atom_str: str, objects: Dict[str, Object]) -> GroundAtom:
        """Convert a string representation back to a GroundAtom."""
        pred_name = atom_str[:atom_str.index("(")]
        obj_names = atom_str[atom_str.index("(")+1:atom_str.index(")")].split(",")
        # Get the predicate from the PREDICATES set
        print(PREDICATES)
        print(pred_name)
        preds = [p for p in PREDICATES if p.name == pred_name]
        assert len(preds)>0
  
        objs = [objects[name] for name in obj_names]
        return GroundAtom(preds[0], objs)

    def save_state(self, save_dir: Optional[Path] = None) -> None:
        """Save state data and metadata.
        
        Saves:
        1. State metadata (gripper state, objects, atoms)
        2. Image data (RGB/depth as .npy)
        3. Image metadata (rotations, transforms)
        
        Args:
            save_dir: Directory to save state data. If None, uses directory from metadata.
        """
        # Update save directory if provided
        if save_dir is not None:
            self.metadata["save_dir"] = str(save_dir)
        elif "save_dir" not in self.metadata:
            raise ValueError("No save directory specified")
            
        if save_dir is None:
            save_dir = Path(self.metadata["save_dir"])
        state_dir = save_dir / self.state_id
        state_dir.mkdir(exist_ok=True)

        # Save state metadata
        state_metadata = {
            "gripper_open": self.gripper_open,
            "objects_in_view": [obj.name for obj in self.objects_in_view],
            "objects_in_hand": [obj.name for obj in self.objects_in_hand],
            "atom_dict": self.atom_dict,
            "non_vlm_atom_dict": self._serialize_atom_dict(self.non_vlm_atom_dict) if self.non_vlm_atom_dict else None
        }
        with open(state_dir / "state_metadata.yaml", "w") as f:
            yaml.dump(state_metadata, f, default_flow_style=False)

        # Save images and their metadata
        image_metadata = {}
        if self.images:
            for img_name, img_data in self.images.items():
                # Save RGB
                rgb_path = state_dir / f"{img_name}_rgb.npy"
                np.save(rgb_path, img_data.rgb)
                
                # Also save as JPG for preview
                jpg_path = state_dir / f"{img_name}_rgb.jpg"
                plt.imsave(str(jpg_path), img_data.rgb)
                
                # Save depth if exists
                depth_path = None
                if img_data.depth is not None:
                    depth_path = state_dir / f"{img_name}_depth.npy"
                    np.save(depth_path, img_data.depth)
                
                # Track image metadata
                image_metadata[img_name] = {
                    "rgb_path": str(rgb_path.relative_to(state_dir)),
                    "depth_path": str(depth_path.relative_to(state_dir)) if depth_path else None,
                    "camera_name": img_data.camera_name,
                    "image_rot": img_data.image_rot
                }

        # Save image metadata
        with open(state_dir / "image_metadata.yaml", "w") as f:
            yaml.dump(image_metadata, f, default_flow_style=False)

    @classmethod
    def load_state(cls, state_id: str, save_dir: Path, objects: Dict[str, Object]) -> "_SavedMockSpotObservation":
        """Load state data and metadata.
        
        Args:
            state_id: ID of state to load
            save_dir: Directory containing saved states
            objects: Dict mapping object names to Object instances
            
        Returns:
            Loaded observation with images and metadata
            
        Raises:
            FileNotFoundError: If state directory or metadata files don't exist
        """
        state_dir = save_dir / state_id
        if not state_dir.exists():
            raise FileNotFoundError(f"State directory not found: {state_dir}")

        # Load state metadata
        state_meta_path = state_dir / "state_metadata.yaml"
        if not state_meta_path.exists():
            raise FileNotFoundError(f"State metadata not found: {state_meta_path}")
        with open(state_meta_path) as f:
            state_metadata = yaml.safe_load(f)

        # Load image metadata
        image_meta_path = state_dir / "image_metadata.yaml"
        if not image_meta_path.exists():
            raise FileNotFoundError(f"Image metadata not found: {image_meta_path}")
        with open(image_meta_path) as f:
            image_metadata = yaml.safe_load(f)

        # Load images
        images = {}
        for img_name, img_meta in image_metadata.items():
            rgb_path = state_dir / img_meta["rgb_path"]
            depth_path = state_dir / img_meta["depth_path"] if img_meta["depth_path"] else None

            images[img_name] = UnposedImageWithContext(
                rgb=np.load(rgb_path),
                depth=np.load(depth_path) if depth_path else None,
                camera_name=img_meta["camera_name"],
                image_rot=img_meta["image_rot"]
            )

        # Deserialize non_vlm_atom_dict if it exists
        non_vlm_atom_dict = {}  # Default to empty dict instead of None
        if state_metadata.get("non_vlm_atom_dict"):
            non_vlm_atom_dict = {
                cls._deserialize_ground_atom(atom_str, objects): bool(value)
                for atom_str, value in state_metadata["non_vlm_atom_dict"].items()
            }

        return cls(
            images=images,
            gripper_open=state_metadata["gripper_open"],
            objects_in_view={objects[name] for name in state_metadata["objects_in_view"]},
            objects_in_hand={objects[name] for name in state_metadata["objects_in_hand"]},
            state_id=state_id,
            atom_dict=state_metadata["atom_dict"],
            metadata={"save_dir": str(save_dir)},
            non_vlm_atom_dict=non_vlm_atom_dict
        )


@dataclass(frozen=True)
class _MockSpotObservation(_SavedMockSpotObservation):
    """An observation from the mock Spot environment."""
    object_dict: Dict[str, Object]
    vlm_atom_dict: Optional[Dict[VLMGroundAtom, bool]] = None
    vlm_predicates: Optional[Set[VLMPredicate]] = None
    action_history: Optional[List[Action]] = None
    
    @classmethod
    def init_from_saved(cls, saved_obs: _SavedMockSpotObservation, object_dict: Dict[str, Object],
                       vlm_atom_dict: Optional[Dict[VLMGroundAtom, bool]] = None,
                       vlm_predicates: Optional[Set[VLMPredicate]] = None,
                       action_history: Optional[List[Action]] = None) -> "_MockSpotObservation":
        """Initialize from a saved observation.
        
        Note: VLM predicates will be evaluated by the perceiver when needed, not here.
        """
        return cls(
            images=saved_obs.images,
            gripper_open=saved_obs.gripper_open,
            objects_in_view=saved_obs.objects_in_view,
            objects_in_hand=saved_obs.objects_in_hand,
            state_id=saved_obs.state_id,
            atom_dict=saved_obs.atom_dict,
            non_vlm_atom_dict=saved_obs.non_vlm_atom_dict,
            metadata=saved_obs.metadata,
            # Fields decided by VLM online evaluation
            object_dict=object_dict,
            vlm_atom_dict=vlm_atom_dict,  # Will be populated by perceiver when needed
            vlm_predicates=vlm_predicates,
            action_history=action_history
        )