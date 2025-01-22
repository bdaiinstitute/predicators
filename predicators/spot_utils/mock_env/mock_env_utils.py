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
import matplotlib.pyplot as plt

from predicators.envs import BaseEnv
from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext, UnposedImageWithContext
from predicators.structs import Action, State, Object, Type, EnvironmentTask, Video, Image
from predicators.structs import LiftedAtom, STRIPSOperator, Variable, Predicate, GroundAtom, VLMPredicate, VLMGroundAtom
from predicators.settings import CFG
from bosdyn.client import math_helpers
from predicators.spot_utils.perception.object_perception import get_vlm_atom_combinations, vlm_predicate_batch_classify
from predicators.utils import get_object_combinations



@dataclass(frozen=False)
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
    non_vlm_atom_dict: Optional[Set[GroundAtom]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

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
        }
        with open(state_dir / "state_metadata.json", "w") as f:
            json.dump(state_metadata, f, indent=2)

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
        with open(state_dir / "image_metadata.json", "w") as f:
            json.dump(image_metadata, f, indent=2)

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
        state_meta_path = state_dir / "state_metadata.json"
        if not state_meta_path.exists():
            raise FileNotFoundError(f"State metadata not found: {state_meta_path}")
        with open(state_meta_path) as f:
            state_metadata = json.load(f)

        # Load image metadata
        image_meta_path = state_dir / "image_metadata.json"
        if not image_meta_path.exists():
            raise FileNotFoundError(f"Image metadata not found: {image_meta_path}")
        with open(image_meta_path) as f:
            image_metadata = json.load(f)

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

        return cls(
            images=images,
            gripper_open=state_metadata["gripper_open"],
            objects_in_view={objects[name] for name in state_metadata["objects_in_view"]},
            objects_in_hand={objects[name] for name in state_metadata["objects_in_hand"]},
            state_id=state_id,
            atom_dict=state_metadata["atom_dict"],
            metadata={"save_dir": str(save_dir)}
        )