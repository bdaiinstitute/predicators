"""Mock perceiver for the Spot environment.

This perceiver simulates perception for the mock Spot environment by providing a simplified
interface for testing and development without requiring the actual Spot robot hardware.

Key Features:
1. RGBD Image Management:
   - Stores and returns mock RGBD images with camera context
   - Supports image rotation and transformation data
   - Maintains depth information for 3D perception

2. State Tracking:
   - Objects currently in the robot's view
   - Objects currently held in the gripper
   - Gripper state (open/closed)

3. Mock Environment Integration:
   - Works with MockSpotEnv for end-to-end testing
   - Supports transition verification in the mock environment
   - Enables testing of perception-dependent behaviors

Usage:
    perceiver = MockSpotPerceiver(data_dir="/path/to/images")
    
    # Update state based on environment changes
    perceiver.update_state(
        gripper_open=True,
        objects_in_view={"cup", "table"},
        objects_in_hand=set()
    )
    
    # Get current observation
    obs = perceiver.get_observation()
    assert obs.gripper_open
    assert "cup" in obs.objects_in_view
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np

from predicators.envs import get_or_create_env
from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext
from predicators.perception.base_perceiver import BasePerceiver
from predicators.structs import Action, DefaultState, EnvironmentTask, GoalDescription, GroundAtom, Observation, Predicate, State, Task, Video
from predicators.settings import CFG


class MockSpotPerceiver(BasePerceiver):
    """A mock perceiver for the Spot environment that returns simulated observations.
    
    This class provides a simplified interface for testing perception-dependent behaviors
    without requiring the actual Spot robot. It maintains the current state of the
    environment including:
    - Current RGBD image with camera context
    - Objects visible to the robot
    - Objects held in the gripper
    - Gripper state (open/closed)
    - VLM predicates and atoms for perception-based planning
    """

    def __init__(self, data_dir: str) -> None:
        """Initialize the mock perceiver.
        
        Args:
            data_dir: Directory to store/load mock images. This directory should contain
                     the RGBD images that will be used during testing. Images can be
                     saved here using the save_image method.
        """
        super().__init__()
        self._data_dir = Path(data_dir)
        self._current_rgbd: Optional[RGBDImageWithContext] = None
        self._gripper_open: bool = True
        self._objects_in_view: Set[str] = set()
        self._objects_in_hand: Set[str] = set()
        
        # VLM-related state
        self._camera_images = None
        self._vlm_atom_dict = None
        self._vlm_predicates = None
        self._curr_env = None

    @classmethod
    def get_name(cls) -> str:
        """Get the name of this perceiver."""
        return "mock_spot"

    def get_observation(self) -> "MockSpotObservation":
        """Get the current observation of the environment."""
        return MockSpotObservation(
            rgbd=self._current_rgbd,
            gripper_open=self._gripper_open,
            objects_in_view=self._objects_in_view,
            objects_in_hand=self._objects_in_hand,
            images=self._camera_images,
            vlm_atom_dict=self._vlm_atom_dict,
            vlm_predicates=self._vlm_predicates
        )

    def save_image(self, rgbd: RGBDImageWithContext) -> None:
        """Save a mock RGBD image to be returned in future observations."""
        self._current_rgbd = rgbd

    def update_state(self, gripper_open: bool, objects_in_view: Set[str], 
                    objects_in_hand: Set[str], camera_images=None,
                    vlm_atom_dict=None, vlm_predicates=None) -> None:
        """Update the current state of the environment."""
        self._gripper_open = gripper_open
        self._objects_in_view = objects_in_view
        self._objects_in_hand = objects_in_hand
        if CFG.spot_vlm_eval_predicate:
            self._camera_images = camera_images
            self._vlm_atom_dict = vlm_atom_dict
            self._vlm_predicates = vlm_predicates

    def reset(self, env_task: EnvironmentTask) -> Task:
        """Reset the perceiver for a new task."""
        self._current_rgbd = None
        self._gripper_open = True
        self._objects_in_view = set()
        self._objects_in_hand = set()
        self._camera_images = None
        self._vlm_atom_dict = None
        self._vlm_predicates = None
        self._curr_env = get_or_create_env(CFG.env)
        # NOTE: this seems to come from "dry run" version - we don't need it here! check and remove
        return env_task.task

    def update_perceiver_with_action(self, action: Action) -> None:
        """Update the perceiver with an action."""
        pass  # No action tracking needed for mock perceiver

    def step(self, observation: Observation) -> State:
        """Process a new observation and return the current state."""
        assert isinstance(observation, MockSpotObservation)
        if CFG.spot_vlm_eval_predicate:
            self._camera_images = observation.images
            self._vlm_atom_dict = observation.vlm_atom_dict
            self._vlm_predicates = observation.vlm_predicates
        return DefaultState

    def render_mental_images(self, observation: Optional[Observation] = None,
                           env_task: Optional[EnvironmentTask] = None) -> Video:
        """Render mental images for visualization."""
        return []  # No mental image rendering needed for mock perceiver


@dataclass
class MockSpotObservation:
    """An observation from the mock Spot environment."""
    rgbd: Optional[RGBDImageWithContext]
    gripper_open: bool
    objects_in_view: Set[str]
    objects_in_hand: Set[str]
    images: Optional[Dict] = None
    vlm_atom_dict: Optional[Dict] = None
    vlm_predicates: Optional[Set[Predicate]] = None
