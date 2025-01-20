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
   - VLM predicates and atoms for perception-based planning
   - Non-VLM predicates from environment

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
from typing import Dict, Optional, Set, Container

import numpy as np

from predicators.envs import get_or_create_env
from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext
from predicators.perception.base_perceiver import BasePerceiver
from predicators.structs import Action, EnvironmentTask, GoalDescription, GroundAtom, Observation, Predicate, State, Task, Video, VLMPredicate, VLMGroundAtom
from predicators.settings import CFG
from predicators.envs.mock_spot_env import _MockSpotObservation


class MockSpotPerceiver(BasePerceiver):
    """A mock perceiver for the Spot environment that returns simulated observations."""

    def __init__(self) -> None:
        """Initialize the mock perceiver."""
        super().__init__()
        # Current observation state
        self._current_rgbd: Optional[RGBDImageWithContext] = None
        self._gripper_open: bool = True
        self._objects_in_view: Set[str] = set()
        self._objects_in_hand: Set[str] = set()
        
        # VLM-related state
        self._camera_images = None
        self._vlm_atom_dict: Dict = {}  # Current VLM predicate evaluations
        self._vlm_predicates: Set[Predicate] = set()  # Current VLM predicates
        self._non_vlm_atoms: Set[GroundAtom] = set()  # Non-VLM atoms from env
        
        # Environment reference
        self._curr_env = None

    @classmethod
    def get_name(cls) -> str:
        """Get the name of this perceiver."""
        return "mock_spot"
    
    def reset(self, env_task: EnvironmentTask) -> Task:
        """Reset the perceiver for a new task.
        
        This resets all internal state but preserves the environment reference.
        """
        self._current_rgbd = None
        self._gripper_open = True
        self._objects_in_view = set()
        self._objects_in_hand = set()
        self._camera_images = None
        self._vlm_atom_dict = {}
        self._vlm_predicates = set()
        self._non_vlm_atoms = set()
        
        # Keep or create environment reference
        if self._curr_env is None:
            self._curr_env = get_or_create_env(CFG.env)
        
        return env_task.task
    
    def step(self, observation: _MockSpotObservation) -> State:
        """Process a new observation and return the current state.
        
        This is the main function that:
        1. Updates internal state from observation
        2. Processes VLM predicates if enabled
        3. Returns the new state
        """
        assert isinstance(observation, _MockSpotObservation)
        
        # Update internal state from observation
        self._update_state_from_observation(observation)
        
        # Convert observation to state
        return self._obs_to_state(observation)

    def get_observation(self) -> _MockSpotObservation:
        """Get the current observation of the environment."""
        # Convert predicates to VLM predicates if enabled
        vlm_preds: Optional[Set[VLMPredicate]] = None
        if CFG.spot_vlm_eval_predicate and self._vlm_predicates:
            vlm_preds = {p for p in self._vlm_predicates if isinstance(p, VLMPredicate)}
            
        return _MockSpotObservation(
            images=self._camera_images,
            gripper_open=self._gripper_open,
            objects_in_view=self._objects_in_view,
            objects_in_hand=self._objects_in_hand,
            state_id="mock",  # Mock state ID since perceiver doesn't track it
            atom_dict={},  # Empty since handled by environment
            non_vlm_atom_dict=self._non_vlm_atoms,
            vlm_atom_dict=self._vlm_atom_dict,
            vlm_predicates=vlm_preds
        )

    def _update_state_from_observation(self, obs: _MockSpotObservation) -> None:
        """Update internal state from an observation."""
        # Update basic state
        self._camera_images = obs.images
        self._gripper_open = obs.gripper_open
        self._objects_in_view = obs.objects_in_view
        self._objects_in_hand = obs.objects_in_hand
        
        # Update non-VLM atoms directly from environment
        if obs.non_vlm_atom_dict is not None:
            self._non_vlm_atoms = obs.non_vlm_atom_dict
            
        # Handle VLM predicates if enabled
        if CFG.spot_vlm_eval_predicate:
            # Update VLM predicates while preserving Unknown predicates
            if obs.vlm_predicates is not None:
                self._vlm_predicates.update(obs.vlm_predicates)
            
            # Update VLM atoms while preserving Unknown states
            if obs.vlm_atom_dict is not None:
                for atom, value in obs.vlm_atom_dict.items():
                    # Only update if atom is a VLM ground atom
                    if isinstance(atom, VLMGroundAtom):
                        self._vlm_atom_dict[atom] = value

    def _obs_to_state(self, obs: _MockSpotObservation) -> State:
        """Convert observation to state."""
        # Start with empty state dict
        state_dict = {}
        
        # Create state with both VLM and non-VLM predicates
        state = State(state_dict)
        
        # Add VLM atoms if enabled
        if CFG.spot_vlm_eval_predicate and self._vlm_atom_dict:
            for atom, value in self._vlm_atom_dict.items():
                if value:  # Only add True atoms
                    state = state.copy()
                    state.set_atoms({atom})
                
        # Add non-VLM atoms from environment
        if self._non_vlm_atoms:
            state = state.copy()
            state.set_atoms(self._non_vlm_atoms)
            
        return state

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
            # Update VLM state while preserving Unknown predicates
            if vlm_atom_dict is not None:
                for pred, value in vlm_atom_dict.items():
                    if not pred.startswith("Unknown"):
                        self._vlm_atom_dict[pred] = value
            if vlm_predicates is not None:
                self._vlm_predicates.update(vlm_predicates)

    def update_perceiver_with_action(self, action: Action) -> None:
        """Update the perceiver with an action."""
        pass  # No action tracking needed for mock perceiver

    def render_mental_images(self, observation: Optional[Observation] = None,
                           env_task: Optional[EnvironmentTask] = None) -> Video:
        """Render mental images for visualization."""
        return []  # No mental image rendering needed for mock perceiver
