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

from typing import Dict, Optional, Set

from predicators.envs import get_or_create_env
from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext
from predicators.perception.base_perceiver import BasePerceiver
from predicators.structs import Action, EnvironmentTask, State, Task, VLMPredicate, VLMGroundAtom, Object
from predicators.settings import CFG
from predicators.envs.mock_spot_env import _MockSpotObservation


class MockSpotPerceiver(BasePerceiver):
    """A mock perceiver for the Spot environment that returns simulated observations."""

    def __init__(self) -> None:
        """Initialize the mock perceiver."""
        super().__init__()
        # Current observation state
        self._camera_images = None
        self._gripper_open: bool = True
        self._objects_in_view: Set[Object] = set()
        self._objects_in_hand: Set[Object] = set()
        
        # VLM-related state
        self._vlm_atom_dict: Dict = {}  # Current VLM predicate evaluations
        self._vlm_predicates: Set[VLMPredicate] = set()  # Current VLM predicates
        self._non_vlm_atoms: Set = set()  # Non-VLM atoms from env

    @classmethod
    def get_name(cls) -> str:
        """Get the name of this perceiver."""
        return "mock_spot"
    
    def reset(self, env_task: EnvironmentTask) -> Task:
        """Reset the perceiver for a new task."""
        init_state = self._obs_to_state(env_task.init_obs)
        goal_description = env_task.goal_description
        
        return Task(init_state, goal_description)
    
    def step(self, observation: _MockSpotObservation) -> State:
        """Process a new observation and return the current state."""
        assert isinstance(observation, _MockSpotObservation)
        return self._obs_to_state(observation)

    def _obs_to_state(self, obs: _MockSpotObservation) -> State:
        """Convert observation to state and update internal state.
        
        This function:
        1. Updates internal perceiver state from observation
        2. Processes VLM predicates and atoms if enabled, preserving previous values
        3. Creates and returns a state with all atoms
        
        Args:
            obs: The current observation containing all state information
            
        Returns:
            The complete state with all atoms
        """
        # Update internal state from observation
        self._camera_images = obs.images
        self._gripper_open = obs.gripper_open
        self._objects_in_view = obs.objects_in_view
        self._objects_in_hand = obs.objects_in_hand
        
        # Update non-VLM atoms
        # NOTE: They should be fully-observable properties and don't rely on past step values
        if obs.non_vlm_atom_dict is not None:
            self._non_vlm_atoms = obs.non_vlm_atom_dict
            
        # Handle VLM predicates and atoms if enabled
        # NOTE: VLM predicates are partially observable and need belief-state update given past step
        if CFG.spot_vlm_eval_predicate:
            # Update VLM predicates
            if obs.vlm_predicates is not None:
                self._vlm_predicates.update(obs.vlm_predicates)
            
            # Update VLM atoms while preserving previous values
            if obs.vlm_atom_dict is not None:
                # Copy current VLM atom dict to preserve previous values
                vlm_atom_return = self._vlm_atom_dict.copy()
                
                # Update with new values from observation
                for atom, value in obs.vlm_atom_dict.items():
                    if value is not None:  # Only update if we have a definite value
                        vlm_atom_return[atom] = value
                    
                # Store updated values
                self._vlm_atom_dict = vlm_atom_return
        
        # Create state with all atoms
        state_dict = {}
        
        # Add VLM atoms if enabled
        if CFG.spot_vlm_eval_predicate and self._vlm_atom_dict:
            # Add True VLM atoms to state dict
            for atom, value in self._vlm_atom_dict.items():
                if value:
                    state_dict[atom] = True
        
        # Add non-VLM atoms
        if self._non_vlm_atoms:
            for atom in self._non_vlm_atoms:
                state_dict[atom] = True
        
        return State(state_dict)
