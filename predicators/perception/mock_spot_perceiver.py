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

from typing import Dict, Optional, Set, Union
import logging

from predicators.spot_utils.perception.object_perception import get_vlm_atom_combinations, vlm_predicate_batch_classify
from predicators.spot_utils.perception.perception_structs import UnposedImageWithContext
from predicators.perception.base_perceiver import BasePerceiver
from predicators.structs import Action, EnvironmentTask, State, Task, VLMPredicate, VLMGroundAtom, Object
from predicators.settings import CFG
from predicators.envs.mock_spot_env import _MockSpotObservation
from predicators.envs.spot_env import _robot_type


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
        
        self._spot_object: Object = Object("robot", _robot_type)
        
        # VLM-related state
        self._vlm_atom_dict: Dict[VLMGroundAtom, bool] = {}  # Current VLM predicate evaluations
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
            # Update VLM predicates (It shouldn't be updated normally)
            if obs.vlm_predicates is not None:
                self._vlm_predicates.update(obs.vlm_predicates)

            vlm_predicates = self._vlm_predicates
            assert vlm_predicates is not None
            
            visible_objects = list(obs.objects_in_view) + [self._spot_object]
            images = obs.images
            assert images is not None
            
            # Compute latest VLM atoms based on visible objects in the current observation
            vlm_atoms = get_vlm_atom_combinations(visible_objects, vlm_predicates)
            # NOTE: There may be new objects, and the set of VLMAtoms will be expanded
            curr_vlm_atom_values: Union[Dict[VLMGroundAtom, bool], Set[VLMGroundAtom]] = vlm_predicate_batch_classify(
                                   vlm_atoms,
                                   images,  # type: ignore
                                   predicates=vlm_predicates,
                                   get_dict=True)
            assert isinstance(curr_vlm_atom_values, dict)
            
            # [Belief Update: Update predicate labels]
            # First get the latest VLM atom values from current observation
            updated_vlm_atom_values = self._vlm_atom_dict.copy()  # Start with previous values
            
            # Update with new values from current observation's VLM evaluation
            # This preserves knowledge from previous observations
            # NOTE: We assume no information loss in belief update
            # NOTE: Rule: Known predicates cannot become unknown
            # Step 1: Basic update - update any atom that has a non-None value
            if obs.vlm_atom_dict is not None:
                for atom, value in curr_vlm_atom_values.items():
                    if value is not None:
                        updated_vlm_atom_values[atom] = value
            
            # Step 2: Handle Known/Unknown predicates
            # Collect all Known/Unknown predicate pairs
            known_unknown_pairs = {}  # Dict[str, tuple[VLMGroundAtom, VLMGroundAtom]]
            for atom in updated_vlm_atom_values:
                if isinstance(atom, VLMGroundAtom):
                    pred_name = atom.predicate.name
                    if pred_name.startswith("Known_"):
                        base_name = pred_name.replace("Known_", "")
                        if base_name not in known_unknown_pairs:
                            known_unknown_pairs[base_name] = [None, None]
                        known_unknown_pairs[base_name][0] = atom
                    elif pred_name.startswith("Unknown_"):
                        base_name = pred_name.replace("Unknown_", "")
                        if base_name not in known_unknown_pairs:
                            known_unknown_pairs[base_name] = [None, None]
                        known_unknown_pairs[base_name][1] = atom

            # Update Known/Unknown pairs based on previous knowledge
            for base_name, (known_atom, unknown_atom) in known_unknown_pairs.items():
                if known_atom is not None and unknown_atom is not None:
                    # If it was known before, keep it known
                    if known_atom in self._vlm_atom_dict and self._vlm_atom_dict[known_atom]:
                        updated_vlm_atom_values[known_atom] = True
                        updated_vlm_atom_values[unknown_atom] = False
                    
                    # Validate Known/Unknown consistency
                    known_val = updated_vlm_atom_values.get(known_atom, False)
                    unknown_val = updated_vlm_atom_values.get(unknown_atom, True)
                    if known_val == unknown_val:
                        logging.warning(
                            f"Inconsistent Known/Unknown values for {base_name}: "
                            f"Known={known_val}, Unknown={unknown_val}"
                        )
            
            # Store updated values
            self._vlm_atom_dict = updated_vlm_atom_values
        
        # Create state with all atoms
        state_dict = {}
        
        # Add VLM atoms if enabled
        if CFG.spot_vlm_eval_predicate and self._vlm_atom_dict:
            for atom, value in self._vlm_atom_dict.items():
                if value:
                    state_dict[atom] = True
        
        # Add non-VLM atoms
        if self._non_vlm_atoms:
            for atom in self._non_vlm_atoms:
                state_dict[atom] = True
        
        return State(state_dict)
