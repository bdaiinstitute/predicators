"""Mock environment for Spot robot."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import cv2
from gym.spaces import Box
from PIL import Image as PILImage

from predicators.envs import BaseEnv
from predicators.structs import Action, State, Object, Type, EnvironmentTask, Video, Image


@dataclass
class MockSpotObservation:
    """Observation for mock Spot environment."""
    rgb_image: Optional[np.ndarray]  # HxWx3 RGB image
    depth_image: Optional[np.ndarray]  # HxW depth image in meters
    gripper_open: bool  # Whether the gripper is open
    objects_in_view: Set[str]  # Names of objects visible in the image
    objects_in_hand: Set[str]  # Names of objects currently held
    state_id: str  # Unique ID for this observation's latent state


class MockSpotEnv(BaseEnv):
    """Mock environment for Spot robot.
    
    This environment is a POMDP where:
    - States are latent (we don't know actual poses)
    - Observations are RGB-D images + gripper state + object detections
    - Actions can succeed or fail based on available images
    """

    def __init__(self, data_dir: str = "spot_mock_data") -> None:
        super().__init__()
        
        # Create data directories
        self._data_dir = Path(data_dir)
        self._images_dir = self._data_dir / "images"
        self._images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment state
        self._current_state_id: Optional[str] = None
        self._gripper_open: bool = True
        self._objects_in_hand: Set[str] = set()
        
        # Load or initialize transition graph
        self._transitions: Dict[str, Dict[str, str]] = {}  # state_id -> {action -> next_state_id}
        self._observations: Dict[str, MockSpotObservation] = {}  # state_id -> observation
        self._load_graph_data()

    @property
    def action_space(self) -> Box:
        """Return action space."""
        return Box(low=0, high=1, shape=(1,))

    def simulate(self, state: State, action: Action) -> State:
        """Simulate a state transition."""
        raise NotImplementedError("Mock environment does not support simulation")

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        """Generate training tasks."""
        return []

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        """Generate test tasks."""
        return []

    @property
    def predicates(self) -> Set[Any]:
        """Return predicates."""
        return set()

    @property
    def goal_predicates(self) -> Set[Any]:
        """Return goal predicates."""
        return set()

    @property
    def types(self) -> Set[Type]:
        """Return types."""
        return set()

    @property
    def percept_predicates(self) -> Set[Any]:
        """Return percept predicates."""
        return set()

    def _load_graph_data(self) -> None:
        """Load transition graph and observations from disk."""
        graph_file = self._data_dir / "graph.json"
        if not graph_file.exists():
            return
            
        import json
        with open(graph_file) as f:
            data = json.load(f)
            
        # Load transitions
        self._transitions = data["transitions"]
        
        # Load observations
        for state_id, obs_data in data["observations"].items():
            # Load images if they exist
            rgb_path = self._images_dir / f"rgb_{state_id}.npy"
            depth_path = self._images_dir / f"depth_{state_id}.npy"
            
            rgb_image = np.load(rgb_path) if rgb_path.exists() else None
            depth_image = np.load(depth_path) if depth_path.exists() else None
            
            self._observations[state_id] = MockSpotObservation(
                rgb_image=rgb_image,
                depth_image=depth_image,
                gripper_open=obs_data["gripper_open"],
                objects_in_view=set(obs_data["objects_in_view"]),
                objects_in_hand=set(obs_data["objects_in_hand"]),
                state_id=state_id
            )

    def _save_graph_data(self) -> None:
        """Save transition graph and observations to disk."""
        # Convert data to serializable format
        observations_data = {}
        for state_id, obs in self._observations.items():
            # Save images separately
            if obs.rgb_image is not None:
                np.save(self._images_dir / f"rgb_{state_id}.npy", obs.rgb_image)
            if obs.depth_image is not None:
                np.save(self._images_dir / f"depth_{state_id}.npy", obs.depth_image)
            
            observations_data[state_id] = {
                "gripper_open": obs.gripper_open,
                "objects_in_view": list(obs.objects_in_view),
                "objects_in_hand": list(obs.objects_in_hand)
            }
        
        data = {
            "transitions": self._transitions,
            "observations": observations_data
        }
        
        # Save to file
        import json
        with open(self._data_dir / "graph.json", "w") as f:
            json.dump(data, f, indent=2)

    def add_state(self, 
                 rgb_image: Optional[np.ndarray] = None,
                 depth_image: Optional[np.ndarray] = None,
                 gripper_open: bool = True,
                 objects_in_view: Optional[Set[str]] = None,
                 objects_in_hand: Optional[Set[str]] = None) -> str:
        """Add a new state to the environment.
        
        Returns:
            state_id: Unique ID for the new state
        """
        # Generate unique state ID
        state_id = str(len(self._observations))
        
        # Create observation
        self._observations[state_id] = MockSpotObservation(
            rgb_image=rgb_image,
            depth_image=depth_image,
            gripper_open=gripper_open,
            objects_in_view=objects_in_view or set(),
            objects_in_hand=objects_in_hand or set(),
            state_id=state_id
        )
        
        # Initialize empty transitions
        if state_id not in self._transitions:
            self._transitions[state_id] = {}
            
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
            
        self._transitions[from_state_id][action_name] = to_state_id
        self._save_graph_data()

    def get_observation(self) -> MockSpotObservation:
        """Get current observation."""
        if self._current_state_id is None:
            raise ValueError("Environment not initialized")
        return self._observations[self._current_state_id]

    def reset(self, train_or_test: str, task_idx: int) -> MockSpotObservation:
        """Reset environment to initial state."""
        if not self._observations:
            raise ValueError("No states added to environment")
            
        # For now, just pick the first state as initial
        state_id = list(self._observations.keys())[0]
        self._current_state_id = state_id
        obs = self._observations[state_id]
        self._gripper_open = obs.gripper_open
        self._objects_in_hand = obs.objects_in_hand.copy()
        
        return obs

    def step(self, action: Action) -> Tuple[MockSpotObservation, float, bool]:
        """Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation: New observation
            reward: Always 0 (no reward function defined)
            done: Always False (no termination condition)
        """
        if self._current_state_id is None:
            raise ValueError("Environment not initialized")
            
        # Get action name from extra info
        if action.extra_info is None:
            raise ValueError("Action must have extra_info with 'name' field")
        action_name = action.extra_info.get("name")
        if action_name is None:
            raise ValueError("Action extra_info must have 'name' field")
            
        # Check if transition exists
        if action_name not in self._transitions[self._current_state_id]:
            # No transition available - action fails
            return self.get_observation(), 0.0, False
            
        # Update state
        self._current_state_id = self._transitions[self._current_state_id][action_name]
        obs = self._observations[self._current_state_id]
        self._gripper_open = obs.gripper_open
        self._objects_in_hand = obs.objects_in_hand.copy()
        
        return obs, 0.0, False

    def render_state_plt(self, *args, **kwargs) -> None:
        """Render the current state using matplotlib."""
        if self._current_state_id is None:
            return
            
        obs = self._observations[self._current_state_id]
        if obs.rgb_image is not None:
            import matplotlib.pyplot as plt
            plt.imshow(obs.rgb_image)
            plt.axis('off')
            plt.show()

    def render_state(self, *args, **kwargs) -> Video:
        """Render the current state."""
        if self._current_state_id is None:
            return []
            
        obs = self._observations[self._current_state_id]
        if obs.rgb_image is not None:
            # Convert RGB image to uint8 array
            if obs.rgb_image.dtype == np.float32:
                rgb_uint8 = (obs.rgb_image * 255).astype(np.uint8)
            else:
                rgb_uint8 = obs.rgb_image.astype(np.uint8)
            return [rgb_uint8]
        return []

    @classmethod
    def get_name(cls) -> str:
        """Return environment name."""
        return "mock_spot"
