"""Base class for mock environment creators."""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from bosdyn.client.math_helpers import SE3Pose, Quat

from predicators.envs.mock_spot_env import MockSpotEnv
from predicators.ground_truth_models.mock_spot_env.nsrts import MockSpotGroundTruthNSRTFactory
from predicators.structs import GroundAtom, EnvironmentTask, State, Task, Type, Predicate, ParameterizedOption, NSRT, Object
from predicators.planning import task_plan_grounding, task_plan
from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext
from predicators.ground_truth_models import get_gt_options
from predicators import utils


class MockEnvCreatorBase(ABC):
    """Base class for mock environment creators."""

    def __init__(self, path_dir: str) -> None:
        """Initialize the creator.
        
        Args:
            path_dir: Directory to store environment data
        """
        self.path_dir = path_dir
        self.image_dir = os.path.join(path_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Initialize environment
        self.env = MockSpotEnv(data_dir=path_dir)
        
        # Get environment info
        self.types: Dict[str, Type] = {t.name: t for t in self.env.types}
        self.predicates: Dict[str, Predicate] = {p.name: p for p in self.env.predicates}
        self.options: Dict[str, ParameterizedOption] = {o.name: o for o in get_gt_options(self.env.get_name())}
        
        # Get NSRTs from factory
        self.nsrts: Set[NSRT] = MockSpotGroundTruthNSRTFactory.get_nsrts(
            env_name="mock_spot",
            types=self.types,
            predicates=self.predicates,
            options=self.options
        )

    def add_state(self, state_id: str, rgbd: RGBDImageWithContext, 
                 objects_in_view: Optional[List[str]] = None,
                 objects_in_hand: Optional[List[str]] = None,
                 gripper_open: bool = True) -> None:
        """Add a state to the environment.
        
        Args:
            state_id: Unique identifier for the state
            rgbd: RGB-D image with context
            objects_in_view: List of object names in view
            objects_in_hand: List of object names in hand
            gripper_open: Whether the gripper is open
        """
        # Create state directory
        state_dir = os.path.join(self.image_dir, f"state_{state_id}")
        os.makedirs(state_dir, exist_ok=True)
        
        # Save RGB and depth images
        np.save(os.path.join(state_dir, "rgb.npy"), rgbd.rgb)
        np.save(os.path.join(state_dir, "depth.npy"), rgbd.depth)
        
        # TODO: Save state metadata (objects, gripper state, etc.)
        logging.info(f"Added state {state_id} to environment")

    def add_transition(self, from_state: str, to_state: str, 
                      operator: str) -> None:
        """Add a transition between states.
        
        Args:
            from_state: Source state ID
            to_state: Target state ID 
            operator: Name of operator used in transition
        """
        # TODO: Validate operator exists
        # TODO: Add transition to graph.json
        logging.info(f"Added transition {from_state} -> {to_state} with operator {operator}")

    def generate_states_and_transitions(self, task: EnvironmentTask) -> Dict[str, Set[GroundAtom]]:
        """Generate all possible states and transitions for a task using planning.
        
        Args:
            task: Environment task to generate states for
            
        Returns:
            Dictionary mapping state IDs to sets of ground atoms
        """
        # Convert EnvironmentTask to Task if needed
        if isinstance(task.init_obs, State):
            task_to_use = task.task
        else:
            raise TypeError("Expected fully observed task")
            
        # Initialize state dict with initial state
        init_atoms = utils.abstract(task_to_use.init, list(self.predicates.values()))
        states: Dict[str, Set[GroundAtom]] = {
            "0": init_atoms
        }
        
        # Get all ground NSRTs
        objects = {obj for obj in task_to_use.init.data.keys() if isinstance(obj, Object)}
        ground_nsrts = task_plan_grounding(init_atoms, nsrts=list(self.nsrts), objects=objects)
        
        # Create heuristic
        heuristic = utils.create_task_planning_heuristic(
            "hadd", init_atoms, task_to_use.goal, ground_nsrts[0],
            list(self.predicates.values()), objects)
        
        # Generate plan
        plan = task_plan(init_atoms=init_atoms,
                        ground_nsrts=ground_nsrts[0], reachable_atoms=set(),
                        goal=task_to_use.goal, heuristic=heuristic,
                        seed=0, timeout=10.0, max_skeletons_optimized=1)
        if plan is None:
            logging.warning("No plan found")
            return states
            
        # Extract states from plan
        curr_atoms = init_atoms
        for step in plan:
            # Apply operator to get next state
            next_atoms = utils.apply_operator(step[0][0].op, curr_atoms)
            
            # Add state and transition
            state_id = str(len(states))
            states[state_id] = next_atoms
            self.add_transition(str(len(states)-1), state_id, step[0][0].op.name)
            curr_atoms = next_atoms
            
        return states

    @abstractmethod
    def create_rgbd_image(self, rgb: np.ndarray, depth: np.ndarray,
                         camera_name: str = "hand_color") -> RGBDImageWithContext:
        """Create an RGBDImageWithContext from RGB and depth arrays.
        
        Args:
            rgb: RGB image array
            depth: Depth image array
            camera_name: Name of camera that captured the image
            
        Returns:
            RGBDImageWithContext instance
        """
        pass 