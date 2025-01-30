"""Base class for mock environment creators.

This module provides a base class for creating mock Spot environments with:
- Predefined states and transitions
- RGB-D observations with object detections
- Task-specific configurations
- Planning and visualization utilities
- Support for belief space planning (when enabled in environment)

The environment data is stored in a directory specified by CFG.mock_env_data_dir.
This includes:
- images/: Directory containing RGB-D images for each state
  - state_0/
    ├── state_metadata.json  # Robot state, objects, atoms
    ├── image_metadata.json  # Image paths and transforms
    ├── cam1_rgb.npy        # RGB image data
    ├── cam1_rgb.jpg        # RGB preview image
    └── cam1_depth.npy      # Optional depth data
- transitions/: Directory containing transition graph visualizations
  - {task_name}.html: Interactive visualization of state transitions

Key Features:
1. State Management:
   - Add states with RGB-D observations and metadata
   - Track objects in view/hand and gripper state
   - Support for belief predicates (e.g., ContainingWater, Known_ContainerEmpty)

2. Image Handling:
   - Flat image naming structure (e.g., "cam1.seed0.rgb")
   - Automatic conversion to UnposedImageWithContext
   - Preview JPG generation for easy visualization

3. Transition Planning:
   - Automatic state transition graph generation
   - Interactive visualization with Cytoscape.js
   - Support for belief space planning

4. Object Tracking:
   - Consistent object tracking across states
   - Automatic object registration during state creation
   - Type-safe object management

Example usage:
    ```python
    # Create environment and creator
    env = MockSpotPickPlaceTwoCupEnv()
    creator = MockEnvCreatorBase("path/to/data_dir", env_info={
        "types": env.types,
        "predicates": env.predicates,
        "options": env.options,
        "nsrts": env.nsrts
    })
    
    # Add state with images
    creator.add_state_from_raw_images({
        "cam1.seed0.rgb": ("path/to/rgb.jpg", "rgb"),
        "cam1.seed0.depth": ("path/to/depth.npy", "depth")
    }, state_id="state_0", objects_in_view={cup1, cup2, table})
    
    # Plan and visualize transitions
    creator.plan_and_visualize(env.initial_atoms, env.goal_atoms, env.objects,
                             task_name="Two Cup Pick Place")
    ```

Implementation Notes:
1. Image Processing:
   - RGB images are saved as both .npy (for processing) and .jpg (for preview)
   - Depth images are saved as .npy in float32 format
   - Camera metadata includes rotations and transforms

2. State Transitions:
   - States are uniquely identified by their atoms
   - Transitions track operator effects on fluent predicates
   - Belief predicates are highlighted in visualization

3. Object Management:
   - Objects are tracked in creator.objects dictionary
   - Objects are automatically added during state creation
   - Objects must be initialized before loading saved states

4. Visualization:
   - Interactive graph shows state contents and transitions
   - States are color-coded based on predicates
   - Transitions show operator effects and belief changes
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Optional, Any, Tuple, FrozenSet, Iterator, Union, Literal, cast
import graphviz
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.table import Table
from rich.box import HEAVY
import numpy as np
from gym.spaces import Box
import yaml
import matplotlib.pyplot as plt
from graphviz import Digraph
from rich.progress import Progress
from collections import deque
from itertools import zip_longest
import json
from pathlib import Path
from jinja2 import Template
from dataclasses import asdict
import pillow_heif
from PIL import Image as PILImage


from predicators.structs import (
    GroundAtom, EnvironmentTask, GroundTruthPredicate, State, Task, Type, Predicate, 
    ParameterizedOption, NSRT, Object, Variable, LiftedAtom, STRIPSOperator,
    Action, DefaultState, _GroundNSRT
)
from predicators.ground_truth_models import get_gt_options
from predicators import utils
from predicators.settings import CFG
from predicators.planning import task_plan_grounding, task_plan, run_task_plan_once
from predicators.spot_utils.perception.perception_structs import UnposedImageWithContext
from predicators.spot_utils.mock_env.mock_env_utils import _SavedMockSpotObservation


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Key predicates that affect observations and world state
KEY_PREDICATES = {
    # VLM predicates for world state
    "Inside",  # Object containment
    "On",      # Object placement
    # "Empty",   # Cup emptiness
    "DrawerOpen",  # Drawer state
    "DrawerClosed",  # Drawer state
    
    # View-related predicates
    # "InHandView",  # Object visible in hand camera
    # "InNavView",   # Object visible in navigation camera
    # "HandEmpty",   # Gripper state
    # "Holding",     # Object being held
}

class MockEnvCreatorBase(ABC):
    """Base class for mock environment creators.
    
    This class provides functionality to:
    - Create and configure mock environments
    - Add states with RGB-D observations
    - Define transitions between states
    - Generate task-specific state sequences
    - Plan and visualize transitions
    - Support belief space planning when enabled in environment
    
    The environment data is stored in a directory specified by CFG.mock_env_data_dir,
    which is set during initialization. This includes:
    - Plan data (plan.yaml)
    - RGB-D images for each state (images/)
    - Observation metadata (gripper state, objects in view/hand)
    - Transition graph visualization (transitions/)
    
    Attributes:
        path_dir (str): Base directory for environment data
        image_dir (str): Directory for RGB-D images
        transitions_dir (str): Directory for transition graph visualizations
        env ("MockSpotEnv"): Mock environment instance
        types (Dict[str, Type]): Available object types
        predicates (Dict[str, Predicate]): Available predicates (including belief predicates if enabled)
        options (Dict[str, ParameterizedOption]): Available options
        nsrts (Set[NSRT]): Available NSRTs (including belief space operators when enabled)
        console (Console): Rich console for pretty printing
    """

    def __init__(self, output_dir: str, env_info: Optional[Dict[str, Any]] = None, env: Optional[Any] = None) -> None:
        """Initialize the mock environment creator.
        
        Args:
            output_dir: Directory to save output files
            env_info: Environment information
            env: Optional environment instance
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.path_dir = self.output_dir
        self.image_dir = self.output_dir / "state_info"
        self.transitions_dir = self.output_dir / "transitions"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.transitions_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize state storage
        self.states: Dict[str, _SavedMockSpotObservation] = {}  # For observation data
        self.state_to_id: Dict[FrozenSet[GroundAtom], str] = {}  # Maps state atoms to IDs
        self.id_to_state: Dict[str, Set[GroundAtom]] = {}  # Maps IDs to state atoms
        self.transitions: List[Tuple[str, _GroundNSRT, str]] = []  # (source_id, op, dest_id)
        self.str_transitions: List[Tuple[str, Dict[str, Any], str]] = []  # For saving to JSON
        
        # Initialize unique state tracking
        self._unique_world_states: Dict[str, Set[str]] = {}  # world state hash -> set of state IDs
        self._state_to_canonical: Dict[str, str] = {}  # state ID -> canonical state ID
        self._canonical_state_to_id: Dict[str, Set[str]] = {}  # canonical state ID -> set of state IDs
        
        # Add ground atom to state mappings
        self._atom_to_states: Dict[str, Set[str]] = {}  # ground atom str -> set of state IDs
        self._atom_to_canonical_states: Dict[str, Set[str]] = {}  # ground atom str -> set of canonical state IDs
        
        # Store key predicates
        self._key_predicates = KEY_PREDICATES

        if env is not None:
            self.types = {t.name: t for t in env.types}
            self.predicates = {p.name: p for p in env.predicates}
            self.options = {o.name: o for o in env.options}
            self.nsrts = env.nsrts
            self.objects = {o.name: o for o in env.objects}
            # Store initial and goal atoms from environment
            self.env_initial_atoms = env.initial_atoms
            self.env_goal_atoms = env.goal_atoms
        elif env_info is not None:
            self.types = {t.name: t for t in env_info["types"]}
            self.predicates = {p.name: p for p in env_info["predicates"]}
            self.options = {o.name: o for o in env_info["options"]}
            self.nsrts = env_info["nsrts"]
            self.objects: Dict[str, Object] = {o.name: o for o in env_info["objects"]}
            # Store initial and goal atoms from env_info if available
            self.env_initial_atoms = env_info.get("initial_atoms", None)
            self.env_goal_atoms = env_info.get("goal_atoms", None)
        else:
            raise ValueError("Either env or env_info must be provided")

        if "robot" not in self.objects:
            self.objects["robot"] = Object(name="robot", type=self.types["robot"])
            
        # Store robot object for easy access
        self.robot_object = self.objects["robot"]

        # Calculate fluent predicates by looking at operator effects
        self.fluent_predicates: Set[str] = self._calculate_fluent_predicates()
        
        # Initialize rich console for pretty printing
        self.console = Console()

        self.current_state_id = "0"  # Default to initial state

    def _format_atoms(self, atoms: set) -> str:
        """Format atoms for display, showing only fluent predicates."""
        formatted_atoms = []
        for atom in sorted(atoms, key=str):
            # Only show fluent predicates
            if atom.predicate.name in self.fluent_predicates:
                args = [obj.name for obj in atom.objects]
                formatted_atoms.append(f"{atom.predicate.name}({', '.join(args)})")
        return "\n".join(formatted_atoms)

    def _calculate_fluent_predicates(self) -> Set[str]:
        """Calculate fluent predicates by looking at operator effects.
        Similar to Fast Downward's get_fluents function."""
        fluent_predicates = set()
        # Look at all operators (NSRTs)
        for nsrt in self.nsrts:
            # Add predicates that appear in add or delete effects
            for effect in nsrt.add_effects:
                fluent_predicates.add(effect.predicate.name)
            for effect in nsrt.delete_effects:
                fluent_predicates.add(effect.predicate.name)
        return fluent_predicates

    def _get_world_state_hash(self, atoms: Set[GroundAtom]) -> str:
        """Get a hash representing the world state based on key predicates."""
        # Only consider key predicates for world state
        key_atoms = self.get_key_predicate_values(atoms)
        return f"world:{','.join(sorted(str(atom) for atom in key_atoms))}"

    def get_key_predicate_values(self, state_atoms: Set[GroundAtom]) -> Set[GroundAtom]:
        """Get values of key predicates from a state.
        
        Args:
            state_atoms: Set of ground atoms representing the state
            
        Returns:
            Set of ground atoms for key predicates only
        """
        return {atom for atom in state_atoms if atom.predicate.name in self._key_predicates}

    def _save_state(self, 
                 state_id: str,
                 images: Dict[str, UnposedImageWithContext],
                 objects_in_view: Set[Object],
                 objects_in_hand: Set[Object],
                 gripper_open: bool = True,
                 atom_dict: Optional[Dict[str, bool]] = None,
                 non_vlm_atom_dict: Optional[Dict[GroundAtom, bool]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 cover_equivalent_states: bool = True) -> None:
        """Add a state to the environment.
        
        Args:
            state_id: ID for the state
            images: Dict of images for this state
            objects_in_view: Objects visible in the images
            objects_in_hand: Objects being held
            gripper_open: Whether gripper is open
            atom_dict: Optional dict of VLM atom values
            non_vlm_atom_dict: Optional dict of non-VLM atom values
            metadata: Optional metadata dict
            cover_equivalent_states: If True, save data for all equivalent states
        """
        # Get all states to save data for
        states_to_save = set()
        if cover_equivalent_states:
            # Get canonical state for this state
            canonical_id = self._state_to_canonical.get(state_id)
            if canonical_id is not None:
                # Add all states that map to this canonical state
                states_to_save.update(self._canonical_state_to_id[canonical_id])
        else:
            # Override state if really want to save this state
            states_to_save.add(state_id)
            
        # Save observation data for each state
        for curr_state_id in states_to_save:
            obs = _SavedMockSpotObservation(
                state_id=curr_state_id,  # Use this state's ID
                images=images,  # Share the same images
                gripper_open=gripper_open,
                non_vlm_atom_dict=non_vlm_atom_dict or {},
                objects_in_view=objects_in_view,  # Not actively used
                objects_in_hand=objects_in_hand,  # Not actively used
                atom_dict=atom_dict or {},  # Not used anymore
                metadata=metadata or {}
            )
            
            # Save state
            obs.save_state(self.image_dir)
            
            # Track objects and state
            for obj in objects_in_view | objects_in_hand:
                self.objects[obj.name] = obj
            self.states[curr_state_id] = obs
            
    def save_observation(self, 
                            state_id: str,
                            images: Dict[str, UnposedImageWithContext],
                            objects_in_view: Set[Object],
                            objects_in_hand: Set[Object],
                            gripper_open: bool = True) -> None:
        """Add observational data for a known state.
        
        Args:
            state_id: ID of an existing state in the transition graph
            images: RGB-D images for this state
            objects_in_view: Objects visible in the images
            objects_in_hand: Objects being held
            gripper_open: Whether gripper is open
            
        Note:
            We use a closed world assumption for non-VLM predicates. This means:
            - All possible ground atoms for GroundTruthPredicates are explicitly stored
            - If a ground atom is in state_atoms, its value is True
            - If a ground atom is not in state_atoms, its value is False
            - There are no "unknown" values for non-VLM predicates
        """
        if state_id not in self.id_to_state:
            raise ValueError(f"State {state_id} not found in transition graph")
            
        # Create observation with ground atoms from known state
        state_atoms = self.id_to_state[state_id]
        
        # Get all possible ground atoms for GroundTruthPredicates
        ground_truth_preds = {p for p in self.predicates.values() 
                            if isinstance(p, GroundTruthPredicate)}
        # NOTE: We add the robot object to the list of objects because it may not be in the objects dict
        objects = list(self.objects.values())
        if self.robot_object not in objects:
            objects.append(self.robot_object)
            
        all_ground_atoms = utils.get_all_ground_atom_combinations_for_predicate_set(
            objects, cast(Set[Predicate], ground_truth_preds)
        )
        
        # Create dictionary with True/False values based on closed world assumption
        non_vlm_atoms = {atom: (atom in state_atoms) for atom in all_ground_atoms}
        
        self._save_state(state_id=state_id, images=images, objects_in_view=objects_in_view, 
                       objects_in_hand=objects_in_hand, gripper_open=gripper_open, 
                       atom_dict={}, non_vlm_atom_dict=non_vlm_atoms, metadata={})

    def add_state_from_raw_images(
        self,
        raw_images: Dict[str, Tuple[str, str]],
        state_id: Optional[str] = None,
        objects_in_view: Optional[Set[Object]] = None,
        objects_in_hand: Optional[Set[Object]] = None,
        gripper_open: bool = True
    ) -> None:
        """Add a state with multiple views to the environment.
        
        Args:
            raw_images: Dict mapping image names to (path, type) tuples:
                {
                    "cam1.seed0.rgb": ("path/to/rgb.jpg", "rgb"),
                    "cam1.seed0.depth": ("path/to/depth.npy", "depth")
                }
            state_id: ID of an existing state in the transition graph
            objects_in_view: Objects visible in images
            objects_in_hand: Objects being held
            gripper_open: Whether gripper is open
        """
        # Convert views to UnposedImageWithContext
        images: Dict[str, UnposedImageWithContext] = {}
        
        # Group images by camera
        camera_images: Dict[str, Dict[str, Tuple[str, str]]] = {}
        for img_name, (path, img_type) in raw_images.items():
            camera_name = img_name.split(".")[0]  # Extract cam1 from cam1.seed0.rgb
            if camera_name not in camera_images:
                camera_images[camera_name] = {}
            camera_images[camera_name][img_name] = (path, img_type)
        
        # Process each camera's images
        for camera_name, camera_data in camera_images.items():
            rgb_data = None
            depth_data = None
            rgb_img_name = None
            
            # Process RGB and depth images
            for img_name, (path, img_type) in camera_data.items():
                if img_type == "rgb":
                    rgb_data = self.process_rgb_image(path)
                    rgb_img_name = img_name
                else:  # depth
                    depth_data = self.process_depth_image(path)
            
            # Skip if no RGB data
            if rgb_data is None or rgb_img_name is None:
                continue
            
            # Create UnposedImageWithContext
            image_key = camera_name
            images[image_key] = UnposedImageWithContext(
                rgb=rgb_data,
                depth=depth_data,
                camera_name=camera_name,
                image_rot=None
            )

        # Verify state_id exists in transition graph
        if state_id is None:
            raise ValueError("state_id must be provided and must exist in transition graph")
        if state_id not in self.id_to_state:
            raise ValueError(f"State {state_id} not found in transition graph")

        # Add observation data for the state
        self.save_observation(
            state_id=state_id,
            images=images,
            objects_in_view=objects_in_view or set(),
            objects_in_hand=objects_in_hand or set(),
            gripper_open=gripper_open
        )

    def load_state(self, state_id: str) -> _SavedMockSpotObservation:
        """Load a state from disk.
        
        Args:
            state_id: ID of the state to load
            
        Returns:
            The state observation data
            
        Raises:
            ValueError: If state cannot be loaded
        """
        # First check if state is already loaded
        if state_id in self.states:
            return self.states[state_id]
            
        # Load state using _SavedMockSpotObservation's load_state
        try:
            obs = _SavedMockSpotObservation.load_state(state_id, self.image_dir, self.objects)
            self.states[state_id] = obs
            return obs
        except Exception as e:
            raise ValueError(f"Error loading state {state_id}: {str(e)}")

    def save_plan(self, plan: List[Any], states: List[Set[GroundAtom]], 
                 init_atoms: Set[GroundAtom], goal_atoms: Set[GroundAtom],
                 objects: Set[Object]) -> None:
        """Save the plan and state sequence.
        
        Args:
            plan: List of operators
            states: List of states (sets of ground atoms)
            init_atoms: Initial state atoms
            goal_atoms: Goal state atoms
            objects: Objects in the environment
        """
        plan_data = {
            "objects": {
                obj.name: {
                    "type": obj.type.name
                }
                for obj in objects
            },
            "initial_state": [str(atom) for atom in init_atoms],
            "goal_state": [str(atom) for atom in goal_atoms],
            "states": [
                {
                    "id": str(i),
                    "atoms": [str(atom) for atom in state_atoms],
                    "operator": operator.name if operator else None,
                    "operator_objects": [obj.name for obj in operator.objects] if operator else None
                }
                for i, (state_atoms, operator) in enumerate(zip(states, plan + [None]))
            ]
        }
        
        # Save plan data
        plan_path = self.path_dir / "plan.yaml"
        with open(plan_path, "w") as f:
            yaml.safe_dump(plan_data, f, default_flow_style=False, sort_keys=False)

    def load_plan(self) -> Optional[Tuple[List[Any], List[Set[GroundAtom]], 
                                        Set[GroundAtom], Set[GroundAtom], Set[Object]]]:
        """Load the saved plan and state sequence.
        
        Returns:
            If plan exists, tuple containing:
            - List of operators
            - List of states (sets of ground atoms)
            - Initial state atoms
            - Goal state atoms
            - Objects in the environment
            
            None if no plan is saved
        """
        plan_path = self.path_dir / "plan.yaml"
        if not plan_path.exists():
            return None
            
        with open(plan_path) as f:
            plan_data = yaml.safe_load(f)
            
        # Recreate objects
        objects = {
            Object(
                name=name,
                type=self.types[data["type"]]
            )
            for name, data in plan_data["objects"].items()
        }
        
        # Parse atoms using predicates
        def parse_atom(atom_str: str) -> GroundAtom:
            # Example format: "Predicate(obj1, obj2)"
            pred_name = atom_str.split("(")[0]
            obj_names = [n.strip(" )") for n in atom_str.split("(")[1].split(",")]
            return GroundAtom(
                self.predicates[pred_name],
                [next(obj for obj in objects if obj.name == name) for name in obj_names]
            )
        
        init_atoms = {parse_atom(s) for s in plan_data["initial_state"]}
        goal_atoms = {parse_atom(s) for s in plan_data["goal_state"]}
        
        # Recreate states and operators
        states = []
        plan = []
        for state_data in plan_data["states"][:-1]:  # Last state has no operator
            states.append({parse_atom(s) for s in state_data["atoms"]})
            if state_data["operator"]:
                operator_objects = [
                    next(obj for obj in objects if obj.name == name)
                    for name in state_data["operator_objects"]
                ]
                # Create empty params array for operator
                params = np.zeros(self.options[state_data["operator"]].params_space.shape, dtype=np.float32)
                plan.append(self.options[state_data["operator"]].ground(operator_objects, params))
                
        # Add final state
        states.append({parse_atom(s) for s in plan_data["states"][-1]["atoms"]})
        
        return plan, states, init_atoms, goal_atoms, objects

    def plan(self, init_atoms: Set[GroundAtom], goal_atoms: Set[GroundAtom],
             objects: Set[Object], timeout: float = 10.0) -> Optional[Tuple[List[Any], List[Set[GroundAtom]], Dict[str, Any]]]:
        """Plan a sequence of actions to achieve the goal.
        
        Args:
            init_atoms: Initial ground atoms
            goal_atoms: Goal ground atoms
            objects: Objects in the environment
            timeout: Planning timeout in seconds
            
        Returns:
            If plan found, tuple containing:
            - List of operators (the plan)
            - List of states (sets of ground atoms)
            - Metrics dictionary
            None if no plan found
        """
        # Ground NSRTs and get reachable atoms
        ground_nsrts, reachable_atoms = task_plan_grounding(
            init_atoms=init_atoms,
            objects=objects,
            nsrts=self.nsrts,
            allow_noops=False
        )
        
        # Create heuristic for planning
        heuristic = utils.create_task_planning_heuristic(
            CFG.sesame_task_planning_heuristic,
            init_atoms,
            goal_atoms,
            ground_nsrts,
            self.predicates.values(),
            objects
        )
        
        # Create and run planner
        try:
            plan_gen = task_plan(
                init_atoms=init_atoms,
                goal=goal_atoms,
                ground_nsrts=ground_nsrts,
                reachable_atoms=reachable_atoms,
                heuristic=heuristic,
                seed=CFG.seed,
                timeout=timeout,
                max_skeletons_optimized=CFG.sesame_max_skeletons_optimized,
                use_visited_state_set=True
            )
            plan, atoms_sequence, metrics = next(plan_gen)
            return plan, atoms_sequence, metrics
        except StopIteration:
            return None

    def plan_and_visualize(self, initial_atoms: Set[GroundAtom], goal_atoms: Set[GroundAtom], objects: Set[Object], task_name: str, use_graphviz: bool = False) -> None:
        """Plan and visualize transitions.
        
        Args:
            initial_atoms: Initial state atoms
            goal_atoms: Goal state atoms
            objects: Objects in the environment
            task_name: Name of the task for visualization
            use_graphviz: Whether to use graphviz instead of cytoscape.js
        """
        # Track all objects first
        for obj in objects:
            self.objects[obj.name] = obj
        
        # Explore states and transitions
        self.explore_states(initial_atoms, objects)
        
        # Get transitions and edges
        transitions = self.get_operator_transitions(initial_atoms, objects)
        edges = self.get_graph_edges(initial_atoms, goal_atoms, objects)
        
        # Track shortest path edges and states
        shortest_path_edges = set()
        shortest_path_states = {frozenset(initial_atoms)}
        
        # Get plan to goal
        planner = self._create_planner(initial_atoms, goal_atoms, objects)
        try:
            # TODO: visualize all shortest-path skeletons
            skeleton, atoms_sequence, metrics = next(planner())
            # Follow plan to get shortest path
            curr_atoms = initial_atoms.copy()
            for op in skeleton:
                next_atoms = self._get_next_atoms(curr_atoms, op)
                source_id = self.state_to_id[frozenset(curr_atoms)]
                dest_id = self.state_to_id[frozenset(next_atoms)]
                shortest_path_edges.add((source_id, dest_id))
                shortest_path_states.add(frozenset(next_atoms))
                curr_atoms = next_atoms
        except StopIteration:
            pass
        
        # Create edge data
        edge_data = []
        edge_count = 0
        for source_atoms, operator, dest_atoms in transitions:
            source_state = frozenset(source_atoms)
            dest_state = frozenset(dest_atoms)
            source_id = self.state_to_id[source_state]
            dest_id = self.state_to_id[dest_state]
            if source_id != dest_id:  # Skip self-loops
                # Create detailed edge label
                op_str = f"{operator.name}({','.join(obj.name for obj in operator.objects)})"
                edge_data.append({
                    'id': f'edge_{edge_count}',
                    'source': source_id,
                    'target': dest_id,
                    'label': op_str,
                    'fullLabel': self._get_edge_label(operator),
                    'is_shortest_path': (source_id, dest_id) in shortest_path_edges,
                    'affects_belief': any(effect.predicate.name.startswith(('Believe', 'Known_', 'Unknown_')) 
                                       for effect in (operator.add_effects | operator.delete_effects))
                })
                edge_count += 1
        
        # Create visualization
        if use_graphviz:
            # Create graphviz visualization
            trans_ops = {(self.state_to_id[frozenset(t[0])], t[1].name,
                        tuple(obj.name for obj in t[1].objects),
                        self.state_to_id[frozenset(t[2])]) for t in transitions}
            # Use task name for the output file
            output_path = self.output_dir / "transitions" / task_name
            create_graphviz_visualization(trans_ops, task_name, output_path)
        else:
            # Create interactive visualization
            # Create graph data
            graph_data = {
                'nodes': {},
                'edges': edge_data,
                'metadata': {
                    'task_name': task_name,
                    'fluent_predicates': []
                }
            }
            
            # Add node data
            for atoms, state_id in self.state_to_id.items():
                is_initial = atoms == frozenset(initial_atoms)
                is_goal = goal_atoms.issubset(atoms)
                is_shortest_path = atoms in shortest_path_states
                
                # Get self loops
                self_loops = []
                for source_atoms, op, dest_atoms in transitions:
                    if (frozenset(source_atoms) == atoms and 
                        frozenset(dest_atoms) == atoms):
                        self_loops.append(f"{op.name}({','.join(obj.name for obj in op.objects)})")
                
                # Create label
                state_label = f"{'Initial ' if is_initial else ''}{'Goal ' if is_goal else ''}State {state_id}"
                full_label_parts = [
                    state_label,
                    "",
                    self._format_atoms(set(atoms))  # Convert frozenset to regular set
                ]
                if self_loops:
                    full_label_parts.extend([
                        "",
                        "Self-loop operators:",
                        *[f"  {op}" for op in self_loops]
                    ])
                
                # Add node
                graph_data['nodes'][state_id] = {
                    'id': state_id,
                    'state_num': state_id,
                    'atoms': [str(atom) for atom in atoms],
                    'is_initial': is_initial,
                    'is_goal': is_goal,
                    'is_shortest_path': is_shortest_path,
                    'label': state_label,
                    'fullLabel': '\n'.join(full_label_parts)
                }
            
            # Create visualization
            # Use task name for the output file
            output_path = self.output_dir / "transitions" / f"{task_name}.html"
            create_interactive_visualization(graph_data, output_path)

    def _build_transition_graph(self, task_name: str, init_atoms: Set[GroundAtom], goal: Set[GroundAtom], objects: List[Object], plan: Optional[List[_GroundNSRT]] = None) -> Dict[str, Any]:
        """Build transition graph data structure that can be used by different visualizers.
        
        Args:
            task_name: Name of the task
            init_atoms: Initial ground atoms
            goal: Goal ground atoms
            objects: List of objects
            plan: Optional plan to highlight shortest path
            
        Returns:
            Dict containing:
            - nodes: Dict mapping node IDs to node data (atoms, type, etc.)
            - edges: List of edge data (source, target, operator, etc.)
            - metadata: Dict of graph metadata (task name, etc.)
        """
        # Track shortest path states and fluent predicates
        shortest_path_states = set()
        fluent_predicates = set()
        state_numbers = {}  # Map state atoms to numbers
        state_count = 0
        
        if plan is not None:
            curr_atoms = init_atoms
            shortest_path_states.add(frozenset(curr_atoms))
            prev_atoms = curr_atoms
            state_numbers[frozenset(curr_atoms)] = state_count
            state_count += 1
            
            for op in plan:
                # Verify operator preconditions
                preconditions_satisfied = all(precond in curr_atoms for precond in op.preconditions)
                if not preconditions_satisfied:
                    logger.warning(f"Invalid transition in plan! Operator {op} preconditions not met in state {state_count-1}")
                
                curr_atoms = self._get_next_atoms(curr_atoms, op)
                # Track predicates that change
                added = curr_atoms - prev_atoms
                removed = prev_atoms - curr_atoms
                fluent_predicates.update(atom.predicate for atom in (added | removed))
                shortest_path_states.add(frozenset(curr_atoms))
                state_numbers[frozenset(curr_atoms)] = state_count
                state_count += 1
                prev_atoms = curr_atoms
        
        # Track visited states and transitions
        visited = set()
        transitions = set()  # Use set to avoid duplicates
        state_self_loops = {}  # Track self-loops for each state
        nodes = {}  # Store node data
        edges = []  # Store edge data
        
        # Initialize frontier with initial state
        frontier: List[Tuple[Set[GroundAtom], Optional[_GroundNSRT]]] = [(init_atoms, None)]
        
        # Track state numbers for consistent labeling
        state_count = 0
        state_numbers: Dict[FrozenSet[GroundAtom], int] = {}
        
        # Explore states following operator flow
        while frontier:
            current_atoms, prev_operator = frontier.pop(0)
            current_id = self._get_state_id(current_atoms)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Get state number
            state_num = state_numbers.get(frozenset(current_atoms))
            if state_num is None:
                state_num = state_count
                state_numbers[frozenset(current_atoms)] = state_num
                state_count += 1
            
            # Get applicable operators that follow from current state
            applicable_ops = self._get_applicable_operators(current_atoms, set(objects))
            
            # Track self-loops for this state
            self_loops = []
            
            # Add transitions to next states
            for op in applicable_ops:
                # Calculate next state
                next_atoms = self._get_next_atoms(current_atoms, op)
                next_id = self._get_state_id(next_atoms)
                
                # Format operator name
                op_label = op.name
                if op.objects:
                    op_label += f"({','.join(obj.name for obj in op.objects)})"
                
                # Check if this is a self-loop
                if current_id == next_id:
                    self_loops.append(op_label)
                    continue
                
                # Only add transition if it follows operator flow
                transition = (current_id, next_id, op.name, tuple(obj.name for obj in op.objects))
                if transition not in transitions:
                    transitions.add(transition)
                    
                    # Add edge data
                    edge_data = {
                        'source': current_id,
                        'target': next_id,
                        'operator': op_label,
                        'is_shortest_path': frozenset(current_atoms) in shortest_path_states
                    }
                    edges.append(edge_data)
                    
                    # Add next state to frontier if not visited
                    if next_id not in visited:
                        frontier.append((next_atoms, op))
            
            # Store self-loops for this state
            if self_loops:
                state_self_loops[current_id] = self_loops
            
            # Add node data
            is_initial = (current_atoms == init_atoms)
            is_goal = goal.issubset(current_atoms)
            is_shortest_path = frozenset(current_atoms) in shortest_path_states
            
            nodes[current_id] = {
                'id': current_id,
                'state_num': state_num,
                'atoms': current_atoms,
                'is_initial': is_initial,
                'is_goal': is_goal,
                'is_shortest_path': is_shortest_path,
                'self_loops': state_self_loops.get(current_id, []),
                'label': self._get_state_label(state_num, current_atoms, fluent_predicates, 
                                             is_initial, is_goal, state_self_loops.get(current_id))
            }
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'task_name': task_name,
                'fluent_predicates': fluent_predicates
            }
        }

    def visualize_transitions(self, task_name: str, init_atoms: Set[GroundAtom], goal: Set[GroundAtom], objects: List[Object], plan: Optional[List[_GroundNSRT]] = None) -> None:
        """Visualize transitions for a task using graphviz."""
        # Get graph data
        graph_data = self._build_transition_graph(task_name, init_atoms, goal, objects, plan)
        
        # Create graph with improved settings
        dot = graphviz.Digraph(comment=f'Transition Graph for {task_name}')
        dot.attr(rankdir='TB')  # Top to bottom layout
        
        # Set global graph attributes        
        dot.attr('graph', {
            'fontname': 'Arial',
            'fontsize': '16',
            'label': f'Transition Graph for {task_name}',
            'labelloc': 't',
            'nodesep': '1.0',
            'ranksep': '1.2',
            'splines': 'curved',
            'concentrate': 'false'
        })
        
        # Set node attributes
        dot.attr('node', {
            'fontname': 'Arial',
            'fontsize': '10',
            'shape': 'box',
            'style': 'rounded,filled',
            'margin': '0.3',
            'width': '2.5',
            'height': '1.5'
        })
        
        # Add nodes
        for node_id, node_data in graph_data['nodes'].items():
            node_attrs = {
                'label': node_data['label'],
                'style': 'rounded,filled',
                'fillcolor': self._get_state_color(node_data['atoms'])
            }
            
            if node_data['is_initial']:
                node_attrs['penwidth'] = '2.0'
                node_attrs['color'] = '#2171b5'  # Dark blue border
            elif node_data['is_goal']:
                node_attrs['penwidth'] = '2.0'
                node_attrs['color'] = '#2ca02c'  # Dark green border
            
            dot.node(node_id, **node_attrs)
        
        # Add edges
        for edge in graph_data['edges']:
            edge_attrs = self._get_edge_style(edge['operator'], edge['is_shortest_path'])
            dot.edge(edge['source'], edge['target'], edge['operator'], **edge_attrs)
        
        # Save graph
        graph_path = self.transitions_dir / f"{task_name}"
        dot.render(graph_path, format='png', cleanup=True)
        
    def process_rgb_image(self, image_path: str) -> np.ndarray:
        """Process an RGB image file.
        
        Args:
            image_path: Path to RGB image file (supports .jpg, .png, .heic)
            
        Returns:
            RGB array (H, W, 3) in uint8 format
            
        Raises:
            RuntimeError: If image loading fails
        """
        try:
            # Check if file is HEIC
            if image_path.lower().endswith('.heic'):
                # Read HEIC file and convert to PIL Image
                heif_file = pillow_heif.read_heif(image_path)
                image = heif_file.to_pillow()
                # Convert to numpy array
                rgb = np.array(image)
            else:
                # Use matplotlib for other formats
                rgb = plt.imread(image_path)
                
            if rgb.dtype == np.float32:
                rgb = (rgb * 255).astype(np.uint8)
            return rgb
        except Exception as e:
            raise RuntimeError(f"Failed to load RGB image from {image_path}: {e}")

    def process_depth_image(self, image_path: str) -> np.ndarray:
        """Process a depth image file.
        
        Args:
            image_path: Path to depth image file
            
        Returns:
            Depth array (H, W) in float32 format
            
        Raises:
            RuntimeError: If image loading fails
        """
        try:
            depth = np.load(image_path)  # Assuming depth is saved as .npy
            if depth.dtype != np.float32:
                depth = depth.astype(np.float32)
            return depth
        except Exception as e:
            raise RuntimeError(f"Failed to load depth image from {image_path}: {e}")

    def _check_operator_preconditions(self, state_atoms: Set[GroundAtom], operator: _GroundNSRT) -> bool:
        """Check if operator preconditions are satisfied in the current state."""
        # Get operator preconditions
        preconditions = operator.preconditions
        
        # Check if all preconditions are satisfied
        return all(precond in state_atoms for precond in preconditions)

    def _get_state_color(self, atoms: Set[GroundAtom]) -> str:
        """Get color for state visualization based on its atoms."""
        # Colors for belief predicates
        belief_colors = {
            "BelieveTrue": "#90EE90",  # Light green
            "BelieveFalse": "#FFB6C1", # Light pink
            "Known": "#ADD8E6",        # Light blue
            "Unknown": "#F0E68C",      # Khaki
        }
        
        # Check for belief predicates first
        for atom in atoms:
            if atom.predicate.name in self.fluent_predicates:
                pred_name = atom.predicate.name
                for belief_type, color in belief_colors.items():
                    if pred_name.startswith(belief_type):
                        return color
        
        # Default colors for other fluent predicates
        colors = {
            "HandEmpty": "#FFFFFF",     # White
            "Holding": "#E6E6FA",       # Lavender
            "Inside": "#DDA0DD",        # Plum
            "DrawerOpen": "#98FB98",    # Pale green
            "DrawerClosed": "#FFA07A",  # Light salmon
        }
        
        # Only color states based on fluent predicates
        for atom in atoms:
            if atom.predicate.name in self.fluent_predicates and atom.predicate.name in colors:
                return colors[atom.predicate.name]
        
        return "#FFFFFF"  # White default

    def _create_planner(self, init_atoms: Set[GroundAtom], goal_atoms: Set[GroundAtom],
                       objects: Set[Object]) -> Any:
        """Create a planner for task planning.
        
        Args:
            init_atoms: Initial ground atoms
            goal_atoms: Goal ground atoms
            objects: Objects in the environment
            
        Returns:
            A planner instance that supports plan() method
        """
        # Ground NSRTs and get reachable atoms
        ground_nsrts, reachable_atoms = task_plan_grounding(
            init_atoms=init_atoms,
            objects=objects,
            nsrts=self.nsrts,
            allow_noops=False
        )
        
        # Create heuristic for planning
        heuristic = utils.create_task_planning_heuristic(
            CFG.sesame_task_planning_heuristic,
            init_atoms,
            goal_atoms,
            ground_nsrts,
            self.predicates.values(),
            objects
        )
        
        # Return planner generator
        return lambda: task_plan(
            init_atoms=init_atoms,
            goal=goal_atoms,
            ground_nsrts=ground_nsrts,
            reachable_atoms=reachable_atoms,
            heuristic=heuristic,
            seed=CFG.seed,
            timeout=10.0,
            max_skeletons_optimized=CFG.sesame_max_skeletons_optimized,
            use_visited_state_set=True
        )
        
    def _visualize_state(self, state: Set[GroundAtom]) -> None:
        """Visualize a state's atoms, focusing on fluents."""
        # Only show fluent predicates
        important_atoms = {atom for atom in state 
                         if atom.predicate.name in self.fluent_predicates}
        
        # Group by object
        atoms_by_obj = {}
        for atom in sorted(important_atoms, key=str):
            obj_name = atom.objects[0].name
            if obj_name not in atoms_by_obj:
                atoms_by_obj[obj_name] = []
            atoms_by_obj[obj_name].append(str(atom))
        
        # Create formatted string
        state_str = ""
        for obj_name, obj_atoms in sorted(atoms_by_obj.items()):
            if obj_atoms:
                state_str += f"\n{obj_name}:\n  "
                state_str += "\n  ".join(obj_atoms)
        
        # Create state panel
        state_panel = Panel(
            state_str.strip(),
            title="Current State (Fluents)",
            border_style="cyan"
        )
        self.console.print(state_panel)

    def create_rgbd_image(self, rgb: np.ndarray, depth: np.ndarray,
                         camera_name: str = "hand_color") -> None:
        """Create an RGBDImageWithContext from RGB and depth arrays."""
        pass

    def _get_state_id(self, atoms: Set[GroundAtom]) -> str:
        """Get a unique ID for a state based on its atoms.
        
        Args:
            atoms: The atoms in the state.
            
        Returns:
            A unique string ID.
        """
        return str(hash(frozenset(str(a) for a in atoms)))

    def explore_states(self, init_atoms: Set[GroundAtom], objects: Set[Object]) -> None:
        """Internally explore and store all reachable states and transitions.
        Also identifies canonical states based on key predicates."""
        # Start with initial state
        init_state_frozen = frozenset(init_atoms)
        init_id = "0"  # Use integer ID
        self.state_to_id[init_state_frozen] = init_id
        self.id_to_state[init_id] = init_atoms
        
        # Explore states via BFS
        frontier: List[Tuple[Set[GroundAtom], Optional[str]]] = [(init_atoms, None)]
        visited = {init_state_frozen}
        state_count = 1
        
        while frontier:
            curr_atoms, _ = frontier.pop(0)
            curr_id = self.state_to_id[frozenset(curr_atoms)]
            
            # Get current state's key predicate values
            curr_key_atoms = self.get_key_predicate_values(curr_atoms)
            curr_hash = self._get_world_state_hash(curr_key_atoms)
            
            # Map to canonical state if needed
            if curr_hash in self._unique_world_states:
                canonical_id = next(iter(self._unique_world_states[curr_hash]))
                self._state_to_canonical[curr_id] = canonical_id
                self._canonical_state_to_id[canonical_id].add(curr_id)
            else:
                # This is a new canonical state
                self._unique_world_states[curr_hash] = {curr_id}
                self._state_to_canonical[curr_id] = curr_id
                self._canonical_state_to_id[curr_id] = {curr_id}
            
            # Get applicable operators
            applicable_ops = self._get_applicable_operators(curr_atoms, objects)
            
            for op in applicable_ops:
                next_atoms = self._get_next_atoms(curr_atoms, op)
                next_frozen = frozenset(next_atoms)
                
                # Add new state if not seen
                if next_frozen not in visited:
                    visited.add(next_frozen)
                    next_id = str(state_count)
                    state_count += 1
                    self.state_to_id[next_frozen] = next_id
                    self.id_to_state[next_id] = next_atoms
                    frontier.append((next_atoms, next_id))
                
                # Add transition
                next_id = self.state_to_id[next_frozen]
                self.add_transition(curr_id, op, next_id)

    def _get_applicable_operators(self, atoms: Set[GroundAtom], objects: Set[Object]) -> List[_GroundNSRT]:
        """Get applicable operators for the given atoms and objects."""
        ground_ops: List[_GroundNSRT] = []
        for nsrt in self.nsrts:
            ground_ops.extend(utils.all_ground_nsrts(nsrt, objects))
        return [op for op in ground_ops if op.preconditions.issubset(atoms)]

    def _get_next_atoms(self, atoms: Set[GroundAtom], operator: Union[_GroundNSRT, List[_GroundNSRT]]) -> Set[GroundAtom]:
        """Get the next atoms after applying an operator."""
        next_atoms = atoms.copy()
        if isinstance(operator, list):
            for op in operator:
                next_atoms.update(op.add_effects)
                next_atoms.difference_update(op.delete_effects)
        else:
            next_atoms.update(operator.add_effects)
            next_atoms.difference_update(operator.delete_effects)
        return next_atoms

    def _get_shortest_path(self, init_atoms: Set[GroundAtom], goal_atoms: Set[GroundAtom], objects: Set[Object], max_steps: int = 1000) -> Optional[List[_GroundNSRT]]:
        """Find shortest path from init_atoms to goal_atoms using BFS."""
        frontier: List[Tuple[Set[GroundAtom], Optional[_GroundNSRT], List[_GroundNSRT]]] = [(init_atoms, None, [])]
        visited: Set[FrozenSet[GroundAtom]] = {frozenset(init_atoms)}
        steps = 0
        
        while frontier and steps < max_steps:
            curr_atoms, curr_op, path = frontier.pop(0)
            if goal_atoms.issubset(curr_atoms):
                return path
            
            for next_op in self._get_applicable_operators(curr_atoms, objects):
                next_atoms = self._get_next_atoms(curr_atoms, next_op)
                next_atoms_frozen = frozenset(next_atoms)
                if next_atoms_frozen not in visited:
                    visited.add(next_atoms_frozen)
                    frontier.append((next_atoms, next_op, path + [next_op]))
            steps += 1
        
        return None

    def _apply_operator(self, atoms: Set[GroundAtom], operator: _GroundNSRT) -> Set[GroundAtom]:
        """Apply an operator to get the next state atoms."""
        next_atoms = atoms.copy()
        next_atoms.update(operator.add_effects)
        next_atoms.difference_update(operator.delete_effects)
        return next_atoms

    def _create_task(self, init_atoms: Set[GroundAtom], goal_atoms: Set[GroundAtom], objects: List[Object]) -> EnvironmentTask:
        """Create a task from initial atoms, goal atoms, and objects."""
        # Store objects and NSRTs for operator application
        self.objects = {obj.name: obj for obj in objects}
        
        # Create initial state
        state_data = {obj: np.zeros(obj.type.dim, dtype=np.float32) for obj in objects}
        init_state = State(state_data, init_atoms)
        return EnvironmentTask(init_state, goal_atoms) 

    def _get_state_label(self, state_num: int, atoms: Set[GroundAtom], fluent_predicates: Set[Predicate],
                       is_init: bool = False, is_goal: bool = False,
                       self_loop_ops: Optional[List[str]] = None) -> str:
        """Get the label for a state node in the transition graph."""
        # Add state header
        prefix = ""
        if is_init:
            prefix = "Initial "
        elif is_goal:
            prefix = "Goal "
        label = [f"{prefix}State {state_num}"]
        label.append("─" * 30)  # Separator line
        
        # Group atoms by predicate
        atoms_by_pred: Dict[Predicate, List[GroundAtom]] = {}
        for atom in sorted(atoms, key=str):
            if atom.predicate not in fluent_predicates:
                continue
            if atom.predicate not in atoms_by_pred:
                atoms_by_pred[atom.predicate] = []
            atoms_by_pred[atom.predicate].append(atom)
        
        # Add predicates
        if not atoms_by_pred:
            label.append("No predicates")
        else:
            for pred, pred_atoms in sorted(atoms_by_pred.items(), key=lambda x: str(x[0])):
                for atom in sorted(pred_atoms, key=str):
                    # Format objects in bold
                    args_str = ", ".join(obj.name for obj in atom.objects)
                    pred_str = f"{pred.name}({args_str})"
                    label.append(pred_str)
        
        # Add self-loop operators if any
        if self_loop_ops:
            label.append("─" * 30)  # Separator line
            label.append("Self-loop operators:")
            for op in sorted(self_loop_ops):
                label.append(op)
        
        return "\n".join(label) 

    def get_operator_transitions(self, init_atoms: Set[GroundAtom], objects: Set[Object]) -> Set[Tuple[FrozenSet[GroundAtom], _GroundNSRT, FrozenSet[GroundAtom]]]:
        """Get all possible operator transitions from initial state.
        
        Args:
            init_atoms: Initial ground atoms
            objects: Objects in the environment
            
        Returns:
            Set of tuples (source_atoms, operator, dest_atoms) representing transitions
        """
        transitions = set()
        frontier: List[Tuple[Set[GroundAtom], Optional[_GroundNSRT]]] = [(init_atoms, None)]  # (state_atoms, parent_op)
        visited = {frozenset(init_atoms)}
        
        while frontier:
            curr_atoms, _ = frontier.pop(0)
            
            # Get applicable operators
            applicable_ops = self._get_applicable_operators(curr_atoms, objects)
            
            for op in applicable_ops:
                # Get next state
                next_atoms = self._get_next_atoms(curr_atoms, op)
                next_atoms_frozen = frozenset(next_atoms)
                
                # Add transition
                transitions.add((frozenset(curr_atoms), op, next_atoms_frozen))
                
                # Add to frontier if not visited
                if next_atoms_frozen not in visited:
                    visited.add(next_atoms_frozen)
                    frontier.append((next_atoms, None))  # Use None for parent_op to match type
        
        return transitions

    def get_graph_edges(self, init_atoms: Set[GroundAtom], goal_atoms: Set[GroundAtom], objects: Set[Object]) -> Set[Tuple[FrozenSet[GroundAtom], _GroundNSRT, FrozenSet[GroundAtom]]]:
        """Get edges in the transition graph based on initial and goal states.
        
        Args:
            init_atoms: Initial ground atoms
            goal_atoms: Goal ground atoms
            objects: Objects in the environment
            
        Returns:
            Set of tuples (source_atoms, operator, dest_atoms) representing edges
        """
        edges = set()
        visited = set()
        frontier: List[Tuple[Set[GroundAtom], Optional[_GroundNSRT]]] = [(init_atoms, None)]
        
        # Explore all reachable states and transitions
        while frontier:
            curr_atoms, _ = frontier.pop(0)
            curr_id = frozenset(curr_atoms)
            
            if curr_id in visited:
                continue
                
            visited.add(curr_id)
            
            # Get applicable operators
            applicable_ops = self._get_applicable_operators(curr_atoms, objects)
            
            # Add transitions to next states
            for op in applicable_ops:
                next_atoms = self._get_next_atoms(curr_atoms, op)
                next_id = frozenset(next_atoms)
                
                # Add edge
                edges.add((curr_id, op, next_id))
                
                # Add next state to frontier if not visited
                if next_id not in visited:
                    # Use type annotation to avoid type error
                    next_op: Optional[_GroundNSRT] = op
                    frontier.append((next_atoms, next_op))
        
        return edges 

    def _get_edge_style(self, op: _GroundNSRT, is_shortest_path: bool) -> Dict[str, str]:
        """Get edge style based on operator type and path status."""
        style = {
            'style': 'solid' if is_shortest_path else 'dashed',
            'color': '#4A90E2',  # Default blue
            'fontcolor': '#2E5894',
            'penwidth': '2.0' if is_shortest_path else '1.0'
        }
        
        # Check if operator affects belief predicates
        affects_belief = False
        for effect in (op.add_effects | op.delete_effects):
            if effect.predicate.name.startswith(('Believe', 'Known_', 'Unknown_')):
                affects_belief = True
                break
        
        if affects_belief:
            style.update({
                'color': '#E74C3C',  # Red for belief transitions
                'fontcolor': '#C0392B',
                'penwidth': '3.0' if is_shortest_path else '2.0'
            })
        
        return style 

    def _get_edge_label(self, op: _GroundNSRT) -> str:
        """Get detailed edge label showing operator effects."""
        # Basic operator name and args
        op_str = f"{op.name}({','.join(obj.name for obj in op.objects)})"
        
        # Format effects
        add_effects = [str(atom) for atom in op.add_effects]
        del_effects = [str(atom) for atom in op.delete_effects]
        
        # Build full label with HTML formatting
        label_parts = [
            f"<strong>{op_str}</strong>",
            "<br><br>",
            "<span style='color: #2ca02c'>",  # Green for add effects
            *[f"+ {eff}<br>" for eff in add_effects],
            "</span><br>",
            "<span style='color: #d62728'>",  # Red for delete effects
            *[f"- {eff}<br>" for eff in del_effects],
            "</span>"
        ]
        
        return "".join(label_parts) 

    def get_initial_observation(self) -> _SavedMockSpotObservation:
        """Get the initial observation."""
        return self.states["0"]

    def get_observation(self, state_id: str) -> _SavedMockSpotObservation:
        """Get the observation for a state."""
        return self.states[state_id]

    def get_next_state_id(self, action: Action) -> str:
        """Get the next state ID based on the action."""
        # Get operator name and objects from action
        operator_info = action.extra_info or {}
        operator_name = operator_info.get("operator_name")
        operator_objects = operator_info.get("objects", [])
        
        if operator_name is None:
            return self.current_state_id  # Stay in current state if no operator name
        
        # Find matching transition
        for source_id, op, dest_id in self.transitions:
            if (op.name == operator_name and 
                tuple(op.objects) == tuple(operator_objects)):
                self.current_state_id = dest_id  # Update current state
                return dest_id
        
        # If no transition found, stay in current state
        return self.current_state_id

    def add_transition(self, source_id: str, operator: _GroundNSRT, dest_id: str) -> None:
        """Add a transition to the environment."""
        self.transitions.append((source_id, operator, dest_id))
        # Also store string version for saving
        op_dict = {
            "name": operator.name,
            "objects": [obj.name for obj in operator.objects]
        }
        self.str_transitions.append((source_id, op_dict, dest_id))
        
    def _find_optimal_path(self, init_atoms: Optional[Set[GroundAtom]] = None, 
                          goal_atoms: Optional[Set[GroundAtom]] = None, 
                          objects: Optional[Set[Object]] = None) -> Optional[Tuple[List[_GroundNSRT], List[Set[GroundAtom]], int]]:
        """Find optimal path from initial state to goal state.
        
        Args:
            init_atoms: Initial state atoms (defaults to environment initial atoms)
            goal_atoms: Goal state atoms (defaults to environment goal atoms)
            objects: Objects in the environment (defaults to all objects)
            
        Returns:
            If path found, tuple containing:
            - List of operators (the plan)
            - List of states (sets of ground atoms)
            - Number of steps in plan
            None if no path found
        """
        # Get default values from environment
        init_atoms_to_use = self.env_initial_atoms if init_atoms is None else init_atoms
        goal_atoms_to_use = self.env_goal_atoms if goal_atoms is None else goal_atoms
        objects_to_use = set(self.objects.values()) if objects is None else objects
        
        planner = self._create_planner(init_atoms_to_use, goal_atoms_to_use, objects_to_use)
        try:
            skeleton, atoms_sequence, metrics = next(planner())
            return skeleton, atoms_sequence, len(skeleton)
        except StopIteration:
            return None

    def save_transitions(self, save_plan: bool = True, init_atoms: Optional[Set[GroundAtom]] = None, 
                        goal_atoms: Optional[Set[GroundAtom]] = None) -> None:
        """Save the transition system and state mapping to a YAML file.
        
        The transition system includes:
        1. Objects with their types and parent types
        2. States with their atoms and fluent atoms
        3. Transitions between states
        4. Metadata about predicates and types
        5. Optimal path information (if save_plan=True)
        6. State mapping information including:
           - Canonical state mappings
           - Unique view and world states
           - Atom to state mappings
           - Key atoms
        
        Args:
            save_plan: Whether to save optimal path information
            init_atoms: Initial state atoms (defaults to environment initial atoms)
            goal_atoms: Goal state atoms (defaults to environment goal atoms)
        """
        assert self.states or self.transitions, "States/Transitions not explored yet"
        
        # Use environment atoms as defaults
        init_atoms_to_use = self.env_initial_atoms if init_atoms is None else init_atoms
        goal_atoms_to_use = self.env_goal_atoms if goal_atoms is None else goal_atoms
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data structure
        system_data = {
            "objects": {
                obj.name: {
                    "type": obj.type.name,
                    "parent_type": obj.type.parent.name if obj.type.parent else None
                }
                for obj in self.objects.values()
            },
            "states": {
                state_id: {
                    "atoms": [str(atom) for atom in atoms],
                    "fluent_atoms": [str(atom) for atom in atoms 
                                   if atom.predicate.name in self.fluent_predicates]
                }
                for state_id, atoms in self.id_to_state.items()
            },
            "transitions": [
                {
                    "source": source_id,
                    "operator": {
                        "name": op_info["name"],
                        "objects": op_info["objects"]
                    },
                    "target": target_id
                }
                for source_id, op_info, target_id in self.str_transitions
            ],
            "metadata": {
                "fluent_predicates": list(self.fluent_predicates),
                "predicate_types": {
                    name: [t.name for t in pred.types] 
                    for name, pred in self.predicates.items()
                }
            }
        }
        
        # Add optimal path information if requested
        if save_plan:
            optimal_path = self._find_optimal_path(init_atoms_to_use, goal_atoms_to_use)
            if optimal_path is not None:
                plan, states, num_steps = optimal_path
                system_data["optimal_path"] = {
                    "num_steps": num_steps,
                    "plan": [
                        {
                            "operator": op.name,
                            "objects": [obj.name for obj in op.objects],
                            "state_id": self.state_to_id[frozenset(state)]
                        }
                        for op, state in zip(plan, states[1:])  # Skip initial state
                    ],
                    "initial_state_id": self.state_to_id[frozenset(init_atoms_to_use)],
                    "goal_state_id": self.state_to_id[frozenset(states[-1])]
                }
        
        # Add state mapping data
        system_data["state_mapping"] = {
            "state_to_canonical": self._state_to_canonical,
            "canonical_state_to_id": {k: list(v) for k, v in self._canonical_state_to_id.items()},
            "unique_view_states": {k: list(v) for k, v in self._unique_world_states.items()},
            "unique_world_states": {k: list(v) for k, v in self._unique_world_states.items()},
            "atom_to_states": {k: list(v) for k, v in self._atom_to_states.items()},
            "atom_to_canonical_states": {k: list(v) for k, v in self._atom_to_canonical_states.items()},
            "key_predicates": list(self._key_predicates)  # Save key predicates instead of atoms
        }
        
        # Save to YAML file
        output_path = self.output_dir / "transition_system.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(system_data, f, default_flow_style=False, sort_keys=False)

    def load_state_mapping(self) -> None:
        """Load the state mapping from the transition system YAML file."""
        transition_path = self.output_dir / "transition_system.yaml"
        if not transition_path.exists():
            return
            
        with open(transition_path) as f:
            system_data = yaml.safe_load(f)
            
        if "state_mapping" not in system_data:
            return
            
        mapping_data = system_data["state_mapping"]
        self._state_to_canonical = mapping_data["state_to_canonical"]
        self._canonical_state_to_id = {k: set(v) for k, v in mapping_data["canonical_state_to_id"].items()}
        self._unique_world_states = {k: set(v) for k, v in mapping_data["unique_view_states"].items()}
        self._atom_to_states = {k: set(v) for k, v in mapping_data["atom_to_states"].items()}
        self._atom_to_canonical_states = {k: set(v) for k, v in mapping_data["atom_to_canonical_states"].items()}
        # Load key predicates if available, otherwise use default
        self._key_predicates = set(mapping_data.get("key_predicates", KEY_PREDICATES))

    def get_unique_states(self) -> Iterator[Tuple[str, Set[str]]]:
        """Get iterator over unique states and their equivalent states.
        
        Returns:
            Iterator yielding (canonical_id, set of equivalent state IDs)
        """
        for canonical_id, equivalent_ids in self._canonical_state_to_id.items():
            yield canonical_id, equivalent_ids

    def get_states_with_atom(self, atom: Union[GroundAtom, str]) -> Set[str]:
        """Get all states that have a particular ground atom.
        
        Args:
            atom: Ground atom or its string representation
            
        Returns:
            Set of state IDs where the atom is true
        """
        atom_str = str(atom)
        return self._atom_to_states.get(atom_str, set())

    def get_canonical_states_with_atom(self, atom: Union[GroundAtom, str]) -> Set[str]:
        """Get all canonical states that have a particular ground atom.
        
        Args:
            atom: Ground atom or its string representation
            
        Returns:
            Set of canonical state IDs where the atom is true
        """
        atom_str = str(atom)
        return self._atom_to_canonical_states.get(atom_str, set())

    def get_key_atoms(self) -> Set[str]:
        """Get all key atoms that are being tracked.
        
        These are atoms whose predicates are in KEY_PREDICATES.
        """
        return self._key_predicates

    def add_key_atom(self, atom: Union[GroundAtom, str]) -> None:
        """Add a key atom to track.
        
        Only adds the atom if its predicate is in KEY_PREDICATES.
        
        Args:
            atom: Ground atom or its string representation to track
        """
        # Convert to string if needed
        atom_str = str(atom) if isinstance(atom, GroundAtom) else atom
        
        # Only add if the predicate is a key predicate
        predicate_name = atom_str.split("(")[0]
        if predicate_name in self._key_predicates:
            self._key_atoms.add(atom_str)
            
    def is_key_predicate(self, predicate_name: str) -> bool:
        """Check if a predicate is a key predicate.
        
        Args:
            predicate_name: Name of the predicate to check
            
        Returns:
            True if the predicate is in KEY_PREDICATES
        """
        return predicate_name in self._key_predicates

def create_graphviz_visualization(transitions: Set[Tuple[str, str, tuple, str]], 
                                task_name: str,
                                output_path: Path) -> None:
    """Create a static visualization using graphviz."""
    # Create graph
    dot = graphviz.Digraph(comment=f'Transition Graph for {task_name}')
    
    # Set graph attributes
    dot.attr('graph', {
        'fontname': 'Arial',
        'fontsize': '16',
        'label': f'State Transitions: {task_name}',
        'labelloc': 't',
        'nodesep': '1.0',
        'ranksep': '1.0',
        'splines': 'curved',
        'concentrate': 'false'
    })
    
    # Set node attributes
    dot.attr('node', {
        'fontname': 'Arial',
        'fontsize': '12',
        'shape': 'circle',
        'style': 'filled',
        'fillcolor': 'white',
        'width': '0.5',
        'height': '0.5',
        'margin': '0.1'
    })
    
    # Add nodes and edges
    visited_nodes = set()
    for source_id, op_name, op_objects, dest_id in transitions:
        # Add nodes if not visited
        for node_id in [source_id, dest_id]:
            if node_id not in visited_nodes:
                dot.node(node_id, f"State {node_id}")
                visited_nodes.add(node_id)
        
        # Format edge label
        edge_label = f"{op_name}({','.join(op_objects)})"
        
        # Add edge
        dot.edge(source_id, dest_id, edge_label)
    
    # Save graph
    dot.render(output_path, format='png', cleanup=True)

def create_interactive_visualization(graph_data: Dict[str, Any], output_path: Path) -> None:
    """Create an interactive visualization using Cytoscape.js."""
    # Convert graph data to Cytoscape.js format
    cytoscape_data = {
        'nodes': [{'data': node_data} for node_data in graph_data['nodes'].values()],
        'edges': [{'data': edge_data} for edge_data in graph_data['edges']]
    }
    
    # Read template
    template_path = os.path.join(os.path.dirname(__file__), 
                               "templates", "interactive_graph.html")
    with open(template_path) as f:
        template = Template(f.read())
    
    # Render template with graph data
    html_content = template.render(
        task_name=graph_data['metadata']['task_name'],
        graph_data_json=json.dumps(cytoscape_data)
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)

# def main():
#     """CLI interface for mapping images to states."""
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Map images to states for mock environment")
#     parser.add_argument("--output_dir", type=str, required=True,
#                        help="Directory to store environment data")
#     parser.add_argument("--image_dir", type=str, required=True,
#                        help="Directory containing raw images")
#     parser.add_argument("--env_name", type=str, required=True,
#                        help="Name of environment to use (e.g., mock_spot_drawer_cleaning)")
    
#     args = parser.parse_args()
    
#     # Import environment dynamically
#     from importlib import import_module
#     env_module = import_module("predicators.envs." + args.env_name)
#     env_class = getattr(env_module, args.env_name.title().replace("_", ""))
#     env = env_class()
    
#     # Create environment creator
#     creator = MockEnvCreatorBase(args.output_dir, env=env)
    
#     # Load existing state mapping if it exists
#     creator.load_state_mapping()
    
#     # Get list of image files
#     image_dir = Path(args.image_dir)
#     image_files = sorted(list(image_dir.glob("*.HEIC")))
    
#     if not image_files:
#         print("No HEIC images found in directory")
#         return
        
#     # Show available states
#     print("\nAvailable states:")
#     for canonical_id, equivalent_ids in creator.get_unique_states():
#         print(f"State {canonical_id} (equivalent to {equivalent_ids})")
#         state = creator.load_state(canonical_id)
#         print(f"  Objects in view: {[obj.name for obj in state.objects_in_view]}")
#         print(f"  Objects in hand: {[obj.name for obj in state.objects_in_hand]}")
#         print(f"  Gripper open: {state.gripper_open}")
#         print()
    
#     # Process each image
#     for img_file in image_files:
#         print(f"\nProcessing {img_file.name}")
        
#         # Show image info
#         print(f"Image size: {img_file.stat().st_size / 1024:.1f} KB")
        
#         # Get state ID from user
#         while True:
#             state_id = input("Enter state ID to map this image to (or 'skip' to skip): ")
#             if state_id == "skip":
#                 break
                
#             if state_id not in creator.id_to_state:
#                 print("Invalid state ID")
#                 continue
                
#             # Add image to state
#             creator.add_state_from_raw_images(
#                 raw_images={"cam1.seed0.rgb": (str(img_file), "rgb")},
#                 state_id=state_id,
#                 objects_in_view=creator.load_state(state_id).objects_in_view,
#                 objects_in_hand=creator.load_state(state_id).objects_in_hand,
#                 gripper_open=creator.load_state(state_id).gripper_open
#             )
#             print(f"Added image to state {state_id}")
#             break
    
#     # Save transitions (which now includes state mapping)
#     creator.save_transitions()
#     print("\nSaved transition system with state mapping")

# if __name__ == "__main__":
#     main() 