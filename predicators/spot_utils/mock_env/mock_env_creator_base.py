"""Base class for mock environment creators.

This module provides a base class for creating mock Spot environments with:
- Predefined states and transitions
- RGB-D observations with object detections
- Task-specific configurations
- Planning and visualization utilities
- Support for belief space planning (when enabled in environment)

The environment data is stored in a directory specified by CFG.mock_env_data_dir.
This includes:
- plan.yaml: Contains objects, states, and transitions
- images/: Directory containing RGB-D images for each state
- transitions/: Directory containing transition graph visualizations
- transitions/transitions.png: Main transition graph visualization
- transitions/cup_emptiness.png: Cup emptiness belief task transition graph

Configuration:
    mock_env_data_dir (str): Directory to store environment data (set during initialization)
    seed (int): Random seed for reproducibility
    sesame_task_planning_heuristic (str): Heuristic for task planning
    sesame_max_skeletons_optimized (int): Maximum number of skeletons to optimize

Example usage:
    ```python
    # Create environment creator
    creator = ManualMockEnvCreator("path/to/data_dir")
    
    # Create objects and predicates
    robot = Object("robot", creator.types["robot"])
    cup = Object("cup", creator.types["container"])
    
    # Create initial and goal atoms for belief space planning
    init_atoms = {
        GroundAtom(creator.predicates["HandEmpty"], [robot]),
        GroundAtom(creator.predicates["ContainingWaterUnknown"], [cup])
    }
    goal_atoms = {
        GroundAtom(creator.predicates["ContainingWaterKnown"], [cup])
    }
    
    # Plan and visualize transitions
    creator.plan_and_visualize(init_atoms, goal_atoms, objects)
    ```
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Optional, Any, Tuple, FrozenSet, Iterator, Union
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

from predicators.envs.mock_spot_env import MockSpotEnv
from predicators.ground_truth_models.mock_spot_env.nsrts import MockSpotGroundTruthNSRTFactory
from predicators.structs import (
    GroundAtom, EnvironmentTask, State, Task, Type, Predicate, 
    ParameterizedOption, NSRT, Object, Variable, LiftedAtom, STRIPSOperator,
    Action, DefaultState, _GroundNSRT
)
from predicators.ground_truth_models import get_gt_options
from predicators import utils
from predicators.settings import CFG
from predicators.planning import task_plan_grounding, task_plan, run_task_plan_once

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        env (MockSpotEnv): Mock environment instance
        types (Dict[str, Type]): Available object types
        predicates (Dict[str, Predicate]): Available predicates (including belief predicates if enabled)
        options (Dict[str, ParameterizedOption]): Available options
        nsrts (Set[NSRT]): Available NSRTs (including belief space operators when enabled)
        console (Console): Rich console for pretty printing
    """

    def __init__(self, output_dir: str, env: Optional[MockSpotEnv] = None) -> None:
        """Initialize the mock environment creator.
        
        Args:
            output_dir: Directory to save output files
            env: Environment instance to use (default: creates new MockSpotEnv)
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.path_dir = output_dir
        self.image_dir = os.path.join(output_dir, "images")
        self.transitions_dir = os.path.join(output_dir, "transitions")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.transitions_dir, exist_ok=True)
        
        # Initialize state storage
        self.states: Dict[str, Dict[str, Any]] = {}
        self.transitions: List[Tuple[str, str, Any]] = []
        
        # Set data directory in config for environment to use
        utils.reset_config({
            "mock_env_data_dir": output_dir
        })
        
        # Use provided environment or create default one
        self.env = env if env is not None else MockSpotEnv()
        
        # Get environment info
        self.types: Dict[str, Type] = {t.name: t for t in self.env.types}
        self.predicates: Dict[str, Predicate] = {p.name: p for p in self.env.predicates}
        self.options: Dict[str, ParameterizedOption] = {o.name: o for o in get_gt_options(self.env.get_name())}
        self.nsrts: Set[NSRT] = self._create_nsrts()

        # Calculate fluent predicates by looking at operator effects
        self.fluent_predicates: Set[str] = self._calculate_fluent_predicates()
        
        # Initialize rich console for pretty printing
        self.console = Console()

    def _format_atoms(self, atoms: set) -> str:
        """Format atoms for display, showing only fluent predicates."""
        formatted_atoms = []
        for atom in sorted(atoms, key=str):
            # Only show fluent predicates
            if atom.predicate.name in self.fluent_predicates:
                args = [obj.name for obj in atom.objects]
                formatted_atoms.append(f"{atom.predicate.name}({', '.join(args)})")
        return "\n".join(formatted_atoms)

    def _create_nsrts(self) -> Set[NSRT]:
        """Create NSRTs from the environment's predicates and options.
        
        Returns NSRTs that can work with both physical and belief predicates
        (when belief space operators are enabled in the environment).
        """
        # Get NSRTs from factory
        factory = MockSpotGroundTruthNSRTFactory()
        # This now uses predicates from env including belief ones if enabled
        base_nsrts = factory.get_nsrts(
            self.env.get_name(),
            self.types,
            self.predicates,  # This now uses predicates from env including belief ones if enabled
            self.options
        )
        return base_nsrts

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

    def add_state(self, 
                 state_id: str,
                 views: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                 objects_in_view: Set[Object],
                 objects_in_hand: Set[Object],
                 gripper_open: bool = True) -> None:
        """Add a state to the environment.
        
        Args:
            state_id: Unique identifier for this state
            views: Dict mapping view names to camera names to image names and arrays
                Example: {
                    "view1": {
                        "cam1": {
                            "image1": array1,
                            "image2": array2
                        },
                        "cam2": {
                            "image1": array1
                        }
                    }
                }
            objects_in_view: Set of object names visible in the image
            objects_in_hand: Set of object names being held
            gripper_open: Whether the gripper is open
        """
        # Create state directory
        state_dir = os.path.join(self.image_dir, state_id)
        os.makedirs(state_dir, exist_ok=True)
        
        # Save images for each view and camera
        for view_name, cameras in views.items():
            # Create view directory
            view_dir = os.path.join(state_dir, view_name)
            os.makedirs(view_dir, exist_ok=True)
            
            for camera_name, images in cameras.items():
                # Save each image
                for image_name, image_data in images.items():
                    image_path = os.path.join(view_dir, f"{camera_name}_{image_name}.npy")
                    np.save(image_path, image_data)
        
        # Create state metadata
        state_data = {
            "objects_in_view": [obj.name for obj in objects_in_view],
            "objects_in_hand": [obj.name for obj in objects_in_hand],
            "gripper_open": gripper_open,
            "views": {
                view_name: {
                    camera_name: {
                        f"{image_name}_path": os.path.join(view_name, f"{camera_name}_{image_name}.npy")
                        for image_name in images.keys()
                    }
                    for camera_name, images in cameras.items()
                }
                for view_name, cameras in views.items()
            }
        }
        
        # Save state metadata
        metadata_path = os.path.join(state_dir, "metadata.yaml")
        with open(metadata_path, "w") as f:
            yaml.safe_dump(state_data, f, default_flow_style=False, sort_keys=False)

    def load_state(self, state_id: str) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], 
                                                List[str], List[str], bool]:
        """Load a state's images and metadata.
        
        Args:
            state_id: ID of state to load
            
        Returns:
            Tuple containing:
            - Dict mapping view names to camera names to image names and arrays
            - List of objects in view
            - List of objects in hand
            - Gripper open state
            
        Raises:
            FileNotFoundError: If state directory or metadata doesn't exist
        """
        state_dir = os.path.join(self.image_dir, state_id)
        if not os.path.exists(state_dir):
            raise FileNotFoundError(f"State directory not found: {state_dir}")
            
        # Load metadata
        metadata_path = os.path.join(state_dir, "metadata.yaml")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
            
        # Load images for each view and camera
        views = {}
        for view_name, cameras in metadata["views"].items():
            views[view_name] = {}
            view_dir = os.path.join(state_dir, view_name)
            for camera_name, paths in cameras.items():
                views[view_name][camera_name] = {}
                
                # Load each image
                for path_key, rel_path in paths.items():
                    image_name = path_key.replace("_path", "")
                    image_path = os.path.join(state_dir, rel_path)
                    views[view_name][camera_name][image_name] = np.load(image_path)
                
        return (
            views,
            metadata["objects_in_view"],
            metadata["objects_in_hand"],
            metadata["gripper_open"]
        )

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
        plan_path = os.path.join(self.path_dir, "plan.yaml")
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
        plan_path = os.path.join(self.path_dir, "plan.yaml")
        if not os.path.exists(plan_path):
            return None
            
        with open(plan_path, "r") as f:
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

    def plan_and_visualize(self, initial_atoms: Set[GroundAtom],
                          goal_atoms: Set[GroundAtom],
                          objects: Set[Object],
                          task_name: str = "Task",
                          use_graphviz: bool = False) -> None:
        """Plan and create visualizations for the transition graph.
        
        Args:
            initial_atoms: Initial state atoms
            goal_atoms: Goal state atoms
            objects: Objects in the environment
            task_name: Name of the task for visualization
            use_graphviz: Whether to use graphviz instead of cytoscape.js
        """
        # Get transitions and edges
        transitions = self.get_operator_transitions(initial_atoms, objects)
        edges = self.get_graph_edges(initial_atoms, goal_atoms, objects)
        
        # Create state to ID mapping
        state_to_id = {}
        state_count = 0
        
        # Start with initial state
        initial_state = frozenset(initial_atoms)
        state_to_id[initial_state] = "0"
        
        # First assign IDs to states in the shortest path
        curr_atoms = initial_atoms
        for edge in edges:
            next_atoms = edge[2]
            next_state = frozenset(next_atoms)
            if next_state not in state_to_id:
                state_to_id[next_state] = str(state_count + 1)
                state_count += 1
        
        # Then assign IDs to remaining states
        for source_atoms, _, dest_atoms in transitions:
            source_state = frozenset(source_atoms)
            dest_state = frozenset(dest_atoms)
            
            if source_state not in state_to_id:
                state_to_id[source_state] = str(state_count + 1)
                state_count += 1
            
            if dest_state not in state_to_id:
                state_to_id[dest_state] = str(state_count + 1)
                state_count += 1
        
        # Track shortest path edges and states
        shortest_path_edges = set()
        shortest_path_states = {frozenset(initial_atoms)}
        
        # Get plan to goal
        planner = self._create_planner(initial_atoms, goal_atoms, objects)
        try:
            skeleton, atoms_sequence, metrics = next(planner())
            # Follow plan to get shortest path
            curr_atoms = initial_atoms.copy()
            for op in skeleton:
                next_atoms = self._get_next_atoms(curr_atoms, op)
                source_id = state_to_id[frozenset(curr_atoms)]
                dest_id = state_to_id[frozenset(next_atoms)]
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
            source_id = state_to_id[source_state]
            dest_id = state_to_id[dest_state]
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
            trans_ops = {(state_to_id[frozenset(t[0])], t[1].name,
                        tuple(obj.name for obj in t[1].objects),
                        state_to_id[frozenset(t[2])]) for t in transitions}
            # Use task name for the output file
            output_path = os.path.join(self.output_dir, "transitions", task_name)
            create_graphviz_visualization(trans_ops, task_name, output_path)
        else:
            # Create interactive visualization
            # Track shortest path edges and states
            shortest_path_edges = set()
            shortest_path_states = {frozenset(initial_atoms)}
            
            for source_atoms, op, dest_atoms in edges:
                source_state = frozenset(source_atoms)
                dest_state = frozenset(dest_atoms)
                source_id = state_to_id[source_state]
                dest_id = state_to_id[dest_state]
                shortest_path_edges.add((source_id, dest_id))
                shortest_path_states.add(dest_state)
            
            # Create edge data
            edge_data = []
            edge_count = 0
            for source_atoms, op, dest_atoms in transitions:
                source_state = frozenset(source_atoms)
                dest_state = frozenset(dest_atoms)
                source_id = state_to_id[source_state]
                dest_id = state_to_id[dest_state]
                if source_id != dest_id:  # Skip self-loops
                    # Create detailed edge label
                    op_str = f"{op.name}({','.join(obj.name for obj in op.objects)})"
                    edge_data.append({
                        'id': f'edge_{edge_count}',
                        'source': source_id,
                        'target': dest_id,
                        'label': op_str,
                        'fullLabel': self._get_edge_label(op),
                        'is_shortest_path': (source_id, dest_id) in shortest_path_edges,
                        'affects_belief': any(effect.predicate.name.startswith(('Believe', 'Known_', 'Unknown_')) 
                                           for effect in (op.add_effects | op.delete_effects))
                    })
                    edge_count += 1
            
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
            for atoms, state_id in state_to_id.items():
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
                    self._format_atoms(atoms)
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
            output_path = os.path.join(self.output_dir, "transitions", f"{task_name}.html")
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
        graph_path = os.path.join(self.transitions_dir, f"{task_name}")
        dot.render(graph_path, format='png', cleanup=True)

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

    def explore_all_states(self, init_atoms: Set[GroundAtom], objects: List[Object]) -> Dict[str, Set[GroundAtom]]:
        """Explore all possible states from initial state using BFS."""
        # Initialize state exploration
        all_states = {}  # Maps state IDs to sets of ground atoms
        frontier: deque[Tuple[Set[GroundAtom], Optional[str]]] = deque([(init_atoms, None)])  # (state_atoms, parent_id)
        visited = set()  # Set of visited state IDs
        self.transitions = []  # List of (source_id, destination_id, operator) tuples
        self.fluent_predicates = set()  # Set of predicates that change during exploration

        while frontier:
            state_atoms, parent_id = frontier.popleft()
            state_id = self._get_state_id(state_atoms)

            # Skip if already visited
            if state_id in visited:
                continue

            # Add to visited and all_states
            visited.add(state_id)
            all_states[state_id] = state_atoms

            # Get applicable operators
            state = State({obj: np.zeros(obj.type.dim, dtype=np.float32) for obj in objects})
            state.simulator_state = state_atoms

            # Ground NSRTs and get reachable atoms
            ground_nsrts, _ = task_plan_grounding(
                init_atoms=state_atoms,
                objects=set(objects),
                nsrts=self.nsrts,
                allow_noops=False
            )

            # Try each operator
            for nsrt in ground_nsrts:
                # Apply operator to get next state
                next_atoms = state_atoms.copy()
                
                # Remove deleted atoms
                for atom in nsrt.delete_effects:
                    if atom in next_atoms:
                        next_atoms.remove(atom)
                        # Track fluent predicates
                        self.fluent_predicates.add(atom.predicate.name)

                # Add new atoms
                for atom in nsrt.add_effects:
                    if atom not in next_atoms:
                        next_atoms.add(atom)
                        # Track fluent predicates
                        self.fluent_predicates.add(atom.predicate.name)

                # Get next state ID
                next_state_id = self._get_state_id(next_atoms)

                # Add transition
                self.transitions.append((state_id, next_state_id, nsrt))

                # Add to frontier if not visited
                if next_state_id not in visited:
                    frontier.append((next_atoms, state_id))

        return all_states 

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
        self.objects = set(objects)
        self.nsrts = self._create_nsrts()
        
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

def create_graphviz_visualization(transitions: Set[Tuple[str, str, tuple, str]], 
                                task_name: str,
                                output_path: str) -> None:
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

def create_interactive_visualization(graph_data: Dict[str, Any], output_path: str) -> None:
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