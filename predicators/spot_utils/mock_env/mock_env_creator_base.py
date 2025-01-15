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

    def __init__(self, path_dir: str) -> None:
        """Initialize the creator.
        
        Sets up the environment data directory and initializes a mock Spot environment.
        The path_dir is stored in CFG.mock_env_data_dir for use by the environment.
        Creates necessary subdirectories for images and transition graphs.
        
        Args:
            path_dir: Directory to store environment data. Will be set as CFG.mock_env_data_dir.
        """
        self.path_dir = path_dir
        self.image_dir = os.path.join(path_dir, "images")
        self.transitions_dir = os.path.join(path_dir, "transitions")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.transitions_dir, exist_ok=True)
        
        # Initialize state storage
        self.states: Dict[str, Dict[str, Any]] = {}
        self.transitions: List[Tuple[str, str, Any]] = []
        
        # Initialize predicate tracking
        self.fluent_predicates: Set[str] = set()
        self.key_predicates: Set[str] = {'Holding', 'Inside', 'On', 'HandEmpty', 'ContainingWaterKnown'}
        
        # Set data directory in config for environment to use
        utils.reset_config({
            "mock_env_data_dir": path_dir
        })
        
        # Initialize environment (will use CFG.mock_env_data_dir)
        self.env = MockSpotEnv()
        
        # Get environment info
        self.types: Dict[str, Type] = {t.name: t for t in self.env.types}
        # Get predicates directly from environment - this will include belief predicates if enabled
        self.predicates: Dict[str, Predicate] = {p.name: p for p in self.env.predicates}
        
        # Create options
        self.options: Dict[str, ParameterizedOption] = {o.name: o for o in get_gt_options(self.env.get_name())}
        
        # Get NSRTs from factory
        self.nsrts: Set[NSRT] = self._create_nsrts()

        # Initialize rich console for pretty printing
        self.console = Console()

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

    def add_state(self, state_id: str, views: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                 objects_in_view: Optional[List[str]] = None,
                 objects_in_hand: Optional[List[str]] = None,
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
            objects_in_view: List of object names visible in the image
            objects_in_hand: List of object names being held
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
            "objects_in_view": objects_in_view or [],
            "objects_in_hand": objects_in_hand or [],
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

    def plan_and_visualize(self, init_atoms: Set[GroundAtom], goal: Set[GroundAtom], objects: Set[Object], *, task_name: str = "task") -> None:
        """Plan and visualize transitions for a task."""
        # Create rich console
        console = Console()

        # Plan first
        result = self.plan(init_atoms, goal, objects)
        if result is None:
            console.print("[red]No plan found![/red]")
            return
        
        plan, atoms_sequence, metrics = result
        
        # Print plan found message
        console.print(f"\n[green]Plan found with {len(plan)} steps![/green]")
        
        # Create and style table
        table = Table(
            title="Plan Steps",
            show_header=True,
            header_style="bold magenta",
            title_style="bold blue",
            box=HEAVY,
            safe_box=True,
            expand=True,
            show_lines=True
        )
        
        # Add columns
        table.add_column("Step", style="cyan", justify="center", width=5)
        table.add_column("Operator", style="green", width=30)
        table.add_column("Added Atoms", style="green", width=54)
        table.add_column("Removed Atoms", style="red", width=54)
        
        # Add rows for each step
        for i, (op, next_atoms) in enumerate(zip(plan, atoms_sequence[1:]), 1):
            # Calculate atom changes
            prev_atoms = atoms_sequence[i-1]
            added_atoms = next_atoms - prev_atoms
            removed_atoms = prev_atoms - next_atoms
            
            # Format atoms
            added_str = "\n".join(str(atom) for atom in sorted(added_atoms, key=str))
            removed_str = "\n".join(str(atom) for atom in sorted(removed_atoms, key=str))
            
            # Add row
            table.add_row(
                str(i),
                op.name,
                added_str if added_atoms else "",
                removed_str if removed_atoms else ""
            )
        
        # Print the table
        console.print("\n")
        console.print(table)
        
        # Print plan metrics in a styled panel
        metrics_table = Table(show_header=False, box=None)
        metrics_table.add_column("Metric", style="blue")
        metrics_table.add_column("Value", style="cyan")
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metrics_table.add_row(key, str(value))
        
        metrics_panel = Panel(
            metrics_table,
            title="Plan Metrics",
            title_align="left",
            border_style="blue",
            padding=(1, 2)
        )
        console.print("\n")
        console.print(metrics_panel)
        
        # Visualize transitions
        console.print("\n[yellow]Generating transition graph...[/yellow]")
        self.visualize_transitions(task_name, init_atoms, goal, list(objects), plan)
        console.print(f"[green]Transition graph saved to {self.transitions_dir}/{task_name}.png[/green]")

    def visualize_transitions(self, task_name: str, init_atoms: Set[GroundAtom], goal: Set[GroundAtom], objects: List[Object], plan: Optional[List[_GroundNSRT]] = None) -> None:
        """Visualize transitions for a task."""
        # Track shortest path states and fluent predicates
        shortest_path_states = set()
        fluent_predicates = set()
        state_numbers = {}  # Map state atoms to numbers
        state_count = 0
        
        if plan is not None:
            logger.info("Checking transitions from plan...")
            curr_atoms = init_atoms
            shortest_path_states.add(frozenset(curr_atoms))
            prev_atoms = curr_atoms
            state_numbers[frozenset(curr_atoms)] = state_count
            state_count += 1
            
            for op in plan:
                # Verify operator preconditions
                preconditions_satisfied = all(precond in curr_atoms for precond in op.preconditions)
                logger.info(f"Plan operator {op}: preconditions satisfied = {preconditions_satisfied}")
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
        
        # Create graph with improved settings
        dot = graphviz.Digraph(comment=f'Transition Graph for {task_name}')
        dot.attr(rankdir='TB')  # Top to bottom layout
        
        # Set global graph attributes        # Set graph attributes
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
            'fillcolor': 'white',
            'margin': '0.3',
            'width': '2.5',
            'height': '1.5'
        })
        
        # Set edge attributes
        dot.attr('edge', {
            'fontname': 'Arial',
            'fontsize': '9',
            'arrowsize': '0.8',
            'penwidth': '1.0',
            'len': '1.2',
            'decorate': 'true',
            'labelfloat': 'true',
            'labelangle': '25',
            'labeldistance': '1.5',
            'minlen': '1',
            'color': '#4A90E2',
            'fontcolor': '#E74C3C',
            'arrowhead': 'normal',
            'arrowcolor': '#E74C3C',
            'weight': '1.0'
        })
        
        # Track visited states and transitions
        visited = set()
        transitions = set()  # Use set to avoid duplicates
        state_self_loops = {}  # Track self-loops for each state
        
        # Initialize frontier with initial state
        frontier: List[Tuple[Set[GroundAtom], Optional[Union[_GroundNSRT, None]]]] = [(init_atoms, None)]
        
        # Explore all possible states
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
            
            # Get applicable operators
            applicable_ops = self._get_applicable_operators(current_atoms, set(objects))
            
            # Track self-loops for this state
            self_loops = []
            
            # Add transitions to next states
            for op in applicable_ops:
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
                
                # Add non-self-loop transition if not already present
                transition = (current_id, next_id, op)
                if transition not in transitions:
                    transitions.add(transition)
                    edge_attrs = {
                        'label': op_label,
                        'style': 'solid' if frozenset(current_atoms) in shortest_path_states else 'dashed',
                        'color': '#4A90E2',  # Blue for edges
                        'fontcolor': '#2E5894',  # Darker blue for labels
                        'arrowhead': 'normal',
                        'arrowcolor': '#E74C3C',  # Red arrows
                        'labelfloat': 'true',
                        'decorate': 'true',
                        'labelangle': '25',
                        'labeldistance': '1.5'
                    }
                    dot.edge(current_id, next_id, **edge_attrs)
                
                # Add next state to frontier if not visited
                if next_id not in visited:
                    frontier.append((next_atoms, op))
            
            # Store self-loops for this state
            if self_loops:
                state_self_loops[current_id] = self_loops
            
            # Add node for current state with self-loops in label
            is_initial = (current_atoms == init_atoms)
            is_goal = goal.issubset(current_atoms)
            is_shortest_path = frozenset(current_atoms) in shortest_path_states
            
            node_attrs = {
                'label': self._get_state_label(state_num, current_atoms, fluent_predicates, 
                                              is_initial, is_goal, state_self_loops.get(current_id)),
                'style': 'rounded,filled'
            }
            
            if is_initial:
                node_attrs['fillcolor'] = '#ADD8E6'  # Light blue
                node_attrs['penwidth'] = '2.0'
            elif is_goal:
                node_attrs['fillcolor'] = '#90EE90'  # Light green
                node_attrs['penwidth'] = '2.0'
            elif is_shortest_path:
                node_attrs['fillcolor'] = '#FFFF99'  # Light yellow
            else:
                node_attrs['fillcolor'] = 'white'
                
            # Add dashed border for unreachable states
            if not is_shortest_path and not is_initial:
                node_attrs['style'] = 'rounded,filled,dashed'
            
            dot.node(current_id, **node_attrs)
        
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
        """Get color for state visualization based on its atoms.
        
        Args:
            atoms: Ground atoms in the state
            
        Returns:
            Hex color code for the state
        """
        # Default colors for common predicates
        colors = {
            "HandEmpty": "#90EE90",  # Light green
            "Holding": "#FFB6C1",    # Light pink
            "On": "#ADD8E6",         # Light blue
            "Inside": "#DDA0DD",     # Plum
        }
        
        # Special colors for belief predicates
        belief_colors = {
            "ContainingWaterUnknown": "#F0E68C",  # Khaki
            "ContainingWaterKnown": "#98FB98",    # Pale green
            "Empty": "#FFA07A",                   # Light salmon
        }
        
        # Find matching predicate
        for atom in atoms:
            if atom.predicate.name in colors:
                return colors[atom.predicate.name]
            if atom.predicate.name in belief_colors:
                return belief_colors[atom.predicate.name]
                
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
        # Only show fluents and key predicates
        important_atoms = {atom for atom in state 
                         if atom.predicate.name in (self.fluent_predicates | self.key_predicates)}
        
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

    @abstractmethod
    def create_rgbd_image(self, rgb: np.ndarray, depth: np.ndarray,
                         camera_name: str = "hand_color") -> Any:
        """Create an RGBD image from RGB and depth arrays.
        
        Args:
            rgb: RGB image array
            depth: Depth image array
            camera_name: Name of camera that captured the image
            
        Returns:
            RGBD image object (implementation specific)
        """
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

    def _get_next_atoms(self, atoms: Set[GroundAtom], operator: _GroundNSRT) -> Set[GroundAtom]:
        """Get the next atoms after applying an operator."""
        next_atoms = atoms.copy()
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