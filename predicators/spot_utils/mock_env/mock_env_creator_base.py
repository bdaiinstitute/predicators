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
from typing import Dict, List, Optional, Set, Tuple, Any, Sequence
import graphviz
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.table import Table
import numpy as np
from gym.spaces import Box
import yaml
import matplotlib.pyplot as plt
from graphviz import Digraph
from rich.progress import Progress
from collections import deque

from predicators.envs.mock_spot_env import MockSpotEnv
from predicators.ground_truth_models.mock_spot_env.nsrts import MockSpotGroundTruthNSRTFactory
from predicators.structs import (
    GroundAtom, EnvironmentTask, State, Task, Type, Predicate, 
    ParameterizedOption, NSRT, Object, Variable, LiftedAtom, STRIPSOperator,
    Action
)
from predicators.ground_truth_models import get_gt_options
from predicators import utils
from predicators.settings import CFG
from predicators.planning import task_plan_grounding, task_plan, run_task_plan_once


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

    def plan_and_visualize(self, init_atoms: Set[GroundAtom], goal_atoms: Set[GroundAtom], 
                          objects: List[Object], task_name: str = "task") -> None:
        """Plan and visualize transitions from initial state to goal.
        
        Args:
            init_atoms: Initial state atoms
            goal_atoms: Goal state atoms
            objects: List of objects in environment
            task_name: Name of task for visualization file
        """
        # Create initial state with empty data
        state_dict = {obj: np.zeros(obj.type.dim, dtype=np.float32) for obj in objects}
        init_state = State(state_dict, simulator_state=init_atoms)
        
        # Create task
        task = Task(init_state, goal_atoms)
        
        # Print initial state
        self.console.print("\n[bold cyan]Initial State:[/bold cyan]")
        self._visualize_state(init_atoms)
        
        # Print goal state
        self.console.print("\n[bold green]Goal State:[/bold green]")
        self._visualize_state(goal_atoms)
        
        # Run task planning
        self.console.print("\n[bold yellow]Planning...[/bold yellow]")
        plan, atoms_sequence, metrics = run_task_plan_once(
            task,
            self.nsrts,
            set(self.predicates.values()),
            set(self.types.values()),
            timeout=10.0,
            seed=CFG.seed,
            task_planning_heuristic="hadd",
            max_horizon=float("inf")
        )
        
        # Print state sequence
        self.console.print("\n[bold magenta]State Sequence:[/bold magenta]")
        for i, state_atoms in enumerate(atoms_sequence):
            self.console.print(f"\n[bold]State {i}:[/bold]")
            self._visualize_state(state_atoms)
        
        # Print plan steps
        self.console.print("\n[bold magenta]Plan Steps:[/bold magenta]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Step", style="dim")
        table.add_column("Action")
        table.add_column("State Changes")
        
        for i, (state_atoms, operator) in enumerate(zip(atoms_sequence, plan + [None])):
            # Get state changes
            prev_atoms = atoms_sequence[i-1] if i > 0 else init_atoms
            removed = prev_atoms - state_atoms
            added = state_atoms - prev_atoms
            
            # Format changes
            changes = []
            if removed:
                changes.extend([f"[red]- {atom}[/red]" for atom in sorted(removed, key=str)])
            if added:
                changes.extend([f"[green]+ {atom}[/green]" for atom in sorted(added, key=str)])
            
            # Add row
            table.add_row(
                str(i),
                str(operator) if operator else "[green]Goal Reached[/green]",
                "\n".join(changes) if changes else "[dim]No changes[/dim]"
            )
        
        self.console.print(table)
        
        # Print shortest path with full state information
        self.console.print("\n[bold magenta]Shortest Path to Goal:[/bold magenta]")
        path_tree = Tree("[bold cyan]Initial State[/bold cyan]")
        current_node = path_tree
        
        for i, (state_atoms, operator) in enumerate(zip(atoms_sequence, plan + [None])):
            # Get state changes
            prev_atoms = atoms_sequence[i-1] if i > 0 else init_atoms
            removed = prev_atoms - state_atoms
            added = state_atoms - prev_atoms
            
            # Create state description
            state_desc = []
            if removed:
                state_desc.extend([f"[red]- {atom}[/red]" for atom in sorted(removed, key=str)])
            if added:
                state_desc.extend([f"[green]+ {atom}[/green]" for atom in sorted(added, key=str)])
                
            # Add operator and next state as child nodes
            if operator:
                op_node = current_node.add(f"[yellow]{operator}[/yellow]")
                state_label = "Goal State" if i == len(plan) else f"State {i+1}"
                current_node = op_node.add(f"[cyan]{state_label}[/cyan]")
                if state_desc:
                    for change in state_desc:
                        current_node.add(change)
                else:
                    current_node.add("[dim]No changes[/dim]")
        
        self.console.print(path_tree)
        
        # Print metrics if available
        if metrics:
            self.console.print("\n[bold blue]Planning Metrics:[/bold blue]")
            metrics_table = Table(show_header=True, header_style="bold")
            metrics_table.add_column("Metric")
            metrics_table.add_column("Value")
            
            for key, value in metrics.items():
                metrics_table.add_row(key, str(value))
            
            self.console.print(metrics_table)
        
        # Visualize transitions
        self.console.print("\n[bold]Generating transition graph...[/bold]")
        self.visualize_transitions(task_name, init_atoms, goal_atoms, objects, metrics)

    def visualize_transitions(self, task_name: str, init_atoms: Set[GroundAtom], goal_atoms: Set[GroundAtom], 
                            objects: List[Object], metrics: Optional[Dict[str, Any]] = None) -> None:
        """Visualize all possible transitions from initial state to goal states.
        
        Creates a graph showing all reachable states and transitions between them.
        States are labeled with their atoms, grouped by object.
        Edges show the operator name and atom changes.
        The main path to the goal is highlighted.
        
        Args:
            task_name: Name of task for output file
            init_atoms: Initial state atoms
            goal_atoms: Goal state atoms 
            objects: List of objects in environment
            metrics: Optional metrics to display
        """
        # Explore all possible states
        self.console.print("\n[bold]Exploring possible states...[/bold]")
        with Progress() as progress:
            task = progress.add_task("[cyan]Exploring...", total=None)
            all_states = self.explore_all_states(init_atoms, objects)
            progress.update(task, completed=True)
        
        self.console.print(f"Found {len(all_states)} reachable states")
        
        # Create graph
        self.console.print("\n[bold]Creating transition graph...[/bold]")
        dot = Digraph(comment=f'Transitions for {task_name}')
        dot.attr(rankdir='TB')
        
        # Track visited states to avoid duplicates
        visited_states = set()
        visited_edges = set()
        
        # Helper to get state label
        def get_state_label(atoms: Set[GroundAtom]) -> str:
            # Group atoms by object
            atoms_by_obj = {}
            for atom in atoms:
                # Only show fluents and key predicates
                if atom.predicate.name not in self.fluent_predicates and \
                   atom.predicate.name not in self.key_predicates:
                    continue
                    
                # Group by first object
                obj = atom.objects[0].name
                if obj not in atoms_by_obj:
                    atoms_by_obj[obj] = []
                atoms_by_obj[obj].append(str(atom))
            
            # Format label
            label = ""
            for obj, obj_atoms in sorted(atoms_by_obj.items()):
                label += f"{obj}:\n"
                for atom in sorted(obj_atoms):
                    label += f"  {atom}\n"
            return label
        
        # Helper to get edge label
        def get_edge_label(src_atoms: Set[GroundAtom], dst_atoms: Set[GroundAtom], op: STRIPSOperator) -> str:
            # Get atom changes
            removed = src_atoms - dst_atoms
            added = dst_atoms - src_atoms
            
            # Only show fluents and key predicates
            removed = {a for a in removed if a.predicate.name in self.fluent_predicates or 
                                          a.predicate.name in self.key_predicates}
            added = {a for a in added if a.predicate.name in self.fluent_predicates or
                                       a.predicate.name in self.key_predicates}
            
            label = op.name + "\n"
            if removed:
                label += "Removed:\n"
                for atom in sorted(removed, key=str):
                    label += f"  - {atom}\n"
            if added:
                label += "Added:\n" 
                for atom in sorted(added, key=str):
                    label += f"  + {atom}\n"
            return label
        
        # Add all states and transitions
        with Progress() as progress:
            # Add states
            state_task = progress.add_task("[cyan]Adding states...", total=len(all_states))
            for state_id, state_atoms in all_states.items():
                # Add state if not visited
                state_key = frozenset(state_atoms)
                if state_key not in visited_states:
                    visited_states.add(state_key)
                    
                    # Check if this is initial, goal or intermediate state
                    is_init = state_atoms == init_atoms
                    is_goal = goal_atoms.issubset(state_atoms)
                    
                    # Set node attributes
                    attrs = {
                        'shape': 'box',
                        'style': 'filled',
                        'fillcolor': 'lightblue' if is_init else 'lightgreen' if is_goal else 'white',
                        'margin': '0.3,0.2'
                    }
                    
                    # Add node
                    dot.node(state_id, get_state_label(state_atoms), **attrs)
                progress.update(state_task, advance=1)
            
            # Add transitions
            edge_task = progress.add_task("[cyan]Adding transitions...", total=len(self.transitions))
            for src_id, dst_id, op in self.transitions:
                edge_key = (src_id, dst_id)
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    
                    # Get source and destination states
                    src_atoms = all_states[src_id]
                    dst_atoms = all_states[dst_id]
                    
                    # Add edge with style based on whether it's part of the shortest path
                    is_shortest_path = metrics and (src_id, dst_id) in metrics.get("shortest_path_edges", set())
                    edge_attrs = {
                        'style': 'solid' if is_shortest_path else 'dashed',
                        'color': 'black' if is_shortest_path else 'gray',
                        'penwidth': '2.0' if is_shortest_path else '1.0'
                    }
                    
                    # Add edge
                    dot.edge(src_id, dst_id, get_edge_label(src_atoms, dst_atoms, op), **edge_attrs)
                progress.update(edge_task, advance=1)
        
        # Save graph
        self.console.print("\n[bold]Saving transition graph...[/bold]")
        output_path = os.path.join(self.transitions_dir, f"{task_name}")
        dot.render(output_path, format='png', view=False, cleanup=True)
        
        # Print summary
        self.console.print(f"\n[bold cyan]Fluent predicates detected:[/bold cyan]")
        for pred in sorted(self.fluent_predicates):
            self.console.print(f"  - {pred}")
        
        # Print shortest path if available
        if metrics and "shortest_path" in metrics:
            self.console.print("\n[bold magenta]Shortest Path:[/bold magenta]")
            path_table = Table(show_header=True, header_style="bold")
            path_table.add_column("Step", style="dim")
            path_table.add_column("State")
            path_table.add_column("Action")
            
            for i, (state_id, op) in enumerate(metrics["shortest_path"]):
                state_atoms = all_states[state_id]
                path_table.add_row(
                    str(i),
                    get_state_label(state_atoms),
                    str(op) if op else "[green]Goal Reached[/green]"
                )
            
            self.console.print(path_table)

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
        """Explore all possible states from the initial state using BFS.
        
        Args:
            init_atoms: The initial state atoms.
            objects: The objects in the environment.
            
        Returns:
            A dictionary mapping state IDs to sets of ground atoms.
        """
        # Initialize variables for state exploration
        frontier = deque([init_atoms])
        visited = {self._get_state_id(init_atoms)}
        states = {self._get_state_id(init_atoms): init_atoms}
        self.fluent_predicates = set()
        self.transitions = []  # Reset transitions

        # Create initial state with zero features
        init_state = State({obj: np.zeros(obj.type.dim, dtype=np.float32) for obj in objects})

        # Explore states using BFS
        while frontier:
            state_atoms = frontier.popleft()
            state_id = self._get_state_id(state_atoms)

            # Get all ground NSRTs
            ground_nsrts = set()
            for nsrt in self.nsrts:
                ground_nsrts.update(utils.all_ground_nsrts(nsrt, objects))

            # Get applicable operators in current state
            applicable_ops = utils.get_applicable_operators(ground_nsrts, state_atoms)

            # Apply each operator to get successor states
            for op in applicable_ops:
                next_atoms = utils.apply_operator(op, state_atoms)
                next_state_id = self._get_state_id(next_atoms)

                # Track which predicates change
                for atom in next_atoms - state_atoms:
                    self.fluent_predicates.add(atom.predicate.name)
                for atom in state_atoms - next_atoms:
                    self.fluent_predicates.add(atom.predicate.name)

                # Add new state if not visited
                if next_state_id not in visited:
                    visited.add(next_state_id)
                    frontier.append(next_atoms)
                    states[next_state_id] = next_atoms

                # Record transition
                self.transitions.append((state_id, next_state_id, op))

        return states 