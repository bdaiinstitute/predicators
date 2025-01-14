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
from predicators.planning import task_plan_grounding, task_plan


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
                         objects: Set[Object], *, task_name: str = "task",
                         show_metrics: bool = True, show_state_viz: bool = True,
                          timeout: float = 10.0) -> None:
        """Plan and visualize a task, showing progress and creating transition graphs.
        
        This method:
        1. Displays available options and NSRTs
        2. Shows initial state and goal atoms
        3. Grounds NSRTs and shows reachable atoms
        4. Creates a plan to achieve the goal
        5. Visualizes the plan steps and transitions
        6. Creates transition graph visualizations:
           - Main graph showing all transitions (transitions/transitions.png)
           - Task-specific graphs (e.g. transitions/cup_emptiness.png for belief tasks)
        
        For belief space planning tasks, this will show belief state transitions
        and create belief-specific visualizations.
        
        Args:
            init_atoms: Initial ground atoms
            goal_atoms: Goal ground atoms
            objects: Objects in the environment
            task_name: Name of task for visualization files
            show_metrics: Whether to show plan metrics after completion
            show_state_viz: Whether to show state visualizations during execution
            timeout: Planning timeout in seconds
        """
        # Create options table
        options_table = Table(title="Available Options")
        options_table.add_column("Option", style="cyan")
        options_table.add_column("Types", style="green")
        options_table.add_column("Parameter Space", style="yellow")
        
        for name, option in self.options.items():
            options_table.add_row(
                name,
                ", ".join(t.name for t in option.types),
                str(option.params_space)
            )
        self.console.print(options_table)

        # Create NSRTs tree
        nsrts_tree = Tree("[bold blue]Created NSRTs")
        for i, nsrt in enumerate(self.nsrts, 1):
            nsrt_node = nsrts_tree.add(f"[cyan]{i}. {nsrt.name}")
            nsrt_node.add(f"[green]Parameters: {[p.name for p in nsrt.parameters]}")
            nsrt_node.add(f"[yellow]Preconditions: {nsrt.preconditions}")
            nsrt_node.add(f"[red]Add effects: {nsrt.add_effects}")
            nsrt_node.add(f"[magenta]Delete effects: {nsrt.delete_effects}")
        self.console.print(nsrts_tree)
        
        # Create initial state panel
        init_atoms_panel = Panel(
            "\n".join(str(atom) for atom in sorted(init_atoms, key=str)),
            title="Initial State Atoms",
            border_style="green"
        )
        self.console.print(init_atoms_panel)
        
        # Create goal atoms panel
        goal_atoms_panel = Panel(
            "\n".join(str(atom) for atom in goal_atoms),
            title="Goal Atoms",
            border_style="red"
        )
        self.console.print(goal_atoms_panel)
        
        # Plan
        result = self.plan(init_atoms, goal_atoms, objects, timeout)
        if result is None:
            self.console.print("[red]No plan found!")
            return
            
        plan, atoms_sequence, metrics = result
            
        # Show plan metrics if requested
        if show_metrics:
            self.console.print(f"[green]Plan found with {len(plan)} steps!")
        
        # Create plan table
        plan_table = Table(title="Plan Steps")
        plan_table.add_column("Step", style="cyan")
        plan_table.add_column("Operator", style="green")
        plan_table.add_column("Added Atoms", style="bold green")
        plan_table.add_column("Removed Atoms", style="bold red")
        
        for i, (nsrt, atoms) in enumerate(zip(plan, atoms_sequence[1:]), 1):
            prev_atoms = atoms_sequence[i-1]
            new_atoms = atoms - prev_atoms
            removed_atoms = prev_atoms - atoms
            
            plan_table.add_row(
                str(i),
                nsrt.name,
                "\n".join(str(atom) for atom in sorted(new_atoms, key=str)),
                "\n".join(str(atom) for atom in sorted(removed_atoms, key=str))
            )
        self.console.print(plan_table)
        
        # Create metrics panel
        metrics_panel = Panel(
            "\n".join(f"{key}: {value}" for key, value in metrics.items()),
            title="Plan Metrics",
            border_style="blue"
        )
        self.console.print(metrics_panel)
        
        # Execute plan and visualize states
        state = init_atoms
        for i, op in enumerate(plan):
            if show_state_viz:
                self.console.print(f"\nStep {i}: {op}")
                self._visualize_state(state)
            # Apply operator effects
            state = state - op.delete_effects | op.add_effects
            
        # Save plan data
        self.save_plan(plan, atoms_sequence, init_atoms, goal_atoms, objects)
            
        # Create transition graph visualizations
        self.visualize_transitions(atoms_sequence, plan, goal_atoms, task_name)
        
        # Verify plan achieves goal
        assert goal_atoms.issubset(atoms_sequence[-1])

    def visualize_transitions(self, atoms_sequence: List[Set[GroundAtom]], 
                            plan: List[Any], goal_atoms: Set[GroundAtom],
                            task_name: str = "task") -> None:
        """Visualize the transition graph focusing on fluents and key predicates.
        
        Creates a graphviz visualization showing:
        - States as nodes with only changed predicates
        - Transitions as edges with operator names
        - Color coding for initial, intermediate, and goal states
        - Vertical layout for better readability
        
        Args:
            atoms_sequence: Sequence of atom sets representing states
            plan: Sequence of operators
            goal_atoms: Goal state atoms for coloring final state
            task_name: Name of task for visualization file
        """
        # Create a new directed graph
        dot = graphviz.Digraph(comment='Transition Graph')
        dot.attr(rankdir='TB')  # Top to bottom layout
        dot.attr('node', shape='box', style='rounded,filled', fontsize='10')
        
        # Track which predicates actually change (fluents)
        all_atoms = set().union(*atoms_sequence)
        fluent_predicates = set()
        key_predicates = {'Holding', 'Inside', 'On', 'HandEmpty', 'ContainingWaterKnown'}
        
        # Find predicates that change
        for i in range(len(atoms_sequence) - 1):
            curr_atoms = atoms_sequence[i]
            next_atoms = atoms_sequence[i + 1]
            changed_atoms = curr_atoms.symmetric_difference(next_atoms)
            for atom in changed_atoms:
                fluent_predicates.add(atom.predicate.name)
        
        # Add nodes and edges
        for i, (atoms, nsrt) in enumerate(zip(atoms_sequence, plan + [None])):
            state_id = str(i)
            state_label = f"State {i}\\n"
            
            # Only show fluents and key predicates
            important_atoms = []
            for atom in sorted(atoms, key=str):
                if (atom.predicate.name in fluent_predicates or 
                    atom.predicate.name in key_predicates):
                    important_atoms.append(str(atom))
            
            if important_atoms:
                state_label += "\\n" + "\\n".join(important_atoms)
            
            # Color nodes based on state type
            fillcolor = 'lightblue'  # Intermediate state
            if i == 0:  # Initial state
                fillcolor = 'lightgreen'
            elif i == len(atoms_sequence) - 1:  # Final state
                fillcolor = 'lightpink' if not goal_atoms.issubset(atoms) else 'lightgreen'
            
            # Add node with custom style
            dot.node(
                state_id, 
                state_label,
                fillcolor=fillcolor,
                margin='0.3'
            )
            
            # Add edge if not at end
            if nsrt is not None:
                # Find atom changes
                next_atoms = atoms_sequence[i + 1]
                added_atoms = {a for a in (next_atoms - atoms) 
                             if a.predicate.name in (fluent_predicates | key_predicates)}
                removed_atoms = {a for a in (atoms - next_atoms)
                               if a.predicate.name in (fluent_predicates | key_predicates)}
                
                # Create edge label with operator and atom changes
                edge_label = f"{nsrt.name}"
                if added_atoms:
                    edge_label += "\\n+ " + "\\n+ ".join(str(a) for a in sorted(added_atoms))
                if removed_atoms:
                    edge_label += "\\n- " + "\\n- ".join(str(a) for a in sorted(removed_atoms))
                
                dot.edge(
                    state_id, 
                    str(i+1),
                    label=edge_label,
                    fontsize='8',
                    color='darkblue'
                )
        
        # Save graph in transitions directory
        output_path = os.path.join(self.transitions_dir, task_name)
        dot.render(output_path, format='png', cleanup=True)
        self.console.print(f"\n[bold green]Saved transition graph to: {output_path}.png")
        self.console.print(f"\n[bold yellow]Fluent predicates detected: {sorted(fluent_predicates)}")

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
        """Visualize a state's atoms.
        
        Args:
            state: Set of ground atoms in the state
        """
        # Create state panel
        state_panel = Panel(
            "\n".join(str(atom) for atom in sorted(state, key=str)),
            title="Current State",
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