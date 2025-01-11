"""Base class for mock environment creators.

This module provides a base class for creating mock Spot environments with:
- Predefined states and transitions
- RGB-D observations with object detections
- Task-specific configurations
- Planning and visualization utilities

The environment data is stored in a directory specified by CFG.mock_env_data_dir.
This includes:
- graph.json: Contains state transitions and observations
- images/: Directory containing RGB-D images for each state
- transitions/: Directory containing transition graph visualizations
- transitions/transitions.png: Main transition graph visualization
- transitions/single_block.png: Single block task transition graph
- transitions/two_blocks.png: Two blocks task transition graph
- transitions/cup_emptiness.png: Cup emptiness belief task transition graph

Configuration:
    mock_env_data_dir (str): Directory to store environment data (set during initialization)
    seed (int): Random seed for reproducibility
    sesame_task_planning_heuristic (str): Heuristic for task planning
    sesame_max_skeletons_optimized (int): Maximum number of skeletons to optimize
    sesame_use_necessary_atoms (bool): Whether to use necessary atoms in planning
    sesame_check_expected_atoms (bool): Whether to check expected atoms in planning

Example usage:
    ```python
    # Create environment creator
    creator = ManualMockEnvCreator("path/to/data_dir")
    
    # Create objects and predicates
    robot = Object("robot", creator.types["robot"])
    block = Object("block", creator.types["movable_object"])
    
    # Create initial and goal atoms
    init_atoms = {
        GroundAtom(creator.predicates["HandEmpty"], [robot]),
        GroundAtom(creator.predicates["On"], [block, table])
    }
    goal_atoms = {
        GroundAtom(creator.predicates["Inside"], [block, container])
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

from predicators.envs.mock_spot_env import MockSpotEnv
from predicators.ground_truth_models.mock_spot_env.nsrts import MockSpotGroundTruthNSRTFactory
from predicators.structs import (
    GroundAtom, EnvironmentTask, State, Task, Type, Predicate, 
    ParameterizedOption, NSRT, Object, Variable, LiftedAtom, STRIPSOperator,
    Action
)
from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext
from predicators.ground_truth_models import get_gt_options
from predicators import utils
from predicators.settings import CFG
from predicators.planning import task_plan_grounding, task_plan

# Keep the mock action info class but comment it out for now
# class MockActionExtraInfo:
#     """Mock action info for mock environment."""
#     def __init__(self, name: str, objects: Sequence[Object], fn: Any, fn_args: Tuple) -> None:
#         self.action_name = name
#         self.operator_objects = objects
#         self.real_world_fn = fn
#         self.real_world_fn_args = fn_args

class MockEnvCreatorBase(ABC):
    """Base class for mock environment creators.
    
    This class provides functionality to:
    - Create and configure mock Spot environments
    - Add states with RGB-D observations
    - Define transitions between states
    - Generate task-specific state sequences
    - Plan and visualize transitions
    
    The environment data is stored in a directory specified by CFG.mock_env_data_dir,
    which is set during initialization. This includes:
    - State transition graph (graph.json)
    - RGB-D images for each state (images/)
    - Observation metadata (gripper state, objects in view/hand)
    - Transition graph visualization (transitions/)
    
    Attributes:
        path_dir (str): Base directory for environment data
        image_dir (str): Directory for RGB-D images
        transitions_dir (str): Directory for transition graph visualizations
        env (MockSpotEnv): Mock Spot environment instance
        types (Dict[str, Type]): Available object types
        predicates (Dict[str, Predicate]): Available predicates
        options (Dict[str, ParameterizedOption]): Available options
        nsrts (Set[NSRT]): Available NSRTs
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
        
        # Set data directory in config for environment to use
        utils.reset_config({
            "mock_env_data_dir": path_dir
        })
        
        # Initialize environment (will use CFG.mock_env_data_dir)
        self.env = MockSpotEnv()
        
        # Get environment info
        self.types: Dict[str, Type] = {t.name: t for t in self.env.types}
        self.predicates: Dict[str, Predicate] = {p.name: p for p in self.env.predicates}
        
        # Add observation predicates
        self._add_observation_predicates()
        
        # Create options
        self.options: Dict[str, ParameterizedOption] = {o.name: o for o in get_gt_options(self.env.get_name())}
        
        # Add observation options
        self._add_observation_options()
        
        # Get NSRTs from factory
        self.nsrts: Set[NSRT] = self._create_nsrts()

        # Initialize rich console for pretty printing
        self.console = Console()

    def _add_observation_predicates(self) -> None:
        """Add predicates for cup observation and belief state."""
        # Get types
        robot_type = self.types["robot"]
        container_type = self.types["container"]
        movable_type = self.types["movable_object"]
        
        # Create observation predicates
        self.predicates.update({
            # Keep these predicates but comment out for now
            # "ContainingWaterUnknown": Predicate("ContainingWaterUnknown", [container_type], lambda s, o: True),
            # "ContainingWaterKnown": Predicate("ContainingWaterKnown", [container_type], lambda s, o: True),
            # "ContainingWater": Predicate("ContainingWater", [container_type], lambda s, o: True),
            # "NotContainingWater": Predicate("NotContainingWater", [container_type], lambda s, o: True),
            # "InHandViewFromTop": Predicate("InHandViewFromTop", [robot_type, movable_type], lambda s, o: True)
        })

    def _add_observation_options(self) -> None:
        """Add options for cup observation."""
        # TODO: Implement observation options after fixing base pick-place functionality
        pass

    def _create_nsrts(self) -> Set[NSRT]:
        """Create NSRTs including observation operators."""
        # Get base NSRTs from factory
        factory = MockSpotGroundTruthNSRTFactory()
        base_nsrts = factory.get_nsrts(
            self.env.get_name(),
            self.types,
            self.predicates,
            self.options
        )
        
        # TODO: Add observation NSRTs after fixing base pick-place functionality
        # observation_nsrts = self._create_observation_nsrts()
        # return base_nsrts | observation_nsrts
        
        return base_nsrts

    def _create_observation_nsrts(self) -> Set[NSRT]:
        """Create NSRTs for cup observation."""
        # TODO: Implement observation NSRTs after fixing base pick-place functionality
        return set()

    def plan_and_visualize(self, init_atoms: Set[GroundAtom], goal_atoms: Set[GroundAtom], 
                          objects: Set[Object], output_file: str = "transitions",
                          timeout: float = 10.0) -> None:
        """Plan a sequence of actions and visualize the transition graph.
        
        This method:
        1. Displays available options and NSRTs
        2. Shows initial state and goal atoms
        3. Grounds NSRTs and shows reachable atoms
        4. Creates a plan to achieve the goal
        5. Visualizes the plan steps and transitions
        6. Saves the transition graph
        
        Args:
            init_atoms: Initial state atoms
            goal_atoms: Goal state atoms
            objects: Objects in the environment
            output_file: Name of output file (without extension)
            timeout: Planning timeout in seconds
            
        Raises:
            StopIteration: If no valid plan is found
            AssertionError: If the plan does not achieve the goal
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

        # Ground NSRTs and get reachable atoms
        ground_nsrts, reachable_atoms = task_plan_grounding(
            init_atoms=init_atoms,
            objects=objects,
            nsrts=self.nsrts,
            allow_noops=False
        )
        self.console.print(f"\n[bold cyan]Grounded {len(ground_nsrts)} NSRTs")
        
        # Create grounded NSRTs tree
        ground_nsrts_tree = Tree("[bold blue]Grounded NSRTs")
        for i, nsrt in enumerate(ground_nsrts, 1):
            nsrt_node = ground_nsrts_tree.add(f"[cyan]{i}. {nsrt.name}")
            nsrt_node.add(f"[green]Objects: {[obj.name for obj in nsrt.objects]}")
            nsrt_node.add(f"[yellow]Preconditions: {nsrt.preconditions}")
            nsrt_node.add(f"[red]Add effects: {nsrt.add_effects}")
            nsrt_node.add(f"[magenta]Delete effects: {nsrt.delete_effects}")
        self.console.print(ground_nsrts_tree)
        
        # Create reachable atoms panel
        reachable_atoms_panel = Panel(
            "\n".join(str(atom) for atom in sorted(reachable_atoms, key=str)),
            title="Reachable Atoms",
            border_style="yellow"
        )
        self.console.print(reachable_atoms_panel)
        
        # Create heuristic for planning
        heuristic = utils.create_task_planning_heuristic(
            CFG.sesame_task_planning_heuristic,
            init_atoms,
            goal_atoms,
            ground_nsrts,
            self.predicates.values(),
            objects
        )
        self.console.print(f"\n[bold green]Created heuristic: {CFG.sesame_task_planning_heuristic}")
        
        # Create goal atoms panel
        goal_atoms_panel = Panel(
            "\n".join(str(atom) for atom in goal_atoms),
            title="Goal Atoms",
            border_style="red"
        )
        self.console.print(goal_atoms_panel)
        
        # Generate plan using task planning
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
        
        # Get first valid plan
        plan, atoms_sequence, metrics = next(plan_gen)
        self.console.print("\n[bold blue]Found plan:")
        
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
        
        # Create transition graph
        self.visualize_transitions(atoms_sequence, plan, goal_atoms, output_file)
        
        # Verify plan achieves goal
        assert goal_atoms.issubset(atoms_sequence[-1])

    def visualize_transitions(self, atoms_sequence: List[Set[GroundAtom]], 
                            plan: List[Any], goal_atoms: Set[GroundAtom],
                            output_file: str = "transitions") -> None:
        """Visualize the transition graph.
        
        Creates a graphviz visualization showing:
        - States as nodes with their ground atoms
        - Transitions as edges with operator names
        - Objects and their states grouped together
        - Color coding for initial, intermediate, and goal states
        
        Args:
            atoms_sequence: Sequence of atom sets representing states
            plan: Sequence of operators
            goal_atoms: Goal state atoms for coloring final state
            output_file: Name of output file (without extension)
            
        The graph is saved in the transitions directory with format:
            {transitions_dir}/{output_file}.png
        """
        # Create a new directed graph
        dot = graphviz.Digraph(comment='Transition Graph for Pick and Place Task')
        dot.attr(rankdir='LR')  # Left to right layout
        dot.attr('node', shape='box', style='rounded,filled', fontsize='10')
        
        # Add nodes and edges
        for i, (atoms, nsrt) in enumerate(zip(atoms_sequence, plan + [None])):
            # Create node label with state atoms
            state_id = str(i)
            state_label = f"State {i}\\n"
            
            # Group atoms by object for better readability
            key_atoms_by_obj = {}
            for atom in sorted(atoms, key=str):
                atom_str = str(atom)
                if any(p in atom_str for p in ['Holding', 'Inside', 'On']):
                    obj_name = atom.objects[0].name
                    if obj_name not in key_atoms_by_obj:
                        key_atoms_by_obj[obj_name] = []
                    key_atoms_by_obj[obj_name].append(atom_str)
            
            # Add atoms grouped by object
            for obj_name, obj_atoms in sorted(key_atoms_by_obj.items()):
                state_label += f"\\n{obj_name}:\\n  "
                state_label += "\\n  ".join(obj_atoms)
            
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
                fillcolor=fillcolor
            )
            
            # Add edge if not at end
            if nsrt is not None:
                # Create edge label with operator and objects
                edge_label = f"{nsrt.name}\\n"
                edge_label += "\\n".join(obj.name for obj in nsrt.objects)
                
                dot.edge(
                    state_id, 
                    str(i+1),
                    label=edge_label,
                    fontsize='8',
                    color='darkblue'
                )
        
        # Save graph in transitions directory
        output_path = os.path.join(self.transitions_dir, output_file)
        dot.render(output_path, format='png', cleanup=True)
        self.console.print(f"\n[bold green]Saved transition graph to: {output_path}.png")

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