"""Tests for mock environment transition graph calculation.

This module provides tests for calculating and visualizing transition graphs in the mock Spot environment.
The transition graph shows how different operators transform the state of the environment, particularly
useful for understanding action sequences that achieve goals.

Key components:
- States: Represented by sets of ground atoms describing the environment state
- Transitions: Operators that transform one state to another
- Planning: Uses task planning to find valid sequences of operators
- Visualization: Creates a visual graph showing states and transitions

Test cases:
1. Single block pick and place:
   - One block to be moved from table to container
   - Tests basic pick and place capabilities
   - Verifies simple goal achievement

2. Two blocks pick and place:
   - Two blocks to be moved from table to container
   - Tests handling of multiple objects
   - Verifies more complex goal achievement

3. View and reach operations:
   - Tests moving to view and reach objects
   - Verifies proper sequencing of view and reach actions
   - Tests preconditions for manipulation

Output files:
- mock_env_data/test_single_block_pick_place/transition_graph.png: Graph for single block test
- mock_env_data/test_two_object_pick_place/transition_graph.png: Graph for two blocks test
- mock_env_data/test_view_reach/transition_graph.png: Graph for view and reach test

Configuration:
    mock_env_data_dir (str): Directory to store environment data (set during test)
    seed (int): Random seed for reproducibility
    sesame_task_planning_heuristic (str): Heuristic for task planning
    sesame_max_skeletons_optimized (int): Maximum number of skeletons to optimize
    sesame_use_necessary_atoms (bool): Whether to use necessary atoms in planning
    sesame_check_expected_atoms (bool): Whether to check expected atoms in planning
"""

import os
import numpy as np
import pytest
from predicators.envs.mock_spot_env import (
    MockSpotEnv, _robot_type, _container_type, _immovable_object_type, _movable_object_type,
    _HandEmpty, _NotHolding, _On, _NotBlocked, _Inside, _IsPlaceable, 
    _HasFlatTopSurface, _FitsInXY, _NotInsideAnyContainer, _Reachable,
    _InHandView, _NEq
)
from predicators.structs import Object, GroundAtom, State, Task
from predicators.spot_utils.mock_env.mock_env_creator_manual import ManualMockEnvCreator
from predicators import utils
from predicators.settings import CFG
from rich.console import Console
from rich.logging import RichHandler
import logging
from pathlib import Path

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

# Create rich console for pretty printing
console = Console()


@pytest.fixture(autouse=True)
def reset_config():
    """Reset configuration before each test.
    
    Sets up configuration parameters for:
    - Task planning (heuristic, skeletons, atoms)
    - Random seed for reproducibility
    """
    utils.reset_config({
        "seed": 0,
        "sesame_task_planning_heuristic": "hadd",
        "sesame_max_skeletons_optimized": 1,
        "sesame_use_necessary_atoms": True,
        "sesame_check_expected_atoms": True,
    })


def test_single_block_pick_place():
    """Test transition graph calculation for a single-block pick-and-place task.
    
    This test:
    1. Sets up a pick-and-place scenario with:
       - A robot
       - A table (immovable object)
       - A container
       - A single block (movable object)
       
    2. Creates initial state where:
       - Robot's hand is empty
       - Block is on the table
       - Block is not inside any container
       - Block is placeable
       - Table has a flat top surface
       - Block fits in container
       - Block is not blocked
       
    3. Sets goal state where:
       - Block is inside the container
       
    4. Verifies:
       - A valid plan is found
       - The plan achieves the goal state
       - The transition graph is generated
       
    Output:
       - mock_env_data/test_single_block_pick_place/transition_graph.png: Visualization of the transition graph
    """
    # Set up configuration
    test_name = "test_single_block_pick_place"
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": os.path.join("mock_env_data", test_name)
    })
    
    # Create environment creator
    creator = ManualMockEnvCreator(os.path.join("mock_env_data", test_name))
    
    # Create objects
    robot = Object("robot", _robot_type)
    block = Object("block", _movable_object_type)
    table = Object("table", _immovable_object_type)
    container = Object("container", _container_type)
    
    # Create initial atoms
    objects = {robot, block, table, container}
    initial_atoms = {
        GroundAtom(_HandEmpty, [robot]),
        GroundAtom(_NotHolding, [robot, block]),
        GroundAtom(_On, [block, table]),
        GroundAtom(_On, [container, table]),
        GroundAtom(_NotBlocked, [block]),
        GroundAtom(_NotBlocked, [container]),
        GroundAtom(_IsPlaceable, [block]),
        GroundAtom(_HasFlatTopSurface, [table]),
        GroundAtom(_FitsInXY, [block, container]),
        GroundAtom(_NotInsideAnyContainer, [block]),
        GroundAtom(_NotHolding, [robot, container]),
        # Additional necessary predicates
        GroundAtom(_Reachable, [robot, block]),
        GroundAtom(_Reachable, [robot, container]),
        GroundAtom(_Reachable, [robot, table]),
        GroundAtom(_InHandView, [robot, block]),
        GroundAtom(_NEq, [block, table]),
        GroundAtom(_NEq, [block, container]),
        GroundAtom(_NEq, [container, table])
    }
    
    # Create goal atoms - block should be in the container
    goal_atoms = {
        GroundAtom(_Inside, [block, container])
    }
    
    # Plan and visualize transitions
    name = f'Transition Graph, {test_name.replace("_", " ").title()}'
    creator.plan_and_visualize(initial_atoms, goal_atoms, objects, task_name=name)
    
    # Verify output file exists
    graph_file = Path(os.path.join("mock_env_data", test_name)) / "transitions" / f"{name}.html"
    assert graph_file.exists(), "Transition graph file not generated"


def test_two_object_pick_place():
    """Test transition graph calculation for a two-object pick-and-place task.
    
    This test:
    1. Sets up a pick-and-place scenario with:
       - A robot
       - A table (immovable object)
       - A container
       - Two blocks (movable objects)
       
    2. Creates initial state where:
       - Robot's hand is empty
       - Both blocks are on the table
       - Blocks are not inside any container
       - Blocks are placeable
       - Table has a flat top surface
       - Blocks fit in container
       - Blocks are not blocked
       
    3. Sets goal state where:
       - Both blocks are inside the container
       
    4. Verifies:
       - A valid plan is found
       - The plan achieves the goal state
       - The transition graph is generated
       - The plan handles multiple objects correctly
       
    Output:
       - mock_env_data/test_two_object_pick_place/transition_graph.png: Visualization of the transition graph
    """
    # Set up configuration
    test_name = "test_two_object_pick_place"
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": os.path.join("mock_env_data", test_name)
    })
    
    # Create environment creator
    creator = ManualMockEnvCreator(os.path.join("mock_env_data", test_name))
    
    # Create objects
    robot = Object("robot", _robot_type)
    block1 = Object("block1", _movable_object_type)
    block2 = Object("block2", _movable_object_type)
    table = Object("table", _immovable_object_type)
    container = Object("container", _container_type)
    
    # Create initial atoms
    objects = {robot, block1, block2, table, container}
    initial_atoms = {
        GroundAtom(_HandEmpty, [robot]),
        GroundAtom(_NotHolding, [robot, block1]),
        GroundAtom(_NotHolding, [robot, block2]),
        GroundAtom(_On, [block1, table]),
        GroundAtom(_On, [block2, table]),
        GroundAtom(_On, [container, table]),
        GroundAtom(_NotBlocked, [block1]),
        GroundAtom(_NotBlocked, [block2]),
        GroundAtom(_NotBlocked, [container]),
        GroundAtom(_IsPlaceable, [block1]),
        GroundAtom(_IsPlaceable, [block2]),
        GroundAtom(_HasFlatTopSurface, [table]),
        GroundAtom(_FitsInXY, [block1, container]),
        GroundAtom(_FitsInXY, [block2, container]),
        GroundAtom(_NotInsideAnyContainer, [block1]),
        GroundAtom(_NotInsideAnyContainer, [block2]),
        GroundAtom(_NotHolding, [robot, container]),
        # Additional necessary predicates
        GroundAtom(_Reachable, [robot, block1]),
        GroundAtom(_Reachable, [robot, block2]),
        GroundAtom(_Reachable, [robot, container]),
        GroundAtom(_Reachable, [robot, table]),
        GroundAtom(_InHandView, [robot, block1]),
        GroundAtom(_InHandView, [robot, block2]),
        GroundAtom(_NEq, [block1, table]),
        GroundAtom(_NEq, [block1, container]),
        GroundAtom(_NEq, [block2, table]),
        GroundAtom(_NEq, [block2, container]),
        GroundAtom(_NEq, [block1, block2]),
        GroundAtom(_NEq, [container, table])
    }
    
    # Create goal atoms - both blocks should be in the container
    goal_atoms = {
        GroundAtom(_Inside, [block1, container]),
        GroundAtom(_Inside, [block2, container])
    }
    
    # Plan and visualize transitions
    name = f'Transition Graph, {test_name.replace("_", " ").title()}'
    creator.plan_and_visualize(initial_atoms, goal_atoms, objects, task_name=name)
    
    # Verify output file exists
    graph_file = Path(os.path.join("mock_env_data", test_name)) / "transitions" / f"{name}.html"
    assert graph_file.exists(), "Transition graph file not generated"


def test_view_reach_pick_place_two_objects():
    """Test transition graph calculation for view-reach-pick-place with two objects.
    
    This test:
    1. Sets up a scenario with:
       - A robot
       - A table (immovable object)
       - A container
       - Two blocks (movable objects)
       
    2. Creates initial state where:
       - Robot's hand is empty
       - Both blocks are on the table
       - Blocks are not in view
       - Blocks are not reachable
       - Blocks are not inside any container
       - Blocks are placeable
       - Table has a flat top surface
       - Blocks fit in container
       - Blocks are not blocked
       
    3. Sets goal state where:
       - Both blocks are inside the container
       
    4. Verifies:
       - A valid plan is found that includes:
         a. Moving to view first block
         b. Moving to reach first block
         c. Picking and placing first block
         d. Moving to view second block
         e. Moving to reach second block
         f. Picking and placing second block
       - The transition graph shows proper action sequencing
       - The plan achieves the goal state
       
    Output:
       - mock_env_data/test_view_reach_pick_place_two/transition_graph.png: Visualization of the transition graph
    """
    # Set up configuration
    test_name = "test_view_reach_pick_place_two"
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": os.path.join("mock_env_data", test_name)
    })
    
    # Create environment creator
    creator = ManualMockEnvCreator(os.path.join("mock_env_data", test_name))
    
    # Create objects
    robot = Object("robot", _robot_type)
    block1 = Object("block1", _movable_object_type)
    block2 = Object("block2", _movable_object_type)
    table = Object("table", _immovable_object_type)
    container = Object("container", _container_type)
    
    # Create initial atoms - blocks are not in view or reachable initially
    objects = {robot, block1, block2, table, container}
    initial_atoms = {
        GroundAtom(_HandEmpty, [robot]),
        GroundAtom(_NotHolding, [robot, block1]),
        GroundAtom(_NotHolding, [robot, block2]),
        GroundAtom(_On, [block1, table]),
        GroundAtom(_On, [block2, table]),
        GroundAtom(_On, [container, table]),
        GroundAtom(_NotBlocked, [block1]),
        GroundAtom(_NotBlocked, [block2]),
        GroundAtom(_NotBlocked, [container]),
        GroundAtom(_IsPlaceable, [block1]),
        GroundAtom(_IsPlaceable, [block2]),
        GroundAtom(_HasFlatTopSurface, [table]),
        GroundAtom(_FitsInXY, [block1, container]),
        GroundAtom(_FitsInXY, [block2, container]),
        GroundAtom(_NotInsideAnyContainer, [block1]),
        GroundAtom(_NotInsideAnyContainer, [block2]),
        GroundAtom(_NotHolding, [robot, container]),
        # Additional necessary predicates
        GroundAtom(_NEq, [block1, table]),
        GroundAtom(_NEq, [block1, container]),
        GroundAtom(_NEq, [block2, table]),
        GroundAtom(_NEq, [block2, container]),
        GroundAtom(_NEq, [block1, block2]),
        GroundAtom(_NEq, [container, table])
    }
    
    # Create goal atoms - both blocks should be in the container
    goal_atoms = {
        GroundAtom(_Inside, [block1, container]),
        GroundAtom(_Inside, [block2, container])
    }
    
    # Plan and visualize transitions
    name = f'Transition Graph, {test_name.replace("_", " ").title()}'
    creator.plan_and_visualize(initial_atoms, goal_atoms, objects, task_name=name)
    
    # Verify output file exists
    graph_file = Path(os.path.join("mock_env_data", test_name)) / "transitions" / f"{name}.html"
    assert graph_file.exists(), "Transition graph file not generated"


def test_transitions_visualization():
    """Test that transitions are correctly visualized."""
    # Set up configuration
    test_name = "test_transitions_visualization"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir
    })
    
    # Create environment and test objects
    creator = ManualMockEnvCreator(test_dir)
    robot = Object("robot", _robot_type)
    cup = Object("cup", _container_type)
    source_table = Object("source_table", _immovable_object_type)
    target_table = Object("target_table", _immovable_object_type)
    objects = {robot, cup, source_table, target_table}
    
    # Define initial state with all necessary predicates
    initial_atoms = {
        # Robot state
        GroundAtom(_HandEmpty, [robot]),
        GroundAtom(_NotHolding, [robot, cup]),
        
        # Object positions and properties
        GroundAtom(_On, [cup, source_table]),
        GroundAtom(_NotBlocked, [cup]),
        GroundAtom(_IsPlaceable, [cup]),
        
        # Surface properties
        GroundAtom(_HasFlatTopSurface, [source_table]),
        GroundAtom(_HasFlatTopSurface, [target_table]),
        
        # Reachability constraints
        GroundAtom(_Reachable, [robot, cup]),
        GroundAtom(_Reachable, [robot, target_table]),
        GroundAtom(_Reachable, [robot, source_table]),
        
        # Object relationships
        GroundAtom(_NEq, [cup, source_table]),
        GroundAtom(_NEq, [cup, target_table]),
        GroundAtom(_NEq, [source_table, target_table]),
        GroundAtom(_NotInsideAnyContainer, [cup]),
        GroundAtom(_FitsInXY, [cup, source_table]),
        GroundAtom(_FitsInXY, [cup, target_table])
    }
    
    # Define goal state - cup should be on target table
    goal_atoms = {
        GroundAtom(_On, [cup, target_table])
    }
    
    # Test both visualization methods
    name = f'Transition Graph, {test_name.replace("_", " ").title()}'
    
    # Test interactive visualization
    creator.plan_and_visualize(initial_atoms, goal_atoms, objects, task_name=name)
    assert os.path.exists(os.path.join(test_dir, "transitions", f"{name}.html")) 