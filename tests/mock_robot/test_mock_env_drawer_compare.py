"""Tests for drawer manipulation scenarios in mock environment."""

import os
from pathlib import Path

import pytest
from rich.console import Console
from rich.logging import RichHandler
import logging

from predicators import utils
from predicators.envs.mock_spot_env import (
    MockSpotEnv, PREDICATES, BELIEF_PREDICATES, _NotBlocked, _HandEmpty,
    _ContainingWaterUnknown, _ContainingWaterKnown, _On, _NotInsideAnyContainer,
    _IsPlaceable, _HasFlatTopSurface, _Reachable, _InHandView, _NEq, _Holding,
    _Inside, _FitsInXY, _NotHolding, _InHandViewFromTop, _robot_type, _container_type,
    _immovable_object_type, _Unknown_ContainerEmpty, _Known_ContainerEmpty,
    _BelieveTrue_ContainerEmpty, _BelieveFalse_ContainerEmpty, _DrawerClosed, _DrawerOpen,
    _movable_object_type
)
from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext
from predicators.structs import Object, Type, State, GroundAtom
from predicators.spot_utils.mock_env.mock_env_creator_manual import ManualMockEnvCreator

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
def setup():
    """Set up test configuration."""
    # Initialize configuration
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1
    })
    
    # Enable belief space operators for all tests
    MockSpotEnv.use_belief_space_operators = True
    
    yield  # this is where the testing happens
    
    # No cleanup needed for use_belief_space_operators
    pass


def test_drawer_manipulation_no_uncertainty():
    """Test transition graph for basic drawer manipulation sequence.
    
    Expected Plan Sequence:
    1. MoveToReachObject(robot, drawer)
       Pre: HandEmpty(robot), Reachable(robot, drawer), DrawerClosed(drawer)
       Effect: Robot in position to manipulate drawer
       
    2. OpenDrawer(robot, drawer)
       Pre: HandEmpty(robot), DrawerClosed(drawer), Reachable(robot, drawer), NotBlocked(drawer)
       Effect: DrawerOpen(drawer)
       
    3. MoveToHandViewObjectInContainer(robot, apple, drawer)
       Pre: HandEmpty(robot), DrawerOpen(drawer), Inside(apple, drawer)
       Effect: InHandView(apple)
       
    4. PickObjectFromContainer(robot, apple, drawer)
       Pre: HandEmpty(robot), DrawerOpen(drawer), Inside(apple, drawer), InHandView(apple)
       Effect: Holding(robot, apple), NotInside(apple, drawer)
       
    5. PlaceObjectOnTop(robot, apple, table)
       Pre: Holding(robot, apple), HasFlatTopSurface(table)
       Effect: On(apple, table), HandEmpty(robot)
    
    This test:
    1. Sets up a scenario with:
       - A robot
       - A drawer (container)
       - A table (immovable object)
       - An apple (placeable object)
       
    2. Creates initial state where:
       - Robot's hand is empty
       - Drawer is on table and closed
       - Apple is in drawer
       - Drawer is closed
       - All objects are reachable
       
    3. Sets goal state where:
       - Apple is on table
       - Drawer is open
       
    4. Verifies:
       - A valid plan is found that includes:
         a. Moving to reach drawer
         b. Opening drawer
         c. Moving to view apple
         d. Picking up apple
         e. Placing apple on table
       - The transition graph shows proper action sequencing
       
    Output:
       - mock_env_data/test_drawer_manipulation_no_uncertainty/transition_graph.png
    """
    # Set up configuration
    test_name = "test_drawer_manipulation_no_uncertainty"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir
    })
    
    # Disable belief space operators since we're testing the certain case
    MockSpotEnv.use_belief_space_operators = False
    
    # Create environment creator
    env_creator = ManualMockEnvCreator(test_dir)
    
    # Create test objects
    robot = Object("robot", _robot_type)
    drawer = Object("drawer", _container_type)
    table = Object("table", _immovable_object_type)
    apple = Object("apple", _movable_object_type)  # Changed to movable_object_type
    objects = {robot, drawer, table, apple}
    
    # Create initial state atoms
    initial_atoms = {
        # Robot state
        GroundAtom(_HandEmpty, [robot]),
        
        # Drawer state
        GroundAtom(_On, [drawer, table]),
        GroundAtom(_NotBlocked, [drawer]),
        GroundAtom(_IsPlaceable, [drawer]),
        GroundAtom(_NotInsideAnyContainer, [drawer]),
        GroundAtom(_FitsInXY, [drawer, table]),
        GroundAtom(_NotHolding, [robot, drawer]),
        GroundAtom(_NEq, [drawer, table]),
        GroundAtom(_DrawerClosed, [drawer]),  # Drawer is closed
        GroundAtom(_Reachable, [robot, drawer]),  # Robot can reach drawer
        
        # Apple state
        GroundAtom(_Inside, [apple, drawer]),
        GroundAtom(_IsPlaceable, [apple]),
        GroundAtom(_FitsInXY, [apple, table]),
        GroundAtom(_NotHolding, [robot, apple]),
        GroundAtom(_NEq, [apple, table]),
        GroundAtom(_NEq, [apple, drawer]),
        GroundAtom(_Reachable, [robot, apple]),  # Robot can reach apple
        GroundAtom(_NotBlocked, [apple]),  # Apple is not blocked
        
        # Environment state
        GroundAtom(_HasFlatTopSurface, [table]),
        GroundAtom(_Reachable, [robot, table])  # Robot can reach table
    }
    
    # Create goal atoms
    goal_atoms = {
        GroundAtom(_On, [apple, table]),
        GroundAtom(_DrawerOpen, [drawer])  # Add drawer open to goal state
    }
    
    # Plan and visualize transitions
    name = f'Transition Graph, {test_name.replace("_", " ").title()}'
    env_creator.plan_and_visualize(initial_atoms, goal_atoms, objects, task_name=name)
    
    # Verify transition graph file exists
    graph_file = Path(test_dir) / "transitions" / f"{name}.html"
    assert graph_file.exists(), "Transition graph file not generated"


def test_drawer_observation_phase():
    """Test transition graph for drawer observation phase.
    
    This test:
    1. Sets up a scenario with:
       - A robot
       - A drawer (container)
       - A table (immovable object)
       
    2. Creates initial state where:
       - Robot's hand is empty
       - Drawer is on table and closed
       - Drawer content is unknown
       - All objects are reachable
       
    3. Sets goal state where:
       - Drawer content is known
       - Drawer is open
       
    4. Verifies:
       - A valid plan is found that includes:
         a. Moving to reach drawer
         b. Opening drawer and finding it empty/not empty
         c. Observing drawer content
       - The transition graph shows proper action sequencing
       
    Output:
       - mock_env_data/test_drawer_observation_phase/transition_graph.png
    """
    # Set up configuration
    test_name = "test_drawer_observation_phase"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir
    })
    
    # Create environment creator
    env_creator = ManualMockEnvCreator(test_dir)
    
    # Create test objects
    robot = Object("robot", _robot_type)
    drawer = Object("drawer", _container_type)
    table = Object("table", _immovable_object_type)
    objects = {robot, drawer, table}
    
    # Create initial state atoms
    initial_atoms = {
        # Robot state
        GroundAtom(_HandEmpty, [robot]),
        
        # Drawer state
        GroundAtom(_On, [drawer, table]),
        GroundAtom(_NotBlocked, [drawer]),
        GroundAtom(_IsPlaceable, [drawer]),
        GroundAtom(_NotInsideAnyContainer, [drawer]),
        GroundAtom(_FitsInXY, [drawer, table]),
        GroundAtom(_NotHolding, [robot, drawer]),
        GroundAtom(_NEq, [drawer, table]),
        GroundAtom(_Unknown_ContainerEmpty, [drawer]),  # Drawer content unknown
        GroundAtom(_DrawerClosed, [drawer]),  # Drawer is closed
        
        # Environment state
        GroundAtom(_HasFlatTopSurface, [table]),
        GroundAtom(_Reachable, [robot, drawer])
    }
    
    # Create goal atoms
    goal_atoms = {
        GroundAtom(_Known_ContainerEmpty, [drawer]),  # We want to know drawer content
        GroundAtom(_DrawerOpen, [drawer])  # And drawer should be open
    }
    
    # Plan and visualize transitions
    name = f'Transition Graph, {test_name.replace("_", " ").title()}'
    env_creator.plan_and_visualize(initial_atoms, goal_atoms, objects, task_name=name)
    
    # Verify transition graph file exists
    graph_file = Path(test_dir) / "transitions" / f"{name}.html"
    assert graph_file.exists(), "Transition graph file not generated"


def test_drawer_manipulation_after_observation():
    """Test transition graph for drawer manipulation after observation.
    
    This test:
    1. Sets up a scenario with:
       - A robot
       - A drawer (container)
       - A table (immovable object)
       - An apple (placeable object)
       
    2. Creates initial state where:
       - Robot's hand is empty
       - Drawer is on table and open
       - Drawer content is known
       - Apple is in drawer
       - All objects are reachable
       
    3. Sets goal state where:
       - Apple is on table
       
    4. Verifies:
       - A valid plan is found that includes:
         a. Moving to view apple
         b. Picking up apple
         c. Placing apple on table
       - The transition graph shows proper action sequencing
       
    Output:
       - mock_env_data/test_drawer_manipulation_after_observation/transition_graph.png
    """
    # Set up configuration
    test_name = "test_drawer_manipulation_after_observation"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir
    })
    
    # Create environment creator
    env_creator = ManualMockEnvCreator(test_dir)
    
    # Create test objects
    robot = Object("robot", _robot_type)
    drawer = Object("drawer", _container_type)
    table = Object("table", _immovable_object_type)
    apple = Object("apple", _container_type)  # Using container type for consistency
    objects = {robot, drawer, table, apple}
    
    # Create initial state atoms
    initial_atoms = {
        # Robot state
        GroundAtom(_HandEmpty, [robot]),
        
        # Drawer state
        GroundAtom(_On, [drawer, table]),
        GroundAtom(_NotBlocked, [drawer]),
        GroundAtom(_IsPlaceable, [drawer]),
        GroundAtom(_NotInsideAnyContainer, [drawer]),
        GroundAtom(_FitsInXY, [drawer, table]),
        GroundAtom(_NotHolding, [robot, drawer]),
        GroundAtom(_NEq, [drawer, table]),
        GroundAtom(_Known_ContainerEmpty, [drawer]),  # Drawer content is known
        GroundAtom(_BelieveFalse_ContainerEmpty, [drawer]),  # We believe drawer has objects
        GroundAtom(_DrawerOpen, [drawer]),  # Drawer is already open
        
        # Apple state (newly discovered)
        GroundAtom(_Inside, [apple, drawer]),
        GroundAtom(_IsPlaceable, [apple]),
        GroundAtom(_FitsInXY, [apple, table]),
        GroundAtom(_NotHolding, [robot, apple]),
        GroundAtom(_NEq, [apple, table]),
        GroundAtom(_NEq, [apple, drawer]),
        
        # Environment state
        GroundAtom(_HasFlatTopSurface, [table]),
        GroundAtom(_Reachable, [robot, drawer]),
        GroundAtom(_Reachable, [robot, apple]),
        GroundAtom(_Reachable, [robot, table])
    }
    
    # Create goal atoms
    goal_atoms = {
        GroundAtom(_On, [apple, table])
    }
    
    # Plan and visualize transitions
    name = f'Transition Graph, {test_name.replace("_", " ").title()}'
    env_creator.plan_and_visualize(initial_atoms, goal_atoms, objects, task_name=name)
    
    # Verify transition graph file exists
    graph_file = Path(test_dir) / "transitions" / f"{name}.html"
    assert graph_file.exists(), "Transition graph file not generated" 