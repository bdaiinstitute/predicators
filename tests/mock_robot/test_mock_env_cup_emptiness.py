"""Tests for cup emptiness scenarios in mock environment."""

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
    _immovable_object_type
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
    yield  # this is where the testing happens
    # Cleanup (if needed)
    pass


def test_with_belief_observe_cup_emptiness():
    """Test transition graph for cup emptiness observation.
    
    Setup:
    - Single cup on table
    - Initial state: Content unknown
    - Goal state: Content known
    - Actions: Move to view, Observe content
    """
    # Set up configuration
    test_name = "test_with_belief_observe_cup_emptiness"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir  # Set data directory for this test
    })
    
    # Enable belief-space operators
    MockSpotEnv.use_belief_space_operators = True
    
    # Create environment creator
    env_creator = ManualMockEnvCreator(test_dir)
    
    # Create test objects
    robot = Object("robot", _robot_type)
    cup = Object("cup", _container_type)
    table = Object("table", _immovable_object_type)
    objects = {robot, cup, table}
    
    # Create initial state atoms
    initial_atoms = {
        GroundAtom(_HandEmpty, [robot]),
        GroundAtom(_On, [cup, table]),
        GroundAtom(_ContainingWaterUnknown, [cup]),
        GroundAtom(_NotBlocked, [cup]),
        GroundAtom(_IsPlaceable, [cup]),
        GroundAtom(_HasFlatTopSurface, [table]),
        GroundAtom(_Reachable, [robot, cup]),
        GroundAtom(_NEq, [cup, table]),
        GroundAtom(_NotInsideAnyContainer, [cup]),
        GroundAtom(_FitsInXY, [cup, table]),
        GroundAtom(_NotHolding, [robot, cup])
    }
    
    # Create goal atoms
    goal_atoms = {
        GroundAtom(_ContainingWaterKnown, [cup])
    }
    
    # Plan and visualize transitions
    env_creator.plan_and_visualize(initial_atoms, goal_atoms, objects, task_name=f'Transition Graph, {test_name.replace("_", " ").title()}')
    
    # Verify transition graph file exists
    graph_file = Path(test_dir) / "transitions" / "transition_graph.png"
    assert graph_file.exists(), "Transition graph file not generated"
    
    # Cleanup
    MockSpotEnv.use_belief_space_operators = False


def test_with_belief_check_and_pick_cup():
    """Test transition graph for checking cup content and then picking it.
    
    This test verifies that belief-space operators can be combined with 
    manipulation operators in a single sequence.
    
    Sequence:
    1. MoveToHandObserveObjectFromTop
    2. ObserveContainerContent
    3. MoveToHandViewObject
    4. PickObjectFromTop
    """
    # Enable belief-space operators
    MockSpotEnv.use_belief_space_operators = True
    env = MockSpotEnv(use_gui=False)

    # Create test objects
    robot = Object("robot", _robot_type)
    cup = Object("cup", _container_type)
    table = Object("table", _immovable_object_type)

    # Create initial state with unknown cup content
    # Initial predicates: NotBlocked(cup), HandEmpty(robot), ContainingWaterUnknown(cup)
    state_id = env.add_state(
        rgbd=None,
        gripper_open=True,
        objects_in_view={cup.name, table.name},
        objects_in_hand=set()
    )

    # State after moving to observe from top
    # Added predicates: InHandViewFromTop(robot, cup)
    observe_state_id = env.add_state(
        rgbd=None,
        gripper_open=True,
        objects_in_view={cup.name, table.name},
        objects_in_hand=set()
    )

    # State after observing content
    # Added: ContainingWaterKnown(cup)
    # Removed: ContainingWaterUnknown(cup)
    known_state_id = env.add_state(
        rgbd=None,
        gripper_open=True,
        objects_in_view={cup.name, table.name},
        objects_in_hand=set()
    )

    # State after moving to hand view for picking
    # Added: InHandView(robot, cup)
    view_state_id = env.add_state(
        rgbd=None,
        gripper_open=True,
        objects_in_view={cup.name, table.name},
        objects_in_hand=set()
    )

    # Final state after picking
    # Added: Holding(robot, cup)
    # Removed: On(cup, table), HandEmpty(robot), InHandView(robot, cup)
    holding_state_id = env.add_state(
        rgbd=None,
        gripper_open=False,  # Gripper closed when holding
        objects_in_view={cup.name, table.name},
        objects_in_hand={cup.name}  # Cup now in hand
    )

    # Add transitions for the complete sequence
    env.add_transition(state_id, "MoveToHandObserveObjectFromTop", observe_state_id)
    env.add_transition(observe_state_id, "ObserveContainerContent", known_state_id)
    env.add_transition(known_state_id, "MoveToHandViewObject", view_state_id)
    env.add_transition(view_state_id, "PickObjectFromTop", holding_state_id)

    # Save graph data
    test_dir = Path("mock_env_data") / "test_with_belief_check_and_pick_cup"
    os.makedirs(test_dir / "transitions", exist_ok=True)
    env._save_graph_data()

    # Verify all transitions in sequence
    assert env._transitions[state_id]["MoveToHandObserveObjectFromTop"] == observe_state_id
    assert env._transitions[observe_state_id]["ObserveContainerContent"] == known_state_id
    assert env._transitions[known_state_id]["MoveToHandViewObject"] == view_state_id
    assert env._transitions[view_state_id]["PickObjectFromTop"] == holding_state_id

    # Verify final state has cup in hand
    final_obs = env._observations[holding_state_id]
    assert cup.name in final_obs.objects_in_hand
    assert not final_obs.gripper_open

    # Cleanup
    MockSpotEnv.use_belief_space_operators = False 


def test_with_belief_plan_check_and_pick_cup():
    """Test transition graph calculation for checking cup content and picking.
    
    This test:
    1. Sets up a scenario with:
       - A robot
       - A table (immovable object)
       - A cup (container)
       
    2. Creates initial state where:
       - Robot's hand is empty
       - Cup is on the table
       - Cup content is unknown
       - Cup is not blocked
       - Cup is placeable
       - Table has a flat top surface
       
    3. Sets goal state where:
       - Cup content is known
       - Cup is being held by robot
       
    4. Verifies:
       - A valid plan is found that includes:
         a. Moving to observe cup from top
         b. Observing cup content
         c. Moving to view cup for picking
         d. Picking up cup
       - The transition graph shows proper action sequencing
       - The plan achieves both belief and physical goals
       
    Output:
       - mock_env_data/test_with_belief_plan_check_and_pick_cup/transition_graph.png
    """
    # Set up configuration
    test_name = "test_with_belief_plan_check_and_pick_cup"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir
    })
    
    # Enable belief-space operators
    MockSpotEnv.use_belief_space_operators = True
    
    # Create environment creator
    env_creator = ManualMockEnvCreator(test_dir)
    
    # Create test objects
    robot = Object("robot", _robot_type)
    cup = Object("cup", _container_type)
    table = Object("table", _immovable_object_type)
    objects = {robot, cup, table}
    
    # Create initial state atoms
    initial_atoms = {
        GroundAtom(_HandEmpty, [robot]),
        GroundAtom(_On, [cup, table]),
        GroundAtom(_ContainingWaterUnknown, [cup]),
        GroundAtom(_NotBlocked, [cup]),
        GroundAtom(_IsPlaceable, [cup]),
        GroundAtom(_HasFlatTopSurface, [table]),
        GroundAtom(_Reachable, [robot, cup]),
        GroundAtom(_NEq, [cup, table]),
        GroundAtom(_NotInsideAnyContainer, [cup]),
        GroundAtom(_FitsInXY, [cup, table]),
        GroundAtom(_NotHolding, [robot, cup])
    }
    
    # Create goal atoms
    goal_atoms = {
        GroundAtom(_ContainingWaterKnown, [cup]),
        GroundAtom(_Holding, [robot, cup])
    }
    
    # Plan and visualize transitions
    name = f'Transition Graph, {test_name.replace("_", " ").title()}'
    env_creator.plan_and_visualize(initial_atoms, goal_atoms, objects, task_name=name)
    
    # Verify transition graph file exists
    graph_file = Path(test_dir) / "transitions" / f"{name}.png"
    assert graph_file.exists(), "Transition graph file not generated"
    
    # Cleanup
    MockSpotEnv.use_belief_space_operators = False


def test_with_belief_plan_check_and_place_cup():
    """Test transition graph calculation for checking cup content and placing in target.
    
    This test:
    1. Sets up a scenario with:
       - A robot
       - A source table (immovable object)
       - A target table (immovable object)
       - A cup (container)
       
    2. Creates initial state where:
       - Robot's hand is empty
       - Cup is on the source table
       - Cup content is unknown
       - Cup is not blocked
       - Cup is placeable
       - Both tables have flat top surfaces
       
    3. Sets goal state where:
       - Cup content is known
       - Cup is on target table
       
    4. Verifies:
       - A valid plan is found that includes:
         a. Moving to observe cup from top
         b. Observing cup content
         c. Moving to view cup for picking
         d. Picking up cup
         e. Moving to target table
         f. Placing cup on target table
       - The transition graph shows proper action sequencing
       - The plan achieves both belief and physical goals
       
    Output:
       - mock_env_data/test_with_belief_plan_check_and_place_cup/transition_graph.png
    """
    # Set up configuration
    test_name = "test_with_belief_plan_check_and_place_cup"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir
    })
    
    # Enable belief-space operators
    MockSpotEnv.use_belief_space_operators = True
    
    # Create environment creator
    env_creator = ManualMockEnvCreator(test_dir)
    
    # Create test objects
    robot = Object("robot", _robot_type)
    cup = Object("cup", _container_type)
    source_table = Object("source_table", _immovable_object_type)
    target_table = Object("target_table", _immovable_object_type)
    objects = {robot, cup, source_table, target_table}
    
    # Create initial state atoms
    initial_atoms = {
        GroundAtom(_HandEmpty, [robot]),
        GroundAtom(_On, [cup, source_table]),
        GroundAtom(_ContainingWaterUnknown, [cup]),
        GroundAtom(_NotBlocked, [cup]),
        GroundAtom(_IsPlaceable, [cup]),
        GroundAtom(_HasFlatTopSurface, [source_table]),
        GroundAtom(_HasFlatTopSurface, [target_table]),
        GroundAtom(_Reachable, [robot, cup]),
        GroundAtom(_NEq, [cup, source_table]),
        GroundAtom(_NEq, [cup, target_table]),
        GroundAtom(_NEq, [source_table, target_table]),
        GroundAtom(_NotInsideAnyContainer, [cup]),
        GroundAtom(_FitsInXY, [cup, source_table]),
        GroundAtom(_FitsInXY, [cup, target_table]),
        GroundAtom(_NotHolding, [robot, cup]),
        GroundAtom(_Reachable, [robot, target_table]),
        GroundAtom(_Reachable, [robot, source_table])
    }
    
    # Create goal atoms
    goal_atoms = {
        GroundAtom(_ContainingWaterKnown, [cup]),
        GroundAtom(_On, [cup, target_table])
    }
    
    # Plan and visualize transitions
    env_creator.plan_and_visualize(initial_atoms, goal_atoms, objects, task_name=f'Transition Graph, {test_name.replace("_", " ").title()}')
    
    # Verify transition graph file exists
    graph_file = Path(test_dir) / "transitions" / "transition_graph.png"
    assert graph_file.exists(), "Transition graph file not generated"
    
    # Cleanup
    MockSpotEnv.use_belief_space_operators = False 


def test_view_pick_and_place_cup():
    """Test transition graph for basic view, pick and place sequence.
    
    This test:
    1. Sets up a scenario with:
       - A robot
       - A source table (immovable object)
       - A target table (immovable object)
       - A cup (container)
       
    2. Creates initial state where:
       - Robot's hand is empty
       - Cup is on the source table
       - Cup is not blocked
       - Cup is placeable
       - Both tables have flat top surfaces
       
    3. Sets goal state where:
       - Cup is on target table
       
    4. Verifies:
       - A valid plan is found that includes:
         a. Moving to view cup
         b. Picking up cup
         c. Placing cup on target table
       - The transition graph shows proper action sequencing
       
    Output:
       - mock_env_data/test_view_pick_and_place_cup/transition_graph.png
    """
    # Set up configuration
    test_name = "test_view_pick_and_place_cup"
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
    cup = Object("cup", _container_type)
    source_table = Object("source_table", _immovable_object_type)
    target_table = Object("target_table", _immovable_object_type)
    objects = {robot, cup, source_table, target_table}
    
    # Create initial state atoms
    initial_atoms = {
        GroundAtom(_HandEmpty, [robot]),
        GroundAtom(_On, [cup, source_table]),
        GroundAtom(_NotBlocked, [cup]),
        GroundAtom(_IsPlaceable, [cup]),
        GroundAtom(_HasFlatTopSurface, [source_table]),
        GroundAtom(_HasFlatTopSurface, [target_table]),
        GroundAtom(_Reachable, [robot, cup]),
        GroundAtom(_NEq, [cup, source_table]),
        GroundAtom(_NEq, [cup, target_table]),
        GroundAtom(_NEq, [source_table, target_table]),
        GroundAtom(_NotInsideAnyContainer, [cup]),
        GroundAtom(_FitsInXY, [cup, source_table]),
        GroundAtom(_FitsInXY, [cup, target_table]),
        GroundAtom(_NotHolding, [robot, cup]),
        GroundAtom(_Reachable, [robot, target_table]),
        GroundAtom(_Reachable, [robot, source_table])
    }
    
    # Create goal atoms
    goal_atoms = {
        GroundAtom(_On, [cup, target_table])
    }
    
    # Plan and visualize transitions
    env_creator.plan_and_visualize(initial_atoms, goal_atoms, objects, task_name=f'Transition Graph, {test_name.replace("_", " ").title()}')
    
    # Verify transition graph file exists
    graph_file = Path(test_dir) / "transitions" / "transition_graph.png"
    assert graph_file.exists(), "Transition graph file not generated" 


def test_observe_two_cups_and_place_empty():
    """Test transition graph for observing two cups and placing empty one in bucket.
    
    This test:
    1. Sets up a scenario with:
       - A robot
       - A table (immovable object)
       - A bucket (container)
       - Two cups:
         * water_cup (container, containing water)
         * empty_cup (container, empty)
       
    2. Creates initial state where:
       - Robot's hand is empty
       - Both cups are on the table
       - Both cups' contents are unknown
       - Both cups are not blocked
       - Both cups are placeable
       - Table has a flat top surface
       - Bucket is reachable
       
    3. Sets goal state where:
       - Both cups' contents are known
       - Empty cup is inside the bucket
       
    4. Verifies:
       - A valid plan is found that includes:
         a. Moving to observe first cup from top
         b. Observing first cup content
         c. Moving to observe second cup from top
         d. Observing second cup content
         e. Moving to view empty cup for picking
         f. Picking up empty cup
         g. Moving to bucket
         h. Dropping empty cup in bucket
       - The transition graph shows proper action sequencing
       
    Output:
       - mock_env_data/test_observe_two_cups_and_place_empty/transition_graph.png
    """
    # Set up configuration
    test_name = "test_observe_two_cups_and_place_empty"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir
    })
    
    # Enable belief-space operators
    MockSpotEnv.use_belief_space_operators = True
    
    # Create environment creator
    env_creator = ManualMockEnvCreator(test_dir)
    
    # Create test objects with descriptive names
    robot = Object("robot", _robot_type)
    water_cup = Object("water_cup", _container_type)  # Will be observed to contain water
    empty_cup = Object("empty_cup", _container_type)  # Will be observed to be empty
    table = Object("table", _immovable_object_type)
    bucket = Object("bucket", _container_type)
    objects = {robot, water_cup, empty_cup, table, bucket}
    
    # Create initial state atoms
    initial_atoms = {
        # Robot state
        GroundAtom(_HandEmpty, [robot]),
        
        # Water cup state
        GroundAtom(_On, [water_cup, table]),
        GroundAtom(_ContainingWaterUnknown, [water_cup]),
        GroundAtom(_NotBlocked, [water_cup]),
        GroundAtom(_IsPlaceable, [water_cup]),
        GroundAtom(_NotInsideAnyContainer, [water_cup]),
        GroundAtom(_FitsInXY, [water_cup, table]),
        GroundAtom(_FitsInXY, [water_cup, bucket]),
        GroundAtom(_NotHolding, [robot, water_cup]),
        GroundAtom(_NEq, [water_cup, table]),
        GroundAtom(_NEq, [water_cup, bucket]),
        
        # Empty cup state
        GroundAtom(_On, [empty_cup, table]),
        GroundAtom(_ContainingWaterUnknown, [empty_cup]),
        GroundAtom(_NotBlocked, [empty_cup]),
        GroundAtom(_IsPlaceable, [empty_cup]),
        GroundAtom(_NotInsideAnyContainer, [empty_cup]),
        GroundAtom(_FitsInXY, [empty_cup, table]),
        GroundAtom(_FitsInXY, [empty_cup, bucket]),
        GroundAtom(_NotHolding, [robot, empty_cup]),
        GroundAtom(_NEq, [empty_cup, table]),
        GroundAtom(_NEq, [empty_cup, bucket]),
        
        # Environment state
        GroundAtom(_HasFlatTopSurface, [table]),
        GroundAtom(_Reachable, [robot, water_cup]),
        GroundAtom(_Reachable, [robot, empty_cup]),
        GroundAtom(_Reachable, [robot, bucket]),
        GroundAtom(_NEq, [water_cup, empty_cup])
    }
    
    # Create goal atoms
    goal_atoms = {
        # Both cups should be observed
        GroundAtom(_ContainingWaterKnown, [water_cup]),
        GroundAtom(_ContainingWaterKnown, [empty_cup]),
        # Empty cup should be in bucket
        GroundAtom(_Inside, [empty_cup, bucket])
    }
    
    # Plan and visualize transitions
    env_creator.plan_and_visualize(initial_atoms, goal_atoms, objects, task_name=f'Transition Graph, {test_name.replace("_", " ").title()}')
    
    # Verify transition graph file exists
    graph_file = Path(test_dir) / "transitions" / "transition_graph.png"
    assert graph_file.exists(), "Transition graph file not generated"
    
    # Cleanup
    MockSpotEnv.use_belief_space_operators = False 