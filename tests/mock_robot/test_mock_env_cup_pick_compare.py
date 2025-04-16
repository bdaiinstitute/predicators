"""Test picking empty cup with and without belief state uncertainty.

This test compares two scenarios:
1. Picking an empty cup with known environment state
2. Picking an empty cup with belief state uncertainty
"""

import os
from pathlib import Path

import pytest
from rich.console import Console
from rich.logging import RichHandler
import logging

from predicators import utils
from predicators.envs.mock_spot_env import (
    MockSpotEnv, _NotBlocked, _HandEmpty, _ContainingWaterUnknown, _ContainingWaterKnown,
    _On, _NotInsideAnyContainer, _IsPlaceable, _HasFlatTopSurface, _Reachable,
    _InHandView, _NEq, _Holding, _Inside, _FitsInXY, _NotHolding, _InHandViewFromTop,
    _robot_type, _container_type, _immovable_object_type
)
from predicators.structs import Object, State, GroundAtom
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
def reset_config():
    """Reset configuration before each test."""
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1
    })


def test_pick_empty_cup_env_state():
    """Test picking empty cup with known environment state.
    
    This test demonstrates the simpler case where we know the cup's contents
    and just need to:
    1. Move to view cup
    2. Pick cup
    
    Does not require belief state tracking.
    """
    # Set up configuration
    test_name = "test_pick_empty_cup_env_state"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir
    })

    # Disable belief-space operators
    MockSpotEnv.use_belief_space_operators = False

    # Create environment creator
    env_creator = ManualMockEnvCreator(test_dir)

    # Create test objects
    robot = Object("robot", _robot_type)
    cup = Object("cup", _container_type)
    table = Object("table", _immovable_object_type)
    objects = {robot, cup, table}

    # Create initial state atoms - cup contents known
    initial_atoms = {
        # Robot state
        GroundAtom(_HandEmpty, [robot]),

        # Cup state
        GroundAtom(_On, [cup, table]),
        GroundAtom(_ContainingWaterKnown, [cup]),  # Key difference: Cup content known
        GroundAtom(_NotBlocked, [cup]),
        GroundAtom(_IsPlaceable, [cup]),
        GroundAtom(_NotInsideAnyContainer, [cup]),
        GroundAtom(_FitsInXY, [cup, table]),
        GroundAtom(_NotHolding, [robot, cup]),

        # Environment state
        GroundAtom(_HasFlatTopSurface, [table]),
        GroundAtom(_Reachable, [robot, cup]),
        GroundAtom(_NEq, [cup, table])
    }

    # Create goal atoms - just need to hold the cup
    goal_atoms = {
        GroundAtom(_Holding, [robot, cup])
    }

    # Plan and visualize transitions
    name = f'Transition Graph, {test_name.replace("_", " ").title()}'
    env_creator.plan_and_visualize(initial_atoms, goal_atoms, objects, task_name=name)
    
    # Verify output file exists
    graph_file = Path(test_dir) / "transitions" / f"{name}.png"
    assert graph_file.exists(), "Transition graph file not generated"


def test_pick_empty_cup_belief_state():
    """Test picking empty cup with belief state uncertainty.
    
    This test demonstrates the more complex case where we maintain uncertainty about
    the cup's contents and need to:
    1. Move to observe cup from top
    2. Observe cup contents
    3. Move to view cup
    4. Pick cup
    
    Requires belief state tracking and observation actions.
    """
    # Set up configuration
    test_name = "test_pick_empty_cup_belief_state"
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
    env = MockSpotEnv(use_gui=False)

    # Create test objects
    robot = Object("robot", _robot_type)
    cup = Object("cup", _container_type)
    table = Object("table", _immovable_object_type)
    objects = {robot, cup, table}

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
    test_dir = Path("mock_env_data") / "test_pick_empty_cup_belief_state"
    os.makedirs(test_dir / "transitions", exist_ok=True)
    env._save_graph_data()

    # Verify all transitions in sequence
    assert env._str_transitions[state_id]["MoveToHandObserveObjectFromTop"] == observe_state_id
    assert env._str_transitions[observe_state_id]["ObserveContainerContent"] == known_state_id
    assert env._str_transitions[known_state_id]["MoveToHandViewObject"] == view_state_id
    assert env._str_transitions[view_state_id]["PickObjectFromTop"] == holding_state_id

    # Verify final state has cup in hand
    final_obs = env._observations[holding_state_id]
    assert cup.name in final_obs.objects_in_hand
    assert not final_obs.gripper_open

    # Cleanup
    MockSpotEnv.use_belief_space_operators = False 