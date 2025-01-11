"""Test cases for the mock Spot environment."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from predicators import utils
from predicators.envs.mock_spot_env import MockSpotEnv
from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext


def test_mock_spot_env():
    """Tests for mock Spot environment."""
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    try:
        # Initialize environment
        utils.reset_config({
            "env": "mock_spot",
            "approach": "oracle",
            "seed": 123,
            "num_train_tasks": 0,
            "num_test_tasks": 1,
            "mock_env_data_dir": temp_dir
        })
        
        # Initialize environment
        env = MockSpotEnv()
        
        # Test basic properties
        assert env.get_name() == "mock_spot"
        
        # Test types
        assert len(env.types) == 5
        robot_type = next(t for t in env.types if t.name == "robot")
        base_object_type = next(t for t in env.types if t.name == "base_object")
        movable_object_type = next(t for t in env.types if t.name == "movable_object")
        container_type = next(t for t in env.types if t.name == "container")
        immovable_object_type = next(t for t in env.types if t.name == "immovable_object")
        
        assert movable_object_type.parent == base_object_type
        assert container_type.parent == movable_object_type
        assert immovable_object_type.parent == base_object_type
        
        # Test predicates
        assert len(env.predicates) == 25
        NEq, On, TopAbove, Inside, NotInsideAnyContainer, FitsInXY, HandEmpty, \
            Holding, NotHolding, InHandView, InView, Reachable, Blocking, NotBlocked, \
            ContainerReadyForSweeping, IsPlaceable, IsNotPlaceable, IsSweeper, \
            HasFlatTopSurface, RobotReadyForSweeping, ContainingWaterUnknown, \
            ContainingWaterKnown, ContainingWater, NotContainingWater, \
            InHandViewFromTop = sorted(env.predicates)
        
        # Test operators
        assert len(env.strips_operators) == 7
        MoveToReachObject, MoveToHandViewObject, MoveToHandObserveObjectFromTop, \
            ObserveFromTop, PickObjectFromTop, PlaceObjectOnTop, \
            DropObjectInside = sorted(env.strips_operators)
        
        # Test state creation and transitions
        state_id_1 = env.add_state(
            rgbd=None,
            gripper_open=True,
            objects_in_view={"cup1", "cup2", "table"},
            objects_in_hand=set()
        )
        assert state_id_1 == "0"
        
        state_id_2 = env.add_state(
            rgbd=None,
            gripper_open=True,
            objects_in_view={"cup1", "cup2", "table"},
            objects_in_hand=set()
        )
        assert state_id_2 == "1"
        
        # Test adding valid transitions
        env.add_transition(state_id_1, "MoveToHandObserveObjectFromTop", state_id_2)
        
        # Test adding invalid transitions
        with pytest.raises(ValueError):
            env.add_transition("invalid_id", "MoveToHandObserveObjectFromTop", state_id_2)
        
        with pytest.raises(ValueError):
            env.add_transition(state_id_1, "InvalidOperator", state_id_2)
        
        # Test graph data persistence
        # Create new environment instance with same data directory
        env2 = MockSpotEnv()
        assert len(env2._observations) == 2
        assert len(env2._transitions) == 1
        assert env2._transitions[state_id_1]["MoveToHandObserveObjectFromTop"] == state_id_2
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def test_pick_from_top():
    """Test pick-from-top operation sequence."""
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    try:
        # Initialize environment
        utils.reset_config({
            "env": "mock_spot",
            "approach": "oracle",
            "seed": 123,
            "num_train_tasks": 0,
            "num_test_tasks": 1,
            "mock_env_data_dir": temp_dir
        })
        
        # Initialize environment
        env = MockSpotEnv()
        
        # Test pick-from-top sequence
        # Initial state: robot hand empty, block on table
        state_id_1 = env.add_state(
            rgbd=None,
            gripper_open=True,
            objects_in_view={"block", "table"},
            objects_in_hand=set()
        )
        
        # State after moving to reach block
        state_id_2 = env.add_state(
            rgbd=None,
            gripper_open=True,
            objects_in_view={"block", "table"},
            objects_in_hand=set()
        )
        
        # State after picking block
        state_id_3 = env.add_state(
            rgbd=None,
            gripper_open=False,
            objects_in_view={"block", "table"},
            objects_in_hand={"block"}
        )
        
        # Add transitions
        env.add_transition(state_id_1, "MoveToReachObject", state_id_2)
        env.add_transition(state_id_2, "PickObjectFromTop", state_id_3)
        
        # Verify transitions
        assert env._transitions[state_id_1]["MoveToReachObject"] == state_id_2
        assert env._transitions[state_id_2]["PickObjectFromTop"] == state_id_3
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def test_move_to_view_from_top():
    """Test move-to-view-from-top operation sequence."""
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    try:
        # Initialize environment
        utils.reset_config({
            "env": "mock_spot",
            "approach": "oracle",
            "seed": 123,
            "num_train_tasks": 0,
            "num_test_tasks": 1,
            "mock_env_data_dir": temp_dir
        })
        
        # Initialize environment
        env = MockSpotEnv()
        
        # Test move-to-view-from-top sequence
        # Initial state: robot hand empty, cup on table
        state_id_1 = env.add_state(
            rgbd=None,
            gripper_open=True,
            objects_in_view={"cup", "table"},
            objects_in_hand=set()
        )
        
        # State after moving to view cup from top
        state_id_2 = env.add_state(
            rgbd=None,
            gripper_open=True,
            objects_in_view={"cup", "table"},
            objects_in_hand=set()
        )
        
        # State after observing cup from top
        state_id_3 = env.add_state(
            rgbd=None,
            gripper_open=True,
            objects_in_view={"cup", "table"},
            objects_in_hand=set()
        )
        
        # Add transitions
        env.add_transition(state_id_1, "MoveToHandObserveObjectFromTop", state_id_2)
        env.add_transition(state_id_2, "ObserveFromTop", state_id_3)
        
        # Verify transitions
        assert env._transitions[state_id_1]["MoveToHandObserveObjectFromTop"] == state_id_2
        assert env._transitions[state_id_2]["ObserveFromTop"] == state_id_3
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir) 