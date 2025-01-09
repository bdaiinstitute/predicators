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
    # Set up configuration
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1
    })

    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    try:
        # Initialize environment
        env = MockSpotEnv(data_dir=temp_dir)
        
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
        assert len(env.strips_operators) == 5
        MoveToReachObject, MoveToHandViewObject, PickObjectFromTop, \
            PlaceObjectOnTop, DropObjectInside = sorted(env.strips_operators)
        
        # Test state creation and transitions
        state_id_1 = env.add_state(
            rgbd=None,
            gripper_open=True,
            objects_in_view={"cup", "table"},
            objects_in_hand=set()
        )
        assert state_id_1 == "0"
        
        state_id_2 = env.add_state(
            rgbd=None,
            gripper_open=False,
            objects_in_view={"cup", "table"},
            objects_in_hand={"cup"}
        )
        assert state_id_2 == "1"
        
        # Test adding valid transition
        env.add_transition(state_id_1, "PickObjectFromTop", state_id_2)
        
        # Test adding invalid transitions
        with pytest.raises(ValueError):
            env.add_transition("invalid_id", "PickObjectFromTop", state_id_2)
        
        with pytest.raises(ValueError):
            env.add_transition(state_id_1, "InvalidOperator", state_id_2)
        
        # Test graph data persistence
        # Create new environment instance with same data directory
        env2 = MockSpotEnv(data_dir=temp_dir)
        assert len(env2._observations) == 2
        assert len(env2._transitions) == 1
        assert env2._transitions[state_id_1]["PickObjectFromTop"] == state_id_2
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir) 