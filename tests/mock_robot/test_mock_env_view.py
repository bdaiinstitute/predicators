"""Tests for mock environment view operators."""

import os
import numpy as np
from predicators.envs.mock_spot_env import (
    MockSpotEnv, _robot_type, _container_type, _immovable_object_type,
    _HandEmpty, _NotHolding, _On, _NotBlocked, _InHandView, _Reachable,
    _Inside, _IsPlaceable, _HasFlatTopSurface, _FitsInXY, _NotInsideAnyContainer
)
from predicators.structs import Object, GroundAtom, State, Task
from predicators.spot_utils.mock_env.mock_env_creator_manual import ManualMockEnvCreator
from predicators import utils
from predicators.settings import CFG


def test_view_operators():
    """Test transition graph calculation for view operators."""
    # Set up configuration
    test_name = "test_view_operators"
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
    cup = Object("cup", _container_type)
    table = Object("table", _immovable_object_type)
    
    # Create initial atoms
    objects = {robot, cup, table}
    initial_atoms = {
        GroundAtom(_HandEmpty, [robot]),
        GroundAtom(_NotHolding, [robot, cup]),
        GroundAtom(_On, [cup, table]),
        GroundAtom(_NotBlocked, [cup])
    }
    
    # Create goal atoms - cup should be in hand view
    goal_atoms = {
        GroundAtom(_InHandView, [robot, cup]),
        GroundAtom(_Reachable, [robot, cup])
    }
    
    # Plan and visualize transitions
    name = f'Transition Graph, {test_name.replace("_", " ").title()}'
    creator.plan_and_visualize(initial_atoms, goal_atoms, objects, task_name=name)
    
    # Verify output file exists
    assert os.path.exists(os.path.join(creator.transitions_dir, f"{name}.html")) 