"""Tests for mock environment transition graph calculation."""

import os
import tempfile
import numpy as np
from predicators.envs.mock_spot_env import (
    MockSpotEnv, _robot_type, _container_type, _immovable_object_type,
    _HandEmpty, _NotHolding, _On, _NotBlocked, _Inside, _IsPlaceable, _HasFlatTopSurface, _FitsInXY, _NotInsideAnyContainer
)
from predicators.structs import Object, GroundAtom, State, Task
from predicators.spot_utils.mock_env.mock_env_creator_manual import ManualMockEnvCreator
from predicators import utils
from predicators.settings import CFG


def test_two_object_pick_place():
    """Test transition graph calculation for a two-object pick-and-place task."""
    # Set up configuration
    temp_dir = tempfile.mkdtemp()
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": temp_dir
    })
    
    # Create environment creator
    creator = ManualMockEnvCreator(temp_dir)
    
    # Create objects
    robot = Object("robot", _robot_type)
    block1 = Object("block1", _container_type)
    block2 = Object("block2", _container_type)
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
        GroundAtom(_NotHolding, [robot, container])
    }
    
    # Create goal atoms - both blocks should be in the container
    goal_atoms = {
        GroundAtom(_Inside, [block1, container]),
        GroundAtom(_Inside, [block2, container])
    }
    
    # Plan and visualize transitions
    creator.plan_and_visualize(initial_atoms, goal_atoms, objects, "two_object_pick_place")
    
    # Verify output file exists
    assert os.path.exists(os.path.join(creator.transitions_dir, "two_object_pick_place.png")) 