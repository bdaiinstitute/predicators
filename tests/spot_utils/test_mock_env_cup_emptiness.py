"""Tests for mock environment cup emptiness belief state planning."""

import os
import tempfile
import numpy as np
from predicators.envs.mock_spot_env import (
    MockSpotEnv, _robot_type, _container_type, _immovable_object_type,
    _HandEmpty, _NotHolding, _On, _NotBlocked
)
from predicators.structs import Object, GroundAtom, State, Task
from predicators.spot_utils.mock_env.mock_env_creator_manual import ManualMockEnvCreator
from predicators import utils
from predicators.settings import CFG


# TODO: Re-enable this test after fixing base pick-place functionality and adding belief state predicates
# def test_cup_emptiness_belief():
#     """Test transition graph calculation for checking cup emptiness using belief space planning."""
#     # Set up configuration
#     temp_dir = tempfile.mkdtemp()
#     utils.reset_config({
#         "env": "mock_spot",
#         "approach": "oracle",
#         "seed": 123,
#         "num_train_tasks": 0,
#         "num_test_tasks": 1,
#         "mock_env_data_dir": temp_dir
#     })
#     
#     # Create environment creator
#     creator = ManualMockEnvCreator(temp_dir)
#     
#     # Create objects
#     robot = Object("robot", _robot_type)
#     cup1 = Object("cup1", _container_type)
#     cup2 = Object("cup2", _container_type)
#     table = Object("table", _immovable_object_type)
#     
#     # Create initial state with proper feature dimensions for each object type
#     objects = {robot, cup1, cup2, table}
#     init_state = State({
#         obj: np.zeros(obj.type.dim, dtype=np.float32)
#         for obj in objects
#     })
#     
#     # Create initial atoms
#     initial_atoms = {
#         GroundAtom(_HandEmpty, [robot]),
#         GroundAtom(_NotHolding, [robot, cup1]),
#         GroundAtom(_NotHolding, [robot, cup2]),
#         GroundAtom(_On, [cup1, table]),
#         GroundAtom(_On, [cup2, table]),
#         GroundAtom(_NotBlocked, [cup1]),
#         GroundAtom(_NotBlocked, [cup2]),
#         GroundAtom(_ContainingWaterUnknown, [cup1]),
#         GroundAtom(_ContainingWaterUnknown, [cup2])
#     }
#     
#     # Create goal atoms
#     goal_atoms = {
#         GroundAtom(_ContainingWaterKnown, [cup1]),
#         GroundAtom(_ContainingWaterKnown, [cup2])
#     }
#     
#     # Create initial state
#     init_state = State({
#         obj: np.zeros(obj.type.dim, dtype=np.float32)
#         for obj in objects
#     }, initial_atoms)
#     
#     # Create task
#     task = Task(init_state, goal_atoms)
#     
#     # Plan and visualize transitions
#     creator.plan_and_visualize(initial_atoms, goal_atoms, objects, "cup_emptiness")
#     
#     # Verify output file exists
#     assert os.path.exists(os.path.join(creator.transitions_dir, "cup_emptiness.png")) 