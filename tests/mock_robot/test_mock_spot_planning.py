"""Test suite for verifying planning in the mock Spot environment.

This test suite verifies that the mock Spot environment correctly handles:
1. Action wrapping with operator information
2. Basic pick and place planning
3. Full pipeline integration
"""

import os
import tempfile
import shutil
from typing import Dict, Any, cast, List

import numpy as np
from rich import print

from predicators import utils
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.structs import State, Task, EnvironmentTask, GroundAtom
from predicators.cogman import CogMan
from predicators.approaches import create_approach
from predicators.execution_monitoring import create_execution_monitor
from predicators.perception import create_perceiver
from predicators.envs.mock_spot_env import (
    MockSpotPickPlaceTwoCupEnv, _MockSpotObservation,
    _Inside, _On, _HandEmpty, _NotHolding, _Reachable, _InHandView
)
from predicators.main import _run_episode


def test_mock_spot_action_wrapping():
    """Test that actions are properly wrapped with operator information."""
    # Create temp directory for test data
    temp_dir = tempfile.mkdtemp()
    try:
        # Set up configuration
        utils.reset_config({
            "env": "mock_spot",
            "approach": "oracle",
            "seed": 123,
            "mock_env_data_dir": temp_dir,
            "num_test_tasks": 1
        })

        # Create environment and get options
        env = create_new_env("mock_spot")
        options = get_gt_options(env.get_name())

        # Verify each option creates properly wrapped actions
        for option in options:
            # Create dummy state and parameters
            dummy_state = State({}, set(), {}, set())
            memory = {}
            objects = []
            params = np.zeros(1, dtype=np.float32)

            # Get action from option's policy
            action = option.policy(dummy_state, memory, objects, params)

            # Verify action has proper operator information
            extra_info = cast(Dict[str, Any], action.extra_info)
            assert extra_info is not None
            assert "operator_name" in extra_info
            assert extra_info["operator_name"] == option.name
            assert isinstance(action.arr, np.ndarray)
            assert action.arr.shape == (1,)  # Mock actions use dummy array
            assert np.all(action.arr == 0)  # Should be zeros

    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_mock_pick_and_place_planning():
    """Test pick and place planning with two cups."""
    # Set up configuration with test data directory
    test_name = "test_mock_two_cup_pick_place_manual_images"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "mock_env_use_belief_operators": True,
        "mock_env_data_dir": test_dir
    })

    # Create environment with two cups
    env = MockSpotPickPlaceTwoCupEnv()
    
    # Get initial observation
    obs = env.reset('test', task_idx=0)
    assert isinstance(obs, _MockSpotObservation)
    
    # Create perceiver and initialize with observation
    perceiver = create_perceiver("mock_spot")
    env_task = cast(EnvironmentTask, env._current_task)  # Get task from environment
    task = perceiver.reset(env_task)  # Reset perceiver with task
    train_tasks = [task]  # No need to cast since we're using the task from perceiver
    
    # Set up planning components
    options = get_gt_options(env.get_name())
    approach = create_approach("oracle", env.predicates, options, env.types, env.action_space, train_tasks)
    exec_monitor = create_execution_monitor("trivial")
    cogman = CogMan(approach, perceiver, exec_monitor)
    cogman.reset(env_task)  # Use env_task here
    
    # Run episode
    max_steps = 20  # Increased steps for two-cup task
    (states, actions), solved, metrics = _run_episode(
        cogman,
        env,
        "test",
        0,
        max_steps,
        do_env_reset=True,
        terminate_on_goal_reached=True
    )
    
    # Verify results
    assert len(states) > 1, "Planning should produce multiple states"
    assert len(actions) == len(states) - 1, "Should have one less action than states"
    assert solved, "Task should be solved"
    
    # Verify action sequence follows expected pattern
    action_names = [action.extra_info["operator_name"] for action in actions]
    expected_operators = {"PickObjectFromTop", "DropObjectInside"}
    assert all(name in expected_operators for name in action_names)
    
    # Verify final state has cups inside target
    final_state = states[-1]
    assert isinstance(final_state, _MockSpotObservation)
    assert final_state.atom_dict is not None
    atoms = set(atom for atom, val in final_state.atom_dict.items() if val)
    assert any(atom.predicate == _Inside for atom in atoms)


def test_mock_spot_pipeline():
    """Test running through the main pipeline with mock Spot environment."""
    # Set up configuration with test data directory
    test_name = "test_mock_two_cup_pick_place_manual_images"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "mock_env_use_belief_operators": True,
        "mock_env_data_dir": test_dir,
        "mock_env_vlm_eval_predicate": True  # Enable VLM predicates
    })
    
    # Create environment and get task
    env = MockSpotPickPlaceTwoCupEnv()
    obs = env.reset('test', task_idx=0)
    assert isinstance(obs, _MockSpotObservation)
    
    # Print initial state info
    print("\n=== Initial State Check ===")
    print("Expected initial atoms from env:")
    print(env.initial_atoms)
    print("\nActual atoms from observation:")
    if obs.non_vlm_atom_dict:
        actual_atoms = {atom for atom, val in obs.non_vlm_atom_dict.items() if val}
        print(actual_atoms)
    else:
        print("No non-VLM atoms in observation")
    
    # Print VLM predicate info
    print("\n=== VLM Predicate Check ===")
    print("VLM predicates in observation:", obs.vlm_predicates)
    print("VLM atom dict in observation:", obs.vlm_atom_dict)
    
    # Create perceiver and initialize with observation
    perceiver = create_perceiver("mock_spot")
    env_task = cast(EnvironmentTask, env._current_task)  # Get task from environment
    task = perceiver.reset(env_task)  # Reset perceiver with task
    train_tasks = [task]  # No need to cast since we're using the task from perceiver
    
    # Print task info
    print("\n=== Task Check ===")
    print("Goal atoms from env:", env.goal_atoms)
    print("Task goal:", task.goal)
    
    # Set up pipeline components
    options = get_gt_options(env.get_name())
    approach = create_approach("oracle", env.predicates, options, env.types, env.action_space, train_tasks)
    exec_monitor = create_execution_monitor("trivial")
    cogman = CogMan(approach, perceiver, exec_monitor)
    cogman.reset(env_task)  # Use env_task here
    
    # Run episode
    max_steps = 20
    (states, actions), solved, metrics = _run_episode(
        cogman,
        env,
        "test",
        0,
        max_steps,
        do_env_reset=True,
        terminate_on_goal_reached=True
    )
    
    # Verify results
    assert len(states) > 1, "Pipeline should produce multiple states"
    assert len(actions) == len(states) - 1, "Should have one less action than states"
    assert solved, "Pipeline should solve task"
    
    # Verify action sequence follows expected pattern
    action_names = [action.extra_info["operator_name"] for action in actions]
    expected_operators = {"PickObjectFromTop", "DropObjectInside"}
    assert all(name in expected_operators for name in action_names)
    
    # Verify state transitions
    for i, state in enumerate(states[:-1]):
        action = actions[i]
        next_state = states[i + 1]
        
        # Verify states have atom dictionaries
        assert isinstance(state, _MockSpotObservation)
        assert isinstance(next_state, _MockSpotObservation)
        assert state.atom_dict is not None
        assert next_state.atom_dict is not None
        
        # Get true atoms from dictionaries
        state_atoms = set(atom for atom, val in state.atom_dict.items() if val)
        next_atoms = set(atom for atom, val in next_state.atom_dict.items() if val)
        
        # Check state transitions make sense
        if action.extra_info["operator_name"] == "PickObjectFromTop":
            # After picking, object should not be on surface
            assert not any(atom.predicate == _On for atom in next_atoms)
            # Robot should not have empty hand
            assert not any(atom.predicate == _HandEmpty for atom in next_atoms)
            
        elif action.extra_info["operator_name"] == "DropObjectInside":
            # After dropping, object should be inside target
            assert any(atom.predicate == _Inside for atom in next_atoms)
            # Robot should have empty hand
            assert any(atom.predicate == _HandEmpty for atom in next_atoms)
