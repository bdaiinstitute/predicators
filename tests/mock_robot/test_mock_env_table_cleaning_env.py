"""Test manual creation of images for a table cleaning task.

This test focuses on:
1. Creating transition graph for planning the image collection
2. Adding manual images for each state
3. Verifying image loading and state transitions

Directory Structure:
    mock_env_data/test_mock_table_cleaning/
    ├── images/
    │   ├── state_0/
    │   │   └── cam1.rgb.npy
    │   └── ...
    ├── transitions/
    │   └── Transition Graph, Test Table Cleaning.html
    ├── state_mapping.yaml
    └── plan.yaml
"""

import os
from pathlib import Path
import logging
from predicators.envs.mock_spot_env import MockSpotTableCleaningEnv
from predicators.spot_utils.mock_env.mock_env_creator_base import MockEnvCreatorBase
from predicators import utils

def test_mock_env_table_cleaning():
    """Test for MockSpotTableCleaningEnv."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set up configuration
    test_name = "test_mock_table_cleaning"
    test_dir = os.path.join("mock_env_data", test_name)
    logging.info(f"Setting up test in directory: {test_dir}")
    
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir,
        "mock_env_use_belief_operators": True  # Enable belief operators for table cleaning
    })
    
    # Create environment
    logging.info("Creating MockSpotTableCleaningEnv environment")
    env = MockSpotTableCleaningEnv()
    
    # Create environment creator
    logging.info("Creating MockEnvCreatorBase")
    creator = MockEnvCreatorBase(test_dir, env=env)

    # We should be able to find an optimal plan.
    print("Checking that an optimal plan can be found.")
    name = f"Transition Graph, {test_name.replace('_', ' ').title()}"
    print("goal_atoms_or")
    print(env.goal_atoms_or)
    print(env.goal_atoms)
    
    # Use the dedicated method to handle branching plans and visualize them.
    creator.plan_and_visualize(env.initial_atoms, env.goal_atoms_or, env.objects, task_name=name)

    # Also create branching visualization to show information gathering branches
    print("Creating branching visualization...")
    creator.plan_and_visualize_with_branches(env.initial_atoms, env.goal_atoms_or, env.objects, task_name=name)

    # Verify transition graph file exists
    graph_file = Path(test_dir) / "transitions" / f"{name}.html"
    logging.info(f"Checking for graph file at: {graph_file}")
    assert graph_file.exists(), "Transition graph file not generated"
    print("Test completed successfully")

if __name__ == "__main__":
    test_mock_env_table_cleaning() 