#!/usr/bin/env python3

"""Test manual creation of images for a cup classification task.

This test focuses on:
1. Creating transition graph for planning the image collection
2. Adding manual images for each state
3. Verifying image loading and state transitions

Directory Structure:
    mock_env_data/test_mock_cup_classification/
    ├── images/
    │   ├── state_0/
    │   │   └── cam1.rgb.npy
    │   └── ...
    ├── transitions/
    │   └── Transition Graph, Test Cup Classification.html
    ├── state_mapping.yaml
    └── plan.yaml
"""

import os
from pathlib import Path
import logging
from predicators.envs.mock_spot_env import MockSpotCupClassificationEnv
from predicators.spot_utils.mock_env.mock_env_creator_base import MockEnvCreatorBase
from predicators import utils

if __name__ == '__main__':
    """Create transition graph for the cup classification task.
    
    This test creates the transition graph to guide the image collection process:
    
    State 0: Initial state - cup on table, type unknown
    State 1: Robot viewing cup from top (inspection)
    State 2: Cup type observed (disposable/washable)
    State 3: Robot inspecting cup contents
    State 4: Cup contents observed (dirty/clean)
    State 5a: (If disposable) Cup placed in trash
    State 5b: (If washable & dirty) Contents dumped
    State 6b: (If washable & dirty) Cup placed in dishwasher
    State 7b: (If washable & clean) Cup placed in cabinet
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set up configuration
    test_name = "test_mock_cup_classification"
    test_dir = os.path.join("mock_env_data", test_name)
    logging.info(f"Setting up test in directory: {test_dir}")
    
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir,
        "mock_env_use_belief_operators": True  # Enable belief operators for cup classification
    })
    
    # Create environment
    logging.info("Creating MockSpotCupClassificationEnv environment")
    env = MockSpotCupClassificationEnv()
    
    # Create environment creator
    logging.info("Creating MockEnvCreatorBase")
    creator = MockEnvCreatorBase(test_dir, env=env)
    
    # Plan and visualize transitions
    name = f'Transition Graph, {test_name.replace("_", " ").title()}'
    logging.info(f"Planning and visualizing transitions with name: {name}")
    
    creator.plan_and_visualize(env.initial_atoms, env.goal_atoms_or, env.objects, task_name=name)
    
    # Verify transition graph file exists
    graph_file = Path(test_dir) / "transitions" / f"{name}.html"
    logging.info(f"Checking for graph file at: {graph_file}")
    assert graph_file.exists(), "Transition graph file not generated"
    logging.info("Test completed successfully") 