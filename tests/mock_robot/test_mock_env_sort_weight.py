"""Test manual creation of images for a drawer cleaning task using phone HEIC images.

This test focuses on:
1. Creating transition graph for planning the image collection
2. Adding manual HEIC images from phone for each state
3. Verifying image loading and state transitions

Directory Structure:
    mock_env_data/test_mock_task_phone_drawer_cleaning/
    ├── images/
    │   ├── state_0/
    │   │   └── cam1.rgb.npy
    │   └── ...
    ├── transitions/
    │   └── Transition Graph, Test Drawer Cleaning.html
    ├── state_mapping.yaml
    └── plan.yaml
"""

import os
from pathlib import Path
from predicators.envs.mock_spot_env import (MockSpotSortWeight)
from predicators.spot_utils.mock_env.mock_env_creator_base import MockEnvCreatorBase
from predicators import utils

if __name__ == '__main__':
    """Create transition graph for the drawer cleaning task.
    
    This test only creates the transition graph without adding images.
    Use this to guide the image collection process:
    
    State 0: Initial state - both cups in drawer, drawer closed
    State 1: Drawer open, both cups visible
    State 2: First cup (red) in hand
    State 3: First cup (red) in container
    State 4: Second cup (blue) in hand
    State 5: Second cup (blue) in container
    State 6: Drawer closed (final state)
    """
    # Set up configuration
    test_name = "task_phone_drawer_cleaning"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir,
        "mock_env_use_belief_operators": False
    })
    
    # Create environment
    env = MockSpotSortWeight()
    
    # Create environment creator
    creator = MockEnvCreatorBase(test_dir, env=env)
    
    # Plan and visualize transitions
    name = f'Transition Graph, {test_name.replace("_", " ").title()}'
    creator.plan_and_visualize(env.initial_atoms, env.goal_atoms_or, env.objects, task_name=name)
    
    # Verify transition graph file exists
    graph_file = Path(test_dir) / "transitions" / f"{name}.html"
    assert graph_file.exists(), "Transition graph file not generated"
