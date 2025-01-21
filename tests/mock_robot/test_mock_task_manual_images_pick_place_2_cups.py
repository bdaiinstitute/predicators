"""Test manual creation of images for a two-cup pick-and-place task.

This test combines:
1. Transition graph creation for a two-cup pick-and-place task
2. Manual addition of images for each state in the transition graph

The test verifies:
1. Transition Graph Creation:
   - Proper state transitions for two-cup pick-and-place
   - Valid action sequences
   - Goal state achievement

2. Image Management:
   - Loading and storing RGB/depth images for each state
   - Proper metadata storage

Directory Structure:
    mock_env_data/test_two_cup_pick_place/
    ├── images/
    │   ├── state_0/
    │   │   ├── cam1.rgb.npy
    │   │   └── cam1.depth.npy
    │   └── ...
    ├── transitions/
    │   └── Transition Graph, Test Two Cup Pick Place.html
    └── plan.yaml
"""

import os
import numpy as np
import pytest
from pathlib import Path
import json
from typing import Dict, Tuple, Literal, cast, Any

from predicators.envs.mock_spot_env import MockSpotPickPlaceTwoCupEnv
from predicators.structs import Object
from predicators.spot_utils.mock_env.mock_env_creator_base import MockEnvCreatorBase
from predicators import utils


def test_two_cup_pick_place_with_manual_images():
    """Test creating a two-cup pick-and-place task with manual images.
    
    This test:
    1. Creates a pick-and-place scenario with:
       - A robot
       - A table (immovable object)
       - A target container
       - Two cups (container objects to be moved)
       
    2. Generates transition graph showing:
       - Initial state (cups on table)
       - Intermediate states (picking and placing each cup)
       - Goal state (cups inside target container)
       
    3. Adds manual images for each state:
       - RGB and depth images
       - Multiple views and cameras
       - Proper metadata
    """
    # Set up configuration
    test_name = "test_mock_two_cup_pick_place_manual_images"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir
    })
    
    # Create environment
    env = MockSpotPickPlaceTwoCupEnv()
    
    # Create environment creator with pick-place specific env
    creator = MockEnvCreatorBase(test_dir, env_info={
        "types": env.types,
        "predicates": env.predicates,
        "options": env.options,
        "nsrts": env.nsrts
    })
    
    # Plan and visualize transitions
    name = f'Transition Graph, {env.name.replace("_", " ").title()}'
    creator.plan_and_visualize(env.initial_atoms, env.goal_atoms, env.objects, task_name=name)
    
    # Get example image paths
    example_rgb_path = os.path.join(os.path.dirname(__file__), "example_manual_mock_task_1", "state1_view1_img1.png")
    example_depth_path = os.path.join(os.path.dirname(__file__), "example_manual_mock_task_1", "state1_view1_depth1.npy")
    depth_img = np.ones((480, 640), dtype=np.float32)
    np.save(example_depth_path, depth_img)
    
    try:
        # Define states and their images
        test_state_images = {
            # Initial state - both cups on table
            "state_0": {
                "cam1.seed0.rgb": (example_rgb_path, "rgb"),
                "cam1.seed0.depth": (example_depth_path, "depth"),
            },
            
            # First cup in hand
            "state_1": {
                "cam1.seed0.rgb": (example_rgb_path, "rgb"),
                "cam1.seed0.depth": (example_depth_path, "depth"),
            },
            
            # First cup in target
            "state_2": {
                "cam1.seed0.rgb": (example_rgb_path, "rgb"),
                "cam1.seed0.depth": (example_depth_path, "depth"),
            },
            
            # Second cup in hand
            "state_3": {
                "cam1.seed0.rgb": (example_rgb_path, "rgb"),
                "cam1.seed0.depth": (example_depth_path, "depth"),
            },
            
            # Final state - both cups in target
            "state_4": {
                "cam1.seed0.rgb": (example_rgb_path, "rgb"),
                "cam1.seed0.depth": (example_depth_path, "depth"),
            },
        }
        
        # Define objects in view
        objects_in_view = {env.cup1, env.cup2, env.table, env.target}
        
        # Process each state
        for state_id in ["state_0", "state_1", "state_2", "state_3", "state_4"]:
            # Determine objects in hand
            objects_in_hand = {env.cup1} if state_id == "state_1" else {env.cup2} if state_id == "state_3" else set()
                        
            # Add state
            creator.add_state_from_raw_images(
                test_state_images[state_id],
                state_id=state_id,
                objects_in_view=objects_in_view,
                objects_in_hand=objects_in_hand,
                gripper_open=(state_id not in ["state_1", "state_3"])
            )
            
            # Load and verify state
            loaded_state = creator.load_state(state_id)
            assert loaded_state is not None, f"Failed to load state {state_id}"
            assert loaded_state.images is not None, f"No images found in state {state_id}"
            
            # Verify loaded data matches original
            state_image_keys = [k.split("/")[1] for k in test_state_images.keys() if k.startswith(f"{state_id}/")]
            for image_key in state_image_keys:
                camera_id = image_key.split(".")[0]
                full_key = f"{camera_id}_{'.'.join(image_key.split('.')[1:])}"
                assert full_key in loaded_state.images, f"Missing image {full_key} in state {state_id}"
                loaded_img = loaded_state.images[full_key]
                assert loaded_img is not None, f"Image {full_key} is None in state {state_id}"
                if "rgb" in image_key:
                    assert loaded_img.rgb is not None
                    assert len(loaded_img.rgb.shape) == 3  # (H, W, 3)
                    assert loaded_img.rgb.dtype == np.uint8
                else:  # depth
                    assert loaded_img.depth is not None
                    assert len(loaded_img.depth.shape) == 2  # (H, W)
                    assert loaded_img.depth.dtype == np.float32
            
            # Verify metadata
            assert loaded_state.objects_in_view == objects_in_view
            if state_id == "state_1":
                assert loaded_state.objects_in_hand == {env.cup1}
                assert not loaded_state.gripper_open
            elif state_id == "state_3":
                assert loaded_state.objects_in_hand == {env.cup2}
                assert not loaded_state.gripper_open
            else:
                assert not loaded_state.objects_in_hand
                assert loaded_state.gripper_open
                
    finally:
        # Clean up temporary depth image
        if os.path.exists(example_depth_path):
            os.remove(example_depth_path)
    
    # Verify transition graph file exists
    graph_file = Path(test_dir) / "transitions" / f"Transition Graph, {env.name.replace('_', ' ').title()}.html"
    assert graph_file.exists(), "Transition graph file not generated"

def test_load_saved_two_cup_pick_place():
    """Test loading a previously saved two-cup pick-and-place task."""
    # Set up configuration with the same test directory
    test_name = "test_mock_two_cup_pick_place_manual_images"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir
    })
    
    # Create environment with same name to ensure object consistency
    env = MockSpotPickPlaceTwoCupEnv()
    
    # Create environment creator for loading
    creator = MockEnvCreatorBase(test_dir, env_info={
        "types": env.types,
        "predicates": env.predicates,
        "options": env.options,
        "nsrts": env.nsrts
    })
    
    # Initialize objects in creator
    for obj in env.objects:
        creator.objects[obj.name] = obj
    
    # Load and verify each state
    for state_id in ["state_0", "state_1", "state_2", "state_3", "state_4"]:
        # Load state
        loaded_state = creator.load_state(state_id)
        
        # Verify basic structure
        assert loaded_state.images, f"No views loaded for {state_id}"
        assert loaded_state.objects_in_view, f"No objects in view for {state_id}"
        
        # Verify expected objects are present
        expected_objects = {env.cup1, env.cup2, env.table, env.target}
        assert loaded_state.objects_in_view == expected_objects
        
        # Verify state-specific conditions
        if state_id == "state_1":
            assert loaded_state.objects_in_hand == {env.cup1}
            assert not loaded_state.gripper_open
        elif state_id == "state_3":
            assert loaded_state.objects_in_hand == {env.cup2}
            assert not loaded_state.gripper_open
        else:
            assert not loaded_state.objects_in_hand
            assert loaded_state.gripper_open 
