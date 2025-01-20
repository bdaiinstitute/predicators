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
   - Multiple views per state
   - Multiple cameras per view
   - Proper metadata storage

Directory Structure:
    mock_env_data/test_two_cup_pick_place/
    ├── images/
    │   ├── state_0/
    │   │   ├── view1/
    │   │   │   ├── cam1_rgb.npy
    │   │   │   ├── cam1_depth.npy
    │   │   │   ├── cam2_rgb.npy
    │   │   │   └── cam2_depth.npy
    │   │   └── view2/
    │   │       └── cam1_rgb.npy
    │   └── ...
    ├── transitions/
    │   └── Transition Graph, Test Two Cup Pick Place.html
    └── plan.yaml
"""

import os
import numpy as np
import pytest
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
from typing import Dict, Tuple, Literal

from predicators.envs.mock_spot_env import (
    MockSpotPickPlaceTwoCupEnv, _robot_type, _container_type, _immovable_object_type,
    _HandEmpty, _NotHolding, _On, _NotBlocked, _Inside, _IsPlaceable, 
    _HasFlatTopSurface, _FitsInXY, _NotInsideAnyContainer, _Reachable,
    _InHandView, _NEq
)
from predicators.structs import Object, GroundAtom
from predicators.spot_utils.mock_env.mock_env_creator_manual import ManualMockEnvCreator
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
    test_name = "test_two_cup_pick_place"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir
    })
    
    # Create pick-place specific environment with test name
    env = MockSpotPickPlaceTwoCupEnv(name=test_name)
    
    # Create environment creator with pick-place specific env
    creator = ManualMockEnvCreator(test_dir, env=env)
    
    # Plan and visualize transitions
    name = f'Transition Graph, {env.name.replace("_", " ").title()}'
    creator.plan_and_visualize(env.initial_atoms, env.goal_atoms, env.objects, task_name=name)
    
    # Get example image paths
    example_rgb_path = os.path.join(os.path.dirname(__file__), 
                                  "example_manual_mock_task_1", 
                                  "state1_view1_img1.png")
    
    # Create temporary depth image
    example_depth_path = os.path.join(os.path.dirname(__file__), 
                                    "example_manual_mock_task_1", 
                                    "state1_view1_depth1.npy")
    depth_img = np.ones((480, 640), dtype=np.float32)
    np.save(example_depth_path, depth_img)
    
    try:
        # Map of states to their views and images
        test_state_images: Dict[str, Dict[str, Dict[str, Dict[str, Tuple[str, Literal["rgb", "depth"]]]]]] = {
            "state_0": {  # Initial state - both cups on table
                "view1": {
                    "cam1": {
                        "rgb_img": (example_rgb_path, "rgb"),
                        "depth_img": (example_depth_path, "depth")
                    },
                    "cam2": {
                        "rgb_img": (example_rgb_path, "rgb")
                    }
                },
                "view2": {
                    "cam1": {
                        "rgb_img": (example_rgb_path, "rgb")
                    }
                }
            },
            "state_1": {  # First cup in hand
                "view1": {
                    "cam1": {
                        "rgb_img": (example_rgb_path, "rgb"),
                        "depth_img": (example_depth_path, "depth")
                    }
                }
            },
            "state_2": {  # First cup in target
                "view1": {
                    "cam1": {
                        "rgb_img": (example_rgb_path, "rgb")
                    },
                    "cam2": {
                        "rgb_img": (example_rgb_path, "rgb")
                    }
                }
            },
            "state_3": {  # Second cup in hand
                "view1": {
                    "cam1": {
                        "rgb_img": (example_rgb_path, "rgb"),
                        "depth_img": (example_depth_path, "depth")
                    }
                }
            },
            "state_4": {  # Final state - both cups in target
                "view1": {
                    "cam1": {
                        "rgb_img": (example_rgb_path, "rgb")
                    }
                }
            }
        }
        
        # Process each state
        for state_id, views in test_state_images.items():
            # Add state with its views
            creator.add_state_from_multiple_images(
                views,
                state_id=state_id,
                objects_in_view=list({env.cup1.name, env.cup2.name, env.table.name, env.target.name}),
                objects_in_hand=[env.cup1.name] if state_id == "state_1" else 
                               [env.cup2.name] if state_id == "state_3" else [],
                gripper_open=(state_id not in ["state_1", "state_3"])
            )
            
            # Verify files exist
            state_dir = os.path.join(creator.image_dir, state_id)
            for view_id, cameras in views.items():
                view_dir = os.path.join(state_dir, view_id)
                for camera_id, images in cameras.items():
                    # Check each image type exists
                    for image_name, (_, image_type) in images.items():
                        image_path = os.path.join(view_dir, f"{camera_id}_{image_name}.npy")
                        assert os.path.exists(image_path)
                        # Load and verify image data
                        saved_img = np.load(image_path)
                        if image_type == "rgb":
                            assert len(saved_img.shape) == 3  # (H, W, 3)
                            assert saved_img.dtype == np.uint8
                        else:  # depth
                            assert len(saved_img.shape) == 2  # (H, W)
                            assert saved_img.dtype == np.float32
            
            # Verify metadata
            metadata_path = os.path.join(state_dir, "metadata.yaml")
            assert os.path.exists(metadata_path)
            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f)
                assert metadata["objects_in_view"] == list({env.cup1.name, env.cup2.name, env.table.name, env.target.name})
                assert metadata["objects_in_hand"] == ([env.cup1.name] if state_id == "state_1" else 
                                                     [env.cup2.name] if state_id == "state_3" else [])
                assert metadata["gripper_open"] == (state_id not in ["state_1", "state_3"])
                
                # Verify image paths in metadata
                for view_id, cameras in views.items():
                    assert view_id in metadata["views"]
                    for camera_id, images in cameras.items():
                        assert camera_id in metadata["views"][view_id]
                        for image_name, (_, _) in images.items():
                            path_key = f"{image_name}_path"
                            assert path_key in metadata["views"][view_id][camera_id]
                            expected_path = os.path.join(view_id, f"{camera_id}_{image_name}.npy")
                            assert metadata["views"][view_id][camera_id][path_key] == expected_path
            
            # Load and verify state
            loaded_views, objects_in_view, objects_in_hand, gripper_open = creator.load_state(state_id)
            
            # Verify loaded data matches original
            for view_id, cameras in views.items():
                assert view_id in loaded_views
                for camera_id, images in cameras.items():
                    assert camera_id in loaded_views[view_id]
                    for image_name, (_, image_type) in images.items():
                        assert image_name in loaded_views[view_id][camera_id]
                        loaded_img = loaded_views[view_id][camera_id][image_name]
                        if image_type == "rgb":
                            assert len(loaded_img.shape) == 3  # (H, W, 3)
                            assert loaded_img.dtype == np.uint8
                        else:  # depth
                            assert len(loaded_img.shape) == 2  # (H, W)
                            assert loaded_img.dtype == np.float32
            
            # Verify metadata
            assert set(objects_in_view) == {env.cup1.name, env.cup2.name, env.table.name, env.target.name}
            if state_id == "state_1":
                assert objects_in_hand == [env.cup1.name]
                assert not gripper_open
            elif state_id == "state_3":
                assert objects_in_hand == [env.cup2.name]
                assert not gripper_open
            else:
                assert not objects_in_hand
                assert gripper_open
                
    finally:
        # Clean up temporary depth image
        if os.path.exists(example_depth_path):
            os.remove(example_depth_path)
    
    # Verify transition graph file exists
    graph_file = Path(test_dir) / "transitions" / f"{name}.html"
    assert graph_file.exists(), "Transition graph file not generated" 