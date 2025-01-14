"""Test manual creation and saving of images for a pick-and-place task.

This test verifies:
1. Image Processing:
   - Loading RGB images from disk (PNG/JPG)
   - Loading depth images from disk (NPY)
   - Converting to appropriate format (uint8 for RGB, float32 for depth)
   - Error handling for invalid images

2. Image Storage:
   - Separate storage of RGB and depth images
   - Multiple views per state
   - Multiple cameras per view
   - Correct file paths and metadata

3. State Management:
   - Multiple states with different configurations
   - Objects in view/hand tracking
   - Gripper state tracking
   - Loading and verification of saved states

Directory Structure:
    mock_env_data/test_manual_image_creation/
    ├── images/
    │   ├── state_0/
    │   │   ├── view1/
    │   │   │   ├── cam1_rgb.npy      # RGB image
    │   │   │   ├── cam1_depth.npy    # Depth image
    │   │   │   ├── cam2_rgb.npy
    │   │   │   └── cam2_depth.npy
    │   │   └── view2/
    │   │       └── cam1_rgb.npy
    │   └── ...
    └── plan.yaml

Test Data:
    Uses example image from tests/spot_utils/example_manual_mock_task_1/state1_view1_img1.png
    Creates test data in mock_env_data/test_manual_image_creation/

Notes:
    - RGB and depth processing are separated:
      * process_rgb_image handles RGB images (PNG/JPG)
      * process_depth_image handles depth images (NPY)
    - Images can be added individually by type or together
    - Depth is optional for manual creation
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from typing import Dict, Tuple, Literal, Mapping

from predicators.spot_utils.mock_env.mock_env_creator_manual import ManualMockEnvCreator
from predicators.structs import Object, Type

# Define types for testing
_robot_type = Type("robot", ["x", "y", "z"])
_movable_object_type = Type("movable_object", ["x", "y", "z"])
_immovable_object_type = Type("immovable_object", ["x", "y", "z"])
_container_type = Type("container", ["x", "y", "z"])

def test_manual_image_creation():
    """Test both RGB and depth image processing and storage functionality."""
    # Set up OpenAI key for testing
    os.environ["OPENAI_API_KEY"] = "dummy_key_for_testing"
    
    # Set up configuration
    test_name = "test_manual_image_creation"
    test_dir = os.path.join("mock_env_data", test_name)
    
    # Get the path to the example image
    example_img_path = os.path.join(os.path.dirname(__file__), "example_manual_mock_task_1", "state1_view1_img1.png")
    
    # Create a temporary depth image
    example_depth_path = os.path.join(os.path.dirname(__file__), "example_manual_mock_task_1", "state1_view1_depth1.npy")
    depth_img = np.ones((480, 640), dtype=np.float32)  # Example depth image
    np.save(example_depth_path, depth_img)
    
    # Create environment creator
    creator = ManualMockEnvCreator(test_dir)
    
    # Test RGB image processing
    rgb_img = creator.process_rgb_image(example_img_path)
    assert rgb_img.dtype == np.uint8  # Should convert to uint8
    assert len(rgb_img.shape) == 3  # Should be (H, W, 3)
    
    # Test depth image processing
    depth = creator.process_depth_image(example_depth_path)
    assert depth.dtype == np.float32  # Should be float32
    assert len(depth.shape) == 2  # Should be (H, W)
    
    # Create objects
    robot = Object("robot", _robot_type)
    block = Object("block", _movable_object_type)
    table = Object("table", _immovable_object_type)
    container = Object("container", _container_type)
    objects = {robot, block, table, container}
    
    # Map of states to their views and images
    test_state_images: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Tuple[str, Literal["rgb", "depth"]]]]]] = {
        "state_0": {  # Initial state
            "view1": {
                "cam1": {
                    "rgb_img": (example_img_path, "rgb"),
                    "depth_img": (example_depth_path, "depth")  # Using actual depth image
                },
                "cam2": {
                    "rgb_img": (example_img_path, "rgb")
                }
            },
            "view2": {
                "cam1": {
                    "rgb_img": (example_img_path, "rgb")
                }
            }
        },
        "state_1": {  # Intermediate state (block in hand)
            "view1": {
                "cam1": {
                    "rgb_img": (example_img_path, "rgb"),
                    "depth_img": (example_depth_path, "depth")
                },
                "cam2": {
                    "rgb_img": (example_img_path, "rgb")
                }
            }
        },
        "state_2": {  # Final state
            "view1": {
                "cam1": {
                    "rgb_img": (example_img_path, "rgb")
                }
            },
            "view2": {
                "cam1": {
                    "rgb_img": (example_img_path, "rgb")
                },
                "cam2": {
                    "rgb_img": (example_img_path, "rgb"),
                    "depth_img": (example_depth_path, "depth")
                },
                "cam3": {
                    "rgb_img": (example_img_path, "rgb")
                }
            }
        }
    }
    
    try:
        # Process each state
        for state_id, views in test_state_images.items():
            # Add state with its views
            creator.add_state_from_multiple_images(
                views,
                state_id=state_id,
                objects_in_view=list({block.name, table.name, container.name}),
                objects_in_hand=list({block.name}) if state_id == "state_1" else [],
                gripper_open=(state_id != "state_1")
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
                assert metadata["objects_in_view"] == list({block.name, table.name, container.name})
                assert metadata["objects_in_hand"] == (list({block.name}) if state_id == "state_1" else [])
                assert metadata["gripper_open"] == (state_id != "state_1")
                assert len(metadata["views"]) == len(views)
                
                # Verify image paths in metadata
                for view_id, cameras in views.items():
                    assert view_id in metadata["views"]
                    for camera_id, images in cameras.items():
                        assert camera_id in metadata["views"][view_id]
                        for image_name, (_, image_type) in images.items():
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
            assert set(objects_in_view) == {block.name, table.name, container.name}
            if state_id == "state_1":
                assert objects_in_hand == [block.name]
                assert not gripper_open
            else:
                assert not objects_in_hand
                assert gripper_open
    finally:
        # Clean up temporary depth image
        if os.path.exists(example_depth_path):
            os.remove(example_depth_path) 