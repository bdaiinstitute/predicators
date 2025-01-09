"""Test cases for the mock Spot perceiver.

This module contains tests for the MockSpotPerceiver class, which provides
simulated perception capabilities for testing robot behaviors without
requiring the actual Spot robot hardware.

The tests verify:
1. Basic initialization and state management
2. RGBD image handling with camera context
3. Object tracking (in view and in hand)
4. Gripper state management
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from bosdyn.api.geometry_pb2 import FrameTreeSnapshot
from bosdyn.client.math_helpers import SE3Pose
from predicators import utils
from predicators.perception.mock_spot_perceiver import MockSpotPerceiver
from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext


def test_mock_spot_perceiver():
    """Tests for the MockSpotPerceiver class.
    
    This test function verifies the core functionality of the mock perceiver:
    
    1. Initialization:
       - Creates a temporary directory for image storage
       - Initializes perceiver with default empty state
    
    2. Initial State Verification:
       - Confirms perceiver name is correct
       - Verifies default state (no images, empty object sets)
    
    3. Image Handling:
       - Creates a mock RGBD image with camera context
       - Verifies image storage and retrieval
       - Checks image dimensions and properties
    
    4. State Management:
       - Updates environment state with new objects
       - Verifies gripper state changes
       - Confirms object tracking in view and in hand
    
    The test uses a temporary directory for image storage which is
    automatically cleaned up after the test completes.
    """
    # Set up configuration for the mock environment
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
    })

    # Create a temporary directory for test image storage
    temp_dir = tempfile.mkdtemp()
    try:
        # Initialize perceiver with temporary directory
        perceiver = MockSpotPerceiver(data_dir=temp_dir)
        
        # Test 1: Verify basic properties
        assert perceiver.get_name() == "mock_spot"
        
        # Test 2: Verify initial state
        obs = perceiver.get_observation()
        assert obs.rgbd is None  # No images initially
        assert obs.gripper_open  # Gripper starts open
        assert not obs.objects_in_view  # No objects in view
        assert not obs.objects_in_hand  # No objects in hand
        
        # Test 3: Create and verify mock RGBD image
        mock_rgbd = RGBDImageWithContext(
            # Create empty RGB image (100x100 pixels)
            rgb=np.zeros((100, 100, 3), dtype=np.uint8),
            # Create empty depth image (100x100 pixels)
            depth=np.zeros((100, 100), dtype=np.uint16),
            camera_name="mock_camera",
            image_rot=0.0,  # No rotation
            # Identity pose for camera
            world_tform_camera=SE3Pose(x=0.0, y=0.0, z=0.0, rot=np.eye(3)),
            depth_scale=1.0,
            transforms_snapshot=FrameTreeSnapshot(),
            frame_name_image_sensor="mock_camera",
            camera_model=None  # No camera model needed for mock
        )
        perceiver.save_image(mock_rgbd)
        
        # Verify image was saved and can be retrieved
        obs = perceiver.get_observation()
        assert obs.rgbd is not None
        assert obs.rgbd.rgb.shape == (100, 100, 3)  # Check RGB dimensions
        assert obs.rgbd.depth.shape == (100, 100)   # Check depth dimensions
        
        # Test 4: Update and verify environment state
        perceiver.update_state(
            gripper_open=False,  # Close gripper
            objects_in_view={"cup", "table"},  # Add objects to view
            objects_in_hand={"cup"}  # Add object to hand
        )
        obs = perceiver.get_observation()
        assert not obs.gripper_open  # Verify gripper closed
        assert obs.objects_in_view == {"cup", "table"}  # Verify visible objects
        assert obs.objects_in_hand == {"cup"}  # Verify held objects
        
    finally:
        # Clean up: Remove temporary directory and contents
        shutil.rmtree(temp_dir) 