"""Mock perceiver for the Spot environment.

This perceiver simulates perception for the mock Spot environment by providing a simplified
interface for testing and development without requiring the actual Spot robot hardware.

Key Features:
1. RGBD Image Management:
   - Stores and returns mock RGBD images with camera context
   - Supports image rotation and transformation data
   - Maintains depth information for 3D perception

2. State Tracking:
   - Objects currently in the robot's view
   - Objects currently held in the gripper
   - Gripper state (open/closed)

3. Mock Environment Integration:
   - Works with MockSpotEnv for end-to-end testing
   - Supports transition verification in the mock environment
   - Enables testing of perception-dependent behaviors

Usage:
    perceiver = MockSpotPerceiver(data_dir="/path/to/images")
    
    # Update state based on environment changes
    perceiver.update_state(
        gripper_open=True,
        objects_in_view={"cup", "table"},
        objects_in_hand=set()
    )
    
    # Get current observation
    obs = perceiver.get_observation()
    assert obs.gripper_open
    assert "cup" in obs.objects_in_view
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

import numpy as np

from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext


class MockSpotPerceiver:
    """A mock perceiver for the Spot environment that returns simulated observations.
    
    This class provides a simplified interface for testing perception-dependent behaviors
    without requiring the actual Spot robot. It maintains the current state of the
    environment including:
    - Current RGBD image with camera context
    - Objects visible to the robot
    - Objects held in the gripper
    - Gripper state (open/closed)
    
    The perceiver is designed to work with MockSpotEnv for testing planning and
    manipulation strategies that depend on visual perception.
    """

    def __init__(self, data_dir: str) -> None:
        """Initialize the mock perceiver.
        
        Args:
            data_dir: Directory to store/load mock images. This directory should contain
                     the RGBD images that will be used during testing. Images can be
                     saved here using the save_image method.
        """
        self._data_dir = Path(data_dir)
        self._current_rgbd: Optional[RGBDImageWithContext] = None
        self._gripper_open: bool = True
        self._objects_in_view: Set[str] = set()
        self._objects_in_hand: Set[str] = set()

    def get_name(self) -> str:
        """Get the name of this perceiver.
        
        Returns:
            The string identifier "mock_spot" for this perceiver type.
        """
        return "mock_spot"

    def get_observation(self) -> "MockSpotObservation":
        """Get the current observation of the environment.
        
        This method returns a MockSpotObservation containing:
        - The current RGBD image with camera context (if any)
        - Current gripper state
        - Set of objects currently visible
        - Set of objects currently held
        
        Returns:
            A MockSpotObservation containing the current state.
        """
        return MockSpotObservation(
            rgbd=self._current_rgbd,
            gripper_open=self._gripper_open,
            objects_in_view=self._objects_in_view,
            objects_in_hand=self._objects_in_hand
        )

    def save_image(self, rgbd: RGBDImageWithContext) -> None:
        """Save a mock RGBD image to be returned in future observations.
        
        Args:
            rgbd: The RGBD image with context to save. This should include:
                - RGB and depth image arrays
                - Camera pose information
                - Camera intrinsics and parameters
                - Frame transformation data
        """
        self._current_rgbd = rgbd

    def update_state(self, gripper_open: bool, objects_in_view: Set[str],
                    objects_in_hand: Set[str]) -> None:
        """Update the mock environment state.
        
        This method allows updating the full state of the environment at once,
        which is useful for testing different scenarios and transitions.
        
        Args:
            gripper_open: Whether the gripper is open (True) or closed (False)
            objects_in_view: Complete set of objects currently visible to the robot.
                           Previous objects not in this set will be removed from view.
            objects_in_hand: Complete set of objects currently held by the gripper.
                           Previous objects not in this set will be removed from hand.
        """
        self._gripper_open = gripper_open
        self._objects_in_view = objects_in_view
        self._objects_in_hand = objects_in_hand


@dataclass
class MockSpotObservation:
    """An observation from the mock Spot environment.
    
    This dataclass encapsulates all the information that would normally be
    perceived by the real Spot robot's sensors, including:
    
    Attributes:
        rgbd: Optional RGBD image with camera context. None if no image is available.
        gripper_open: Boolean indicating if the gripper is currently open.
        objects_in_view: Set of string identifiers for objects currently visible.
        objects_in_hand: Set of string identifiers for objects currently held.
    """
    rgbd: Optional[RGBDImageWithContext]
    gripper_open: bool
    objects_in_view: Set[str]
    objects_in_hand: Set[str]
