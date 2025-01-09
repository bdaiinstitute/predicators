"""Manual creator for mock environment data."""

import os
from typing import List, Optional

import numpy as np
from bosdyn.client.math_helpers import SE3Pose, Quat

from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext
from predicators.spot_utils.mock_env.mock_env_creator_base import MockEnvCreatorBase


class ManualMockEnvCreator(MockEnvCreatorBase):
    """Manual creator for mock environment data."""

    def create_rgbd_image(self, rgb: np.ndarray, depth: np.ndarray,
                         camera_name: str = "hand_color") -> RGBDImageWithContext:
        """Create an RGBDImageWithContext from RGB and depth arrays.
        
        Args:
            rgb: RGB image array
            depth: Depth image array
            camera_name: Name of camera that captured the image
            
        Returns:
            RGBDImageWithContext instance
        """
        # Create a default transform for manual images
        world_tform_camera = SE3Pose.from_identity()
        
        return RGBDImageWithContext(
            rgb=rgb,
            depth=depth,
            image_rot=0.0,
            camera_name=camera_name,
            world_tform_camera=world_tform_camera,
            depth_scale=1.0,
            transforms_snapshot=None,  # Not needed for manual images
            frame_name_image_sensor=camera_name,
            camera_model=None  # Not needed for manual images
        )

    def add_state_from_images(self, rgb_path: str,
                            objects_in_view: Optional[List[str]] = None,
                            objects_in_hand: Optional[List[str]] = None,
                            gripper_open: bool = True) -> None:
        """Add a state to the environment from image files.
        
        Args:
            rgb_path: Path to RGB image file
            objects_in_view: List of object names in view
            objects_in_hand: List of object names in hand
            gripper_open: Whether the gripper is open
            
        Raises:
            RuntimeError: If unable to load images
        """
        try:
            # Load RGB image
            rgb = np.load(rgb_path)
            
            # Create default depth image
            depth = np.zeros_like(rgb[:, :, 0], dtype=np.uint16)
            
            # Create RGBDImageWithContext
            rgbd = self.create_rgbd_image(rgb, depth)
            
            # Add state
            state_id = str(len(os.listdir(self.image_dir)))
            self.add_state(state_id, rgbd, objects_in_view, objects_in_hand, gripper_open)
            
        except (IOError, ValueError) as e:
            raise RuntimeError(f"Failed to load image from {rgb_path}: {str(e)}") 