"""Manual creator for mock environment data."""

import os
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import matplotlib.pyplot as plt

from predicators.spot_utils.mock_env.mock_env_creator_base import MockEnvCreatorBase


class ManualMockEnvCreator(MockEnvCreatorBase):
    """Manual creator for mock environment data.
    
    This class provides functionality to:
    - Create mock environments from manually collected images
    - Add states with RGB-D observations
    - Define transitions between states
    - Support belief space planning when enabled in environment
    
    The environment data is stored in a directory specified by CFG.mock_env_data_dir,
    which is set during initialization. This includes:
    - Plan data (plan.yaml)
    - RGB-D images for each state (images/)
        - state0/
            - view1/
                - cam1_rgb.npy
                - cam1_depth.npy
                - cam2_rgb.npy
                - cam2_depth.npy
            - view2/
                - cam1_rgb.npy
                - cam1_depth.npy
        - state1/
            ...
    - Observation metadata (gripper state, objects in view/hand)
    - Transition graph visualization (transitions/)
    """

    def create_rgbd_image(self, rgb: np.ndarray, depth: np.ndarray,
                         camera_name: str = "hand_color") -> None:
        """Create an RGBDImageWithContext from RGB and depth arrays.
        
        This is a simplified version that doesn't require RGBDImageWithContext.
        The method is required by the base class but not used in this implementation.
        
        Args:
            rgb: RGB image array
            depth: Depth image array
            camera_name: Name of camera that captured the image
            
        Returns:
            None since we don't use RGBDImageWithContext in this implementation
        """
        return None  # Not used in this implementation

    def process_rgb_image(self, image_path: str) -> np.ndarray:
        """Process an RGB image file.
        
        Args:
            image_path: Path to RGB image file
            
        Returns:
            RGB array (H, W, 3) in uint8 format
            
        Raises:
            RuntimeError: If image loading fails
        """
        try:
            rgb = plt.imread(image_path)
            if rgb.dtype == np.float32:
                rgb = (rgb * 255).astype(np.uint8)
            return rgb
        except Exception as e:
            raise RuntimeError(f"Failed to load RGB image from {image_path}: {e}")

    def process_depth_image(self, image_path: str) -> np.ndarray:
        """Process a depth image file.
        
        Args:
            image_path: Path to depth image file
            
        Returns:
            Depth array (H, W) in float32 format
            
        Raises:
            RuntimeError: If image loading fails
        """
        try:
            depth = np.load(image_path)  # Assuming depth is saved as .npy
            if depth.dtype != np.float32:
                depth = depth.astype(np.float32)
            return depth
        except Exception as e:
            raise RuntimeError(f"Failed to load depth image from {image_path}: {e}")

    def add_state_from_images(self, 
                            image_path: str,
                            image_type: Literal["rgb", "depth"],
                            view_name: str = "view1",
                            camera_name: str = "cam1",
                            objects_in_view: Optional[List[str]] = None,
                            objects_in_hand: Optional[List[str]] = None,
                            gripper_open: bool = True) -> None:
        """Add a state to the environment from image files.
        
        Args:
            image_path: Path to image file
            image_type: Type of image ("rgb" or "depth")
            view_name: Name of the view (default: "view1")
            camera_name: Name of the camera (default: "cam1")
            objects_in_view: List of object names visible in the image
            objects_in_hand: List of object names being held
            gripper_open: Whether the gripper is open
            
        Raises:
            RuntimeError: If unable to load images
        """
        # Process image based on type
        if image_type == "rgb":
            image = self.process_rgb_image(image_path)
        else:  # depth
            image = self.process_depth_image(image_path)
        
        # Create views dict with camera structure
        views = {
            view_name: {
                camera_name: {
                    image_type: image
                }
            }
        }
        
        # Add state
        state_id = str(len(os.listdir(self.image_dir)))
        self.add_state(state_id, views, objects_in_view, objects_in_hand, gripper_open)

    def add_state_from_multiple_images(
        self,
        views: Dict[str, Dict[str, Dict[str, Tuple[str, Literal["rgb", "depth"]]]]],
        state_id: Optional[str] = None,
        objects_in_view: Optional[List[str]] = None,
        objects_in_hand: Optional[List[str]] = None,
        gripper_open: bool = True
    ) -> None:
        """Add a state from multiple image files.
        
        Args:
            views: Dictionary mapping view_id -> camera_id -> image_name -> (path, type)
                Example:
                {
                    "view1": {
                        "cam1": {
                            "image1": ("path/to/rgb.png", "rgb"),
                            "image2": ("path/to/depth.npy", "depth")
                        }
                    }
                }
            state_id: Optional state ID. If None, will be auto-generated.
            objects_in_view: List of object names visible in the state
            objects_in_hand: List of object names being held
            gripper_open: Whether the gripper is open
            
        Raises:
            RuntimeError: If image loading fails
        """
        # Process images for each view and camera
        processed_views = {}
        for view_id, cameras in views.items():
            processed_views[view_id] = {}
            for camera_id, images in cameras.items():
                processed_views[view_id][camera_id] = {}
                for image_name, (image_path, image_type) in images.items():
                    # Process image based on type
                    if image_type == "rgb":
                        processed_views[view_id][camera_id][image_name] = self.process_rgb_image(image_path)
                    else:  # depth
                        processed_views[view_id][camera_id][image_name] = self.process_depth_image(image_path)
        
        # Generate state ID if not provided
        if state_id is None:
            state_id = f"state_{len(self.states)}"
        
        # Add the processed state
        self.add_state(
            views=processed_views,
            state_id=state_id,
            objects_in_view=objects_in_view,
            objects_in_hand=objects_in_hand,
            gripper_open=gripper_open
        ) 