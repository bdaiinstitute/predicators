"""Spot robot creator for mock environment data."""

import os
from typing import List, Optional, Tuple, Any, cast

import numpy as np
from bosdyn.client import create_standard_sdk, math_helpers
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.client.util import authenticate
from bosdyn.api import robot_state_pb2, image_pb2

from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext
from predicators.spot_utils.mock_env.mock_env_creator_base import MockEnvCreatorBase


def get_spot_clients(hostname: str) -> Tuple[RobotCommandClient, RobotStateClient, ImageClient]:
    """Get Spot robot clients.
    
    Args:
        hostname: Hostname or IP of Spot robot
        
    Returns:
        robot_command_client: Client for sending commands to robot
        robot_state_client: Client for getting robot state
        image_client: Client for getting images from robot
    """
    # Create SDK
    sdk = create_standard_sdk('MockEnvCreator')
    robot = sdk.create_robot(hostname)
    
    # Authenticate robot
    authenticate(robot)
    
    # Get clients
    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    
    return robot_command_client, robot_state_client, image_client


class SpotMockEnvCreator(MockEnvCreatorBase):
    """Spot robot creator for mock environment data."""

    def __init__(self, hostname: str, path_dir: str) -> None:
        """Initialize the creator.
        
        Args:
            hostname: Hostname or IP of Spot robot
            path_dir: Directory to store environment data
        """
        super().__init__(path_dir)
        
        # Get Spot clients
        self._robot_command_client, self._robot_state_client, self._image_client = \
            get_spot_clients(hostname)

    def create_rgbd_image(self, rgb: np.ndarray, depth: np.ndarray,
                         camera_name: str = "hand_color") -> RGBDImageWithContext:
        """Create an RGBDImageWithContext from RGB and depth arrays.
        
        Args:
            rgb: RGB image array
            depth: Depth image array
            camera_name: Name of camera that captured the image
            
        Returns:
            RGBDImageWithContext instance
            
        Raises:
            RuntimeError: If unable to get robot state or camera transform
        """
        # Get current robot state for camera transform
        robot_state = cast(Any, self._robot_state_client.get_robot_state())
        if not robot_state:
            raise RuntimeError("Failed to get robot state")
            
        # Get transform snapshot from robot state
        transforms_snapshot = getattr(robot_state, 'transforms_snapshot', None)
        if transforms_snapshot is None:
            raise RuntimeError("Failed to get transforms snapshot")
        
        # Get camera transform
        world_tform_camera = transforms_snapshot.get_se3_from_frame_to_frame(
            "vision", camera_name)
        if world_tform_camera is None:
            raise RuntimeError(f"Failed to get transform from vision to {camera_name}")
        
        return RGBDImageWithContext(
            rgb=rgb,
            depth=depth,
            image_rot=0.0,
            camera_name=camera_name,
            world_tform_camera=world_tform_camera,
            depth_scale=1.0,
            transforms_snapshot=transforms_snapshot,
            frame_name_image_sensor=camera_name,
            camera_model=None  # TODO: Add proper camera model
        )

    def add_state_from_robot(self,
                           objects_in_view: Optional[List[str]] = None,
                           objects_in_hand: Optional[List[str]] = None,
                           gripper_open: bool = True) -> None:
        """Add a state to the environment from robot cameras.
        
        Args:
            objects_in_view: List of object names in view
            objects_in_hand: List of object names in hand
            gripper_open: Whether the gripper is open
            
        Raises:
            RuntimeError: If unable to capture images from robot
        """
        # Get RGB and depth images from robot
        rgb, depth = self._capture_images()
        
        # Create RGBDImageWithContext
        rgbd = self.create_rgbd_image(rgb, depth)
        
        # Add state
        state_id = str(len(os.listdir(self.image_dir)))
        self.add_state(state_id, rgbd, objects_in_view, objects_in_hand, gripper_open)

    def _capture_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture RGB and depth images from robot cameras.
        
        Returns:
            rgb: RGB image array
            depth: Depth image array
            
        Raises:
            RuntimeError: If unable to get images from robot
        """
        # Get RGB and depth images
        image_responses = self._image_client.get_image_from_sources(["hand_color", "hand_depth"])
        if not image_responses:
            raise RuntimeError("Failed to get image responses from robot")
            
        if len(image_responses) < 2:
            raise RuntimeError("Did not receive both RGB and depth images")
            
        # Extract RGB image
        rgb_response = cast(Any, image_responses[0])
        if not hasattr(rgb_response, 'shot') or not hasattr(rgb_response.shot, 'image'):
            raise RuntimeError("Invalid RGB image response format")
            
        if not rgb_response.shot.image.rows or not rgb_response.shot.image.cols:
            raise RuntimeError("Invalid RGB image dimensions")
            
        rgb = np.frombuffer(rgb_response.shot.image.data, dtype=np.uint8)
        rgb = rgb.reshape(rgb_response.shot.image.rows, rgb_response.shot.image.cols, 3)
        
        # Extract depth image
        depth_response = cast(Any, image_responses[1])
        if not hasattr(depth_response, 'shot') or not hasattr(depth_response.shot, 'image'):
            raise RuntimeError("Invalid depth image response format")
            
        if not depth_response.shot.image.rows or not depth_response.shot.image.cols:
            raise RuntimeError("Invalid depth image dimensions")
            
        depth = np.frombuffer(depth_response.shot.image.data, dtype=np.uint16)
        depth = depth.reshape(depth_response.shot.image.rows, depth_response.shot.image.cols)
        
        return rgb, depth 