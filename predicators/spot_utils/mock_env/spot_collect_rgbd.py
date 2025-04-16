#!/usr/bin/env python3

"""
RGB-D data collection example for Spot robot's hand camera.
Captures RGB and depth images along with camera poses in the vision frame.
"""

import argparse
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import bosdyn.client
import bosdyn.client.util
import cv2
import numpy as np
from bosdyn.client import Robot
from bosdyn.client.frame_helpers import get_a_tform_b
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    block_until_arm_arrives,
)
from rich import pretty, print
from rich.console import Console
from rich.progress import track

pretty.install()
console = Console()


@dataclass
class DataCollectionConfig:
    """Configuration for data collection."""
    manual_images: bool = True
    path_dir: str = "../data"
    dir_name: str = "Spot"
    visualize: bool = False
    robot_pose: bool = True
    pic_hz: float = 2.0
    movement_threshold: float = 0.2
    sources: List[str] = ["hand_depth_in_hand_color_frame", "hand_color_image"]


@dataclass
class PoseData:
    """Container for pose information."""
    position: List[float]  # [x, y, z]
    quaternion: List[float]  # [w, x, y, z]
    rotation_matrix: Optional[np.ndarray] = None
    rpy: Optional[List[float]] = None  # [roll, pitch, yaw]
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for saving."""
        return np.array(self.position + self.quaternion)


def verify_credentials() -> Tuple[str, str]:
    """Verify robot credentials are set.
    
    Returns:
        Tuple of username and password.
        
    Raises:
        ValueError: If credentials are not set.
    """
    username = os.environ.get("BOSDYN_CLIENT_USERNAME")
    password = os.environ.get("BOSDYN_CLIENT_PASSWORD")
    
    if not username or not password:
        raise ValueError(
            "Robot credentials not set. Please set BOSDYN_CLIENT_USERNAME "
            "and BOSDYN_CLIENT_PASSWORD environment variables."
        )
    console.print(f"[green]Using credentials[/green]: username={username}")
    return username, password


@contextmanager
def robot_connection(hostname: str) -> Generator[Robot, None, None]:
    """Context manager for robot connection.
    
    Args:
        hostname: Robot's hostname or IP address.
        
    Yields:
        Connected robot instance.
    """
    sdk = bosdyn.client.create_standard_sdk("image_depth_plus_visual")
    robot = sdk.create_robot(hostname)
    try:
        bosdyn.client.util.authenticate(robot)
        yield robot
    finally:
        robot.disconnect()


def change_gripper(
    robot: Robot,
    fraction: float,
    duration: float = 2.0,
) -> None:
    """Change the Spot gripper angle.
    
    Args:
        robot: Initialized Robot instance
        fraction: Gripper opening fraction (0.0=closed, 1.0=open)
        duration: Time to wait for gripper movement (seconds)
    
    Raises:
        bosdyn.client.robot_command.RobotCommandError: If command fails
    """
    assert 0.0 <= fraction <= 1.0, "Fraction must be between 0 and 1"

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    robot.time_sync.wait_for_sync()

    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    # Build the command.
    cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(fraction)
    # Send the request.
    cmd_id = robot_command_client.robot_command(cmd)
    # Wait until the arm arrives at the goal.
    block_until_arm_arrives(robot_command_client, cmd_id, duration)


def setup_data_directories(config: DataCollectionConfig) -> Path:
    """Create necessary directories for data storage.
    
    Args:
        config: Data collection configuration
        
    Returns:
        Path to base data directory
    """
    base_dir = Path(config.path_dir) / config.dir_name
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for subdir in ["Pose", "RGB", "Depth", "Visualized"]:
        (base_dir / subdir).mkdir(exist_ok=True)
    
    return base_dir


def process_depth_image(depth_data: bytes, shape: Tuple[int, int]) -> np.ndarray:
    """Process raw depth data into meters.
    
    Args:
        depth_data: Raw depth data bytes
        shape: (height, width) tuple
        
    Returns:
        Depth image in meters as float32 array
    """
    depth = np.frombuffer(depth_data, dtype=np.uint16)
    depth = depth.reshape(shape)
    return depth.astype(np.float32) / 1000.0


def create_rgbd_overlay(
    rgb_img: np.ndarray,
    depth_img: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """Create RGB-D overlay visualization.
    
    Args:
        rgb_img: RGB image array
        depth_img: Depth image array (in meters)
        alpha: Blend factor for overlay
        
    Returns:
        Blended RGB-D visualization
    """
    # Convert depth to color
    depth_min = np.min(depth_img)
    depth_max = np.max(depth_img)
    depth_range = depth_max - depth_min
    
    depth8 = ((depth_img - depth_min) * 255.0 / depth_range).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth8, cv2.COLORMAP_JET)
    
    # Ensure RGB image is in correct format
    rgb = rgb_img if len(rgb_img.shape) == 3 else cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)
    
    # Blend images
    return cv2.addWeighted(rgb, alpha, depth_color, 1 - alpha, 0)


def save_frame_data(
    base_dir: Path,
    counter: int,
    rgb_img: np.ndarray,
    depth_img: np.ndarray,
    pose_data: Optional[PoseData] = None,
    save_visualization: bool = True,
) -> None:
    """Save all data for a single frame.
    
    Args:
        base_dir: Base directory for data storage
        counter: Frame counter
        rgb_img: RGB image array
        depth_img: Depth image array
        pose_data: Optional pose data
        save_visualization: Whether to save visualization images
    """
    # Save raw data
    np.save(base_dir / "RGB" / f"rgb_img_{counter}.npy", rgb_img)
    np.save(base_dir / "Depth" / f"depth_img_{counter}.npy", depth_img)
    
    if pose_data is not None:
        np.save(base_dir / "Pose" / f"pose_{counter}.npy", pose_data.to_array())
    
    if save_visualization:
        # Save RGB image
        cv2.imwrite(
            str(base_dir / "Visualized" / f"RGB_{counter}.jpg"),
            cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        )
        
        # Save RGB-D overlay
        overlay = create_rgbd_overlay(rgb_img, depth_img)
        cv2.imwrite(
            str(base_dir / "Visualized" / f"RGB-D_{counter}.jpg"),
            overlay
        )


def main(argv):
    """Main function for RGB-D data collection."""
    # Parse arguments
    parser = argparse.ArgumentParser(description=__doc__)
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument(
        "--manual_images",
        help="Whether images are taken manually or continuously",
        default="True",
    )
    parser.add_argument(
        "--path_dir",
        help="path to directory where data is saved",
        default="../data"
    )
    parser.add_argument(
        "--dir_name",
        help="name of directory for data storage",
        default="Spot",
    )
    options = parser.parse_args(argv)

    # Create configuration
    config = DataCollectionConfig(
        manual_images=options.manual_images.lower() == "true",
        path_dir=options.path_dir,
        dir_name=options.dir_name,
    )

    # Verify credentials and setup directories
    verify_credentials()
    base_dir = setup_data_directories(config)

    # Connect to robot
    with robot_connection(options.hostname) as robot:
        # Initialize clients
        image_client = robot.ensure_client(ImageClient.default_service_name)
        
        # Main collection loop
        counter = 0
        last_pose = None
        
        try:
            while True:
                if config.manual_images:
                    if input("Take image [y/n]").lower() != "y":
                        break
                else:
                    time.sleep(1.0 / config.pic_hz)

                # Capture images
                image_responses = image_client.get_image_from_sources(config.sources)
                
                if len(image_responses) < 2:
                    console.print("[red]Error:[/] Failed to get images")
                    continue

                # Process depth image
                depth_img = process_depth_image(
                    image_responses[0].shot.image.data,
                    (
                        image_responses[0].shot.image.rows,
                        image_responses[0].shot.image.cols
                    )
                )

                # Process RGB image
                rgb_img = cv2.imdecode(
                    np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8),
                    -1
                )

                # Get pose if enabled
                pose_data = None
                if config.robot_pose:
                    vision_tform_hand = get_a_tform_b(
                        robot.get_frame_tree_snapshot(),
                        "vision",
                        "hand"
                    )
                    
                    pose_data = PoseData(
                        position=[
                            vision_tform_hand.position.x,
                            vision_tform_hand.position.y,
                            vision_tform_hand.position.z,
                        ],
                        quaternion=[
                            vision_tform_hand.rotation.w,
                            vision_tform_hand.rotation.x,
                            vision_tform_hand.rotation.y,
                            vision_tform_hand.rotation.z,
                        ]
                    )

                    # Check for movement in automatic mode
                    if not config.manual_images:
                        if last_pose is not None:
                            pos_diff = np.linalg.norm(
                                np.array(pose_data.position) - np.array(last_pose.position)
                            )
                            if pos_diff > config.movement_threshold:
                                console.print("[yellow]Robot moving, skipping frame[/]")
                                continue
                        last_pose = pose_data

                # Save all data
                save_frame_data(
                    base_dir,
                    counter,
                    rgb_img,
                    depth_img,
                    pose_data,
                    save_visualization=True
                )

                console.print(f"[green]Saved frame {counter}[/]")
                counter += 1

        except KeyboardInterrupt:
            console.print("\n[yellow]Data collection interrupted[/]")
        
        console.print(f"[green]Collected {counter} frames[/]")
        return True


if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
