"""Spot robot creator for mock environment data.

This module provides a Spot-specific implementation of the mock environment creator
with support for:
1. Interactive RGB-D data collection
2. Robot state tracking
3. Camera transform management
4. Live visualization
"""

import os
from typing import List, Optional, Tuple, Any, Dict, Set, cast
from pathlib import Path
import time
import cv2
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track
from datetime import datetime
import argparse
import sys

import bosdyn.client
from bosdyn.client import Robot, create_standard_sdk
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.client.util import authenticate
from bosdyn.api import robot_state_pb2, image_pb2

from predicators.spot_utils.perception.perception_structs import (
    RGBDImageWithContext, UnposedImageWithContext
)
from predicators.spot_utils.mock_env.mock_env_creator_base import MockEnvCreatorBase
from predicators.structs import Action, GroundAtom
from predicators import utils


class SpotDataCollector:
    """Interactive data collector for Spot robot."""

    def __init__(self, robot: Robot, output_dir: Path) -> None:
        """Initialize collector.
        
        Args:
            robot: Connected Spot robot instance
            output_dir: Directory to save collected data
        """
        self.robot = robot
        self.output_dir = output_dir
        self.image_client = robot.ensure_client(ImageClient.default_service_name)
        self.console = Console()

    def collect_state(self, state_id: str, prompt: str = None) -> bool:
        """Collect data for a state interactively.
        
        Args:
            state_id: ID of state to collect
            prompt: Optional custom prompt
            
        Returns:
            True if data collected, False if skipped/ended
        """
        # Show state info
        self.console.print(f"\n[bold]Collecting data for State {state_id}[/bold]")
        if prompt:
            self.console.print(prompt)
            
        # Get user input
        choice = input("[t]ake photo / [s]kip / [e]nd: ").lower()
        if choice == 'e':
            return False
        elif choice == 's':
            return True
            
        # Collect RGB-D data
        rgb, depth = self.capture_rgbd()
        if rgb is None or depth is None:
            self.console.print("[red]Failed to capture images[/red]")
            return True
            
        # Save data
        state_dir = self.output_dir / "state_info" / f"state_{state_id}"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw data
        np.save(state_dir / "rgb.npy", rgb)
        np.save(state_dir / "depth.npy", depth)
        
        # Save previews
        cv2.imwrite(str(state_dir / "preview_rgb.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(state_dir / "preview_rgbd.jpg"), self.create_rgbd_overlay(rgb, depth))
        
        # Save metadata
        self.save_metadata(state_dir, state_id)
        
        return True

    def capture_rgbd(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture synchronized RGB-D images."""
        try:
            # Get images from robot
            images = self.image_client.get_image_from_sources([
                "hand_depth_in_hand_color_frame",
                "hand_color_image"
            ])
            
            if len(images) < 2:
                return None, None
                
            # Process depth
            depth = self.process_depth_image(
                images[0].shot.image.data,
                (images[0].shot.image.rows, images[0].shot.image.cols)
            )
            
            # Process RGB
            rgb = cv2.imdecode(
                np.frombuffer(images[1].shot.image.data, dtype=np.uint8),
                -1
            )
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
            return rgb, depth
            
        except Exception as e:
            self.console.print(f"[red]Error capturing images:[/red] {e}")
            return None, None

    def process_depth_image(self, depth_data: bytes, shape: Tuple[int, int]) -> np.ndarray:
        """Process raw depth data into meters."""
        depth = np.frombuffer(depth_data, dtype=np.uint16)
        depth = depth.reshape(shape)
        return depth.astype(np.float32) / 1000.0  # Convert to meters

    def create_rgbd_overlay(self, rgb: np.ndarray, depth: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Create RGB-D overlay visualization."""
        # Convert depth to color
        depth_min = np.min(depth)
        depth_max = np.max(depth)
        depth_range = depth_max - depth_min
        
        depth8 = ((depth - depth_min) * 255.0 / depth_range).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth8, cv2.COLORMAP_JET)
        
        # Blend images
        return cv2.addWeighted(rgb, alpha, depth_color, 1 - alpha, 0)

    def save_metadata(self, state_dir: Path, state_id: str) -> None:
        """Save state metadata including camera transforms."""
        # Get robot state
        robot_state = self.robot.ensure_client(RobotStateClient.default_service_name).get_robot_state()
        
        # Get camera transform
        transforms_snapshot = robot_state.transforms_snapshot
        world_tform_camera = transforms_snapshot.get_se3_from_frame_to_frame(
            "vision", "hand_color")
            
        metadata = {
            "state_id": state_id,
            "timestamp": datetime.now().isoformat(),
            "camera": {
                "transform": world_tform_camera.to_matrix().tolist() if world_tform_camera else None,
                "name": "hand_color"
            },
            "robot": {
                "position": robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map["body"].parent_tform_child.position.to_list(),
                "orientation": robot_state.kinematic_state.transforms_snapshot.child_to_parent_edge_map["body"].parent_tform_child.rotation.to_list(),
                "gripper_open": robot_state.manipulator_state.is_gripper_holding_item
            }
        }
        
        # Save metadata
        with open(state_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


class SpotMockEnvCreator(MockEnvCreatorBase):
    """Spot robot creator for mock environment data."""

    def __init__(self, hostname: str, output_dir: str) -> None:
        """Initialize creator.
        
        Args:
            hostname: Hostname or IP of Spot robot
            output_dir: Directory to store environment data
        """
        super().__init__(output_dir)
        
        # Connect to robot
        self.robot = self._connect_robot(hostname)
        self.collector = SpotDataCollector(self.robot, Path(output_dir))

    def _connect_robot(self, hostname: str) -> Robot:
        """Connect to Spot robot."""
        sdk = create_standard_sdk('MockEnvCreator')
        robot = sdk.create_robot(hostname)
        authenticate(robot)
        return robot

    def collect_plan_data(self, plan: List[Action]) -> None:
        """Collect data following a plan."""
        for i, action in enumerate(plan):
            # Show action info
            self.console.print(f"\nStep {i+1}/{len(plan)}")
            self.console.print(f"Action: {action}")
            
            # Collect state data
            if not self.collector.collect_state(
                str(i),
                prompt=f"Executing: {action}"
            ):
                break

    def collect_manual_data(self) -> None:
        """Collect data manually."""
        state_count = 0
        
        while True:
            # Show options
            self.console.print("\nOptions:")
            self.console.print("1. Add new state")
            self.console.print("2. Review collected states")
            self.console.print("3. End collection")
            
            choice = input("Choice: ")
            if choice == "3":
                break
                
            if choice == "1":
                if not self.collector.collect_state(str(state_count)):
                    break
                state_count += 1
            elif choice == "2":
                self.show_progress()

    def show_progress(self) -> None:
        """Show collection progress."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("State")
        table.add_column("Images")
        table.add_column("Status")
        
        # Add rows
        for state_id in sorted(self.states.keys()):
            state_dir = Path(self.output_dir) / "state_info" / f"state_{state_id}"
            has_images = (state_dir / "rgb.npy").exists() and (state_dir / "depth.npy").exists()
            
            table.add_row(
                state_id,
                "✓" if has_images else "✗",
                "[green]Complete[/green]" if has_images else "[yellow]Incomplete[/yellow]"
            )
        
        self.console.print(table)

    def create_rgbd_image(self, rgb: np.ndarray, depth: np.ndarray,
                         camera_name: str = "hand_color") -> UnposedImageWithContext:
        """Create an UnposedImageWithContext from RGB and depth arrays.
        
        Args:
            rgb: RGB image array (H, W, 3)
            depth: Depth image array (H, W)
            camera_name: Name of the camera
            
        Returns:
            UnposedImageWithContext containing the RGB-D data
        """
        return UnposedImageWithContext(
            rgb=rgb,
            depth=depth,
            camera_name=camera_name,
            image_rot=None  # No rotation information needed for unposed images
        ) 

def main(argv):
    """Main function for Spot mock environment data collection."""
    # Parse arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hostname",
        type=str,
        required=True,
        help="Hostname or IP of Spot robot"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mock_env_data",
        help="Directory to store environment data"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="mock_spot",
        help="Environment name"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="test_task",
        help="Name of task for data collection"
    )
    parser.add_argument(
        "--mode",
        choices=["manual", "plan"],
        default="manual",
        help="Data collection mode: manual or plan-based"
    )
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)

    # Set up environment
    utils.reset_config({
        "env": options.env,
        "approach": "oracle",
        "seed": 123,
    })

    # Create environment to get info
    env = utils.get_env(options.env)
    env_info = {
        "types": env.types,
        "predicates": env.predicates,
        "options": env.options,
        "nsrts": env.nsrts
    }

    # Create creator
    creator = SpotMockEnvCreator(
        hostname=options.hostname,
        output_dir=options.output_dir,
        env_info=env_info
    )

    try:
        if options.mode == "manual":
            # Manual data collection
            creator.collect_manual_data()
        else:
            # Plan-based collection
            # Get initial state and goal
            task = env.get_train_tasks()[0]  # Use first task for now
            plan = creator.plan(
                init_atoms=task.init_atoms,
                goal_atoms=task.goal_atoms,
                objects=set(task.init_atoms)
            )
            if plan:
                creator.collect_plan_data(plan[0])  # Use first plan found
            else:
                print("No plan found!")

        # Save transitions
        creator.save_transitions()
        print(f"\nData collection complete. Output saved to {options.output_dir}")
        return True

    except KeyboardInterrupt:
        print("\nData collection interrupted")
        return False
    except Exception as e:
        print(f"\nError during data collection: {e}")
        return False

if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)