"""Mock environment creator that uses the Spot robot."""

import os
import time
from pathlib import Path
from typing import List, Set, Dict, Optional, Tuple
import argparse
import networkx as nx
import numpy as np

from bosdyn.client import create_standard_sdk, ResponseError, RpcError
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import image_pb2
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, BODY_FRAME_NAME
from bosdyn.client.util import authenticate

from predicators.envs.mock_spot_env import MockSpotEnv


class SpotMockEnvCreator:
    """Helper class for creating mock Spot environments using the robot."""

    def __init__(self, hostname: str, path_dir: str = "spot_mock_data") -> None:
        self.operators = [
            "MoveToReachObject",
            "MoveToHandViewObject",
            "PickObjectFromTop",
            "PlaceObjectOnTop",
            "DropObjectInside",
            "SweepIntoContainer",
        ]
        self.env = MockSpotEnv(path_dir)
        self.graph = nx.DiGraph()  # For path planning
        
        # Connect to Spot
        sdk = create_standard_sdk('MockEnvCreator')
        robot = sdk.create_robot(hostname)
        
        # Authenticate robot
        authenticate(robot)
        
        # Get clients
        self.image_client = robot.ensure_client(ImageClient.default_service_name)
        self.command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self.robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        
    def capture_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture RGB and depth images from Spot's cameras."""
        # Build image requests
        image_requests = []
        
        # RGB request
        rgb_request = build_image_request(
            image_source_name="hand_color_image",
            pixel_format=None,  # Let Spot choose the format
            quality_percent=100,
            resize_ratio=1
        )
        image_requests.append(rgb_request)
        
        # Depth request
        depth_request = build_image_request(
            image_source_name="hand_depth_in_hand_color_frame",
            pixel_format=None,  # Let Spot choose the format
            quality_percent=100,
            resize_ratio=1
        )
        image_requests.append(depth_request)
        
        try:
            # Get images from robot
            image_responses = self.image_client.get_image(image_requests)
            
            # Process RGB image
            rgb_response = image_responses[0]
            rgb_image = np.frombuffer(rgb_response.shot.image.data, dtype=np.uint8)
            rgb_image = rgb_image.reshape(rgb_response.shot.image.rows,
                                        rgb_response.shot.image.cols, 3)
            
            # Process depth image
            depth_response = image_responses[1]
            depth_image = np.frombuffer(depth_response.shot.image.data, dtype=np.uint16)
            depth_image = depth_image.reshape(depth_response.shot.image.rows,
                                            depth_response.shot.image.cols)
            # Convert to meters
            depth_image = depth_image.astype(float) / 1000.0
            
            return rgb_image, depth_image
            
        except (ResponseError, RpcError) as e:
            print(f"Error capturing images: {e}")
            return None, None
            
    def get_gripper_state(self) -> bool:
        """Get current gripper state (open/closed)."""
        try:
            robot_state = self.robot_state_client.get_robot_state()
            gripper_open = robot_state.manipulator_state.is_gripper_holding_item
            return not gripper_open  # Invert since is_gripper_holding_item means closed
        except (ResponseError, RpcError) as e:
            print(f"Error getting gripper state: {e}")
            return True  # Default to open
        
    def add_state(self, objects_in_view: Optional[Set[str]] = None,
                 objects_in_hand: Optional[Set[str]] = None) -> str:
        """Add current robot state as a new state.
        
        Args:
            objects_in_view: Set of object names visible in the image
            objects_in_hand: Set of object names being held
            
        Returns:
            state_id: ID of the new state
        """
        # Capture current state
        rgb_image, depth_image = self.capture_images()
        gripper_open = self.get_gripper_state()
        
        # Add state to environment
        state_id = self.env.add_state(
            rgb_image=rgb_image,
            depth_image=depth_image,
            objects_in_view=objects_in_view or set(),
            objects_in_hand=objects_in_hand or set(),
            gripper_open=gripper_open
        )
        
        # Add to graph
        self.graph.add_node(state_id)
        return state_id
        
    def add_transition(self, from_state: str, action: str, to_state: str) -> None:
        """Add a transition between states."""
        if action not in self.operators:
            raise ValueError(f"Unknown operator: {action}")
        self.env.add_transition(from_state, action, to_state)
        self.graph.add_edge(from_state, to_state, action=action)
    
    def get_paths_to_goal(self, goal_state: str) -> List[List[Tuple[str, str, str]]]:
        """Get all possible paths to the goal state.
        
        Returns a list of paths, where each path is a list of
        (from_state, action, to_state) tuples.
        """
        paths = []
        for node in self.graph.nodes:
            if node == goal_state:
                continue
            try:
                # Find all simple paths from this node to goal
                for path in nx.all_simple_paths(self.graph, node, goal_state):
                    path_with_actions = []
                    for i in range(len(path)-1):
                        from_state = path[i]
                        to_state = path[i+1]
                        edge = self.graph.edges[from_state, to_state]
                        action = edge["action"]
                        path_with_actions.append((from_state, action, to_state))
                    paths.append(path_with_actions)
            except nx.NetworkXNoPath:
                continue
        return paths
    
    def get_missing_transitions(self) -> List[Tuple[str, str]]:
        """Get list of (state, operator) pairs that have no transition defined."""
        missing = []
        for state in self.graph.nodes:
            for op in self.operators:
                if not any(edge["action"] == op 
                          for edge in self.graph.edges(state, data=True)):
                    missing.append((state, op))
        return missing


def create_mock_env_interactive(hostname: str, path_dir: str) -> None:
    """Interactive function to create a mock environment using Spot."""
    print("Creating mock Spot environment...")
    
    # Initialize creator
    creator = SpotMockEnvCreator(hostname, path_dir)
    
    # Show available operators
    print("\nAvailable operators:")
    for op in creator.operators:
        print(f"- {op}")
    
    # Iteratively build graph
    while True:
        print("\nOptions:")
        print("1. Add current state")
        print("2. Add transition")
        print("3. View paths to goal")
        print("4. View missing transitions")
        print("5. Finish")
        
        choice = input("\nEnter choice (1-5): ")
        
        if choice == "1":
            # Add current state
            objects_in_view = set(input("Enter objects in view (comma-separated): ").split(","))
            if "" in objects_in_view:
                objects_in_view.remove("")
                
            objects_in_hand = set(input("Enter objects in hand (comma-separated): ").split(","))
            if "" in objects_in_hand:
                objects_in_hand.remove("")
            
            state_id = creator.add_state(
                objects_in_view=objects_in_view,
                objects_in_hand=objects_in_hand
            )
            print(f"Added state with ID: {state_id}")
            
        elif choice == "2":
            # Add transition
            from_state = input("Enter from state ID: ")
            print("\nAvailable operators:")
            for op in creator.operators:
                print(f"- {op}")
            action = input("Enter operator name: ")
            to_state = input("Enter to state ID: ")
            
            try:
                creator.add_transition(from_state, action, to_state)
                print("Added transition successfully")
            except ValueError as e:
                print(f"Error: {e}")
            
        elif choice == "3":
            # View paths to goal
            goal = input("Enter goal state ID: ")
            paths = creator.get_paths_to_goal(goal)
            print("\nPaths to goal:")
            for i, path in enumerate(paths):
                print(f"\nPath {i+1}:")
                for from_state, action, to_state in path:
                    print(f"  {from_state} --({action})--> {to_state}")
                    
        elif choice == "4":
            # View missing transitions
            missing = creator.get_missing_transitions()
            print("\nMissing transitions:")
            for state, op in missing:
                print(f"- State '{state}' has no transition for '{op}'")
                
        elif choice == "5":
            break
            
        else:
            print("Invalid choice!")


def main():
    parser = argparse.ArgumentParser(description="Create mock Spot environment using robot")
    parser.add_argument("--hostname", type=str, required=True,
                      help="Hostname or IP of Spot robot")
    parser.add_argument("--path_dir", type=str, default="spot_mock_data",
                      help="Directory to store environment data")
    parser.add_argument("--dir_name", type=str, required=True,
                      help="Name of subdirectory for this environment")
    
    args = parser.parse_args()
    
    # Create full path
    path_dir = os.path.join(args.path_dir, args.dir_name)
    
    # Create environment interactively
    create_mock_env_interactive(args.hostname, path_dir)


if __name__ == "__main__":
    main() 