"""Manual mock environment creator without Spot dependencies."""

import os
from pathlib import Path
from typing import List, Set, Dict, Optional, Tuple
import argparse
import networkx as nx
import numpy as np
from PIL import Image

from predicators.envs.mock_spot_env import MockSpotEnv


class MockEnvCreator:
    """Helper class for creating mock Spot environments manually."""

    def __init__(self, path_dir: str = "spot_mock_data") -> None:
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
        
    def add_state(self, rgb_path: Optional[str] = None,
                 depth_path: Optional[str] = None,
                 objects_in_view: Optional[Set[str]] = None,
                 objects_in_hand: Optional[Set[str]] = None,
                 gripper_open: bool = True) -> str:
        """Add a state to the environment.
        
        Args:
            rgb_path: Path to RGB image file (jpg/png)
            depth_path: Path to depth image file (npy)
            objects_in_view: Set of object names visible in the image
            objects_in_hand: Set of object names being held
            gripper_open: Whether the gripper is open
            
        Returns:
            state_id: ID of the new state
        """
        # Load images if provided
        rgb_image = None
        depth_image = None
        
        if rgb_path:
            # Load and convert RGB image
            rgb_img = Image.open(rgb_path)
            rgb_image = np.array(rgb_img)
            
        if depth_path:
            # Load depth image
            depth_image = np.load(depth_path)
            
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


def create_mock_env_interactive(path_dir: str) -> None:
    """Interactive function to create a mock environment."""
    print("Creating mock Spot environment...")
    
    # Initialize creator
    creator = MockEnvCreator(path_dir)
    
    # Show available operators
    print("\nAvailable operators:")
    for op in creator.operators:
        print(f"- {op}")
    
    # Iteratively build graph
    while True:
        print("\nOptions:")
        print("1. Add new state")
        print("2. Add transition")
        print("3. View paths to goal")
        print("4. View missing transitions")
        print("5. Finish")
        
        choice = input("\nEnter choice (1-5): ")
        
        if choice == "1":
            # Add new state
            rgb_path = input("Enter path to RGB image (jpg/png) or press enter for none: ")
            rgb_path = rgb_path if rgb_path else None
            
            depth_path = input("Enter path to depth image (npy) or press enter for none: ")
            depth_path = depth_path if depth_path else None
            
            objects_in_view = set(input("Enter objects in view (comma-separated): ").split(","))
            if "" in objects_in_view:
                objects_in_view.remove("")
                
            objects_in_hand = set(input("Enter objects in hand (comma-separated): ").split(","))
            if "" in objects_in_hand:
                objects_in_hand.remove("")
                
            gripper_open = input("Is gripper open? (y/n): ").lower() == "y"
            
            state_id = creator.add_state(
                rgb_path=rgb_path,
                depth_path=depth_path,
                objects_in_view=objects_in_view,
                objects_in_hand=objects_in_hand,
                gripper_open=gripper_open
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
    parser = argparse.ArgumentParser(description="Create mock Spot environment manually")
    parser.add_argument("--path_dir", type=str, default="spot_mock_data",
                      help="Directory to store environment data")
    parser.add_argument("--dir_name", type=str, required=True,
                      help="Name of subdirectory for this environment")
    
    args = parser.parse_args()
    
    # Create full path
    path_dir = os.path.join(args.path_dir, args.dir_name)
    
    # Create environment interactively
    create_mock_env_interactive(path_dir)


if __name__ == "__main__":
    main() 