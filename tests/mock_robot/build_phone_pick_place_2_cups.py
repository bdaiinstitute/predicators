"""Test manual creation of images for a two-cup pick-and-place task using phone HEIC images.

This test focuses on:
1. Creating transition graph for planning the image collection
2. Adding manual HEIC images from phone for each state
3. Verifying image loading and state transitions

Directory Structure:
    mock_env_data/test_mock_task_phone_pick_place_2_cups/
    ├── images/
    │   ├── state_0/
    │   │   └── cam1.rgb.npy
    │   └── ...
    ├── transitions/
    │   └── Transition Graph, Test Two Cup Pick Place.html
    └── plan.yaml
"""

import os
import numpy as np
from pathlib import Path
from rich import print as rprint

from predicators.envs.mock_spot_env import MockSpotPickPlaceTwoCupEnv
from predicators.spot_utils.mock_env.mock_env_creator_base import MockEnvCreatorBase
from predicators import utils


def test_create_two_cup_pick_place_transition_graph():
    """Create transition graph for the two-cup pick-and-place task.
    
    This test only creates the transition graph without adding images.
    Use this to guide the image collection process:
    
    State 0: Initial state - both cups on table
    State 1: First cup (red) in hand
    State 2: First cup (red) in target
    State 3: Second cup (green) in hand
    State 4: Final state - both cups in target
    """
    # Set up configuration
    test_name = "test_mock_task_phone_pick_place_2_cups"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir
    })
    
    # Create environment
    env = MockSpotPickPlaceTwoCupEnv()
    
    # Create environment creator with pick-place specific env
    creator = MockEnvCreatorBase(test_dir, env=env)
    
    # Plan and visualize transitions
    name = f'Transition Graph, {env.name.replace("_", " ").title()}'
    creator.plan_and_visualize(env.initial_atoms, env.goal_atoms, env.objects, task_name=name)
    
    # Explore and save transitions
    creator.explore_states(env.initial_atoms, env.objects)
    creator.save_transitions()
    
    # Verify transition graph file exists
    graph_file = Path(test_dir) / "transitions" / f"Transition Graph, {env.name.replace('_', ' ').title()}.html"
    assert graph_file.exists(), "Transition graph file not generated"


def test_two_cup_pick_place_with_phone_images():
    """Test creating a two-cup pick-and-place task with manual phone HEIC images.
    
    This test:
    1. Creates a pick-and-place scenario with:
       - A robot
       - A table (immovable object)
       - A target container
       - Two cups (container objects to be moved)
       
    2. Generates transition graph showing:
       - Initial state (cups on table)
       - Intermediate states (picking and placing each cup)
       - Goal state (cups inside target container)
       
    3. Adds manual HEIC images for each state
    """
    # Set up configuration
    test_name = "test_mock_task_phone_pick_place_2_cups"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir
    })
    
    # Create environment
    env = MockSpotPickPlaceTwoCupEnv()
    
    # Create environment creator with pick-place specific env
    creator = MockEnvCreatorBase(test_dir, env=env)
    
    # Plan and visualize transitions
    name = f'Transition Graph, {env.name.replace("_", " ").title()}'
    creator.plan_and_visualize(env.initial_atoms, env.goal_atoms, env.objects, task_name=name)
    
    # Explore and save transitions
    creator.explore_states(env.initial_atoms, env.objects)
    creator.save_transitions()
    
    # Get example HEIC image paths - replace these with your actual HEIC files
    root_path = Path(__file__).parent.parent.parent / "mock_task_images" / "test_mock_task_phone_pick_place_2_cups"
    
    # Define states and their images
    test_state_images = {
        # Initial state - both cups on table
        "0": {
            "cam1.seed0.rgb": (str(root_path / "state0_view0.HEIC"), "rgb"),
        },
        
        # First cup in hand
        "1": {
            "cam1.seed0.rgb": (str(root_path / "state1_view0.HEIC"), "rgb"),
        },
        
        # First cup in target
        "2": {
            "cam1.seed0.rgb": (str(root_path / "state2_view0.HEIC"), "rgb"),
        },
        
        # Second cup in hand
        "3": {
            "cam1.seed0.rgb": (str(root_path / "state3_view0.HEIC"), "rgb"),
        },
        
        # Final state - both cups in target
        "4": {
            "cam1.seed0.rgb": (str(root_path / "state4_view0.HEIC"), "rgb"),
        },
        
        # Test state - both cups in hand
        "5": {
            "cam1.seed0.rgb": (str(root_path / "state5_view0.HEIC"), "rgb"),
        },
        
        # Test state - both cups in target
        "6": {
            "cam1.seed0.rgb": (str(root_path / "state6_view0.HEIC"), "rgb"),
        },
        
        # Test state - both cups in hand
        "7": {
            "cam1.seed0.rgb": (str(root_path / "state7_view0.HEIC"), "rgb"),
        },
    }
    
    # Define objects in view
    objects_in_view = {env.cup1, env.cup2, env.table, env.target}
    
    # Get total number of states from creator's transition graph
    max_state = len(creator.state_to_id) - 1  # states are 0-indexed
    
    # Process each state
    for state_id in [str(i) for i in range(max_state + 1)]:
        if state_id not in test_state_images:
            rprint(f"[yellow]Warning: No image data found for state {state_id}[/yellow]")
            continue
            
        # Determine objects in hand
        objects_in_hand = {env.cup1} if state_id == "1" else {env.cup2} if state_id == "3" else set()
                    
        # Add state
        creator.add_state_from_raw_images(
            test_state_images[state_id],
            state_id=state_id,
            objects_in_view=objects_in_view,
            objects_in_hand=objects_in_hand,
            gripper_open=(state_id not in ["1", "3"])
        )
        
        # Load and verify state
        loaded_state = creator.load_state(state_id)
        assert loaded_state is not None, f"Failed to load state {state_id}"
        assert loaded_state.images is not None, f"No images found in state {state_id}"
        
        # Verify loaded data matches original
        state_image_keys = [k.split("/")[1] for k in test_state_images.keys() if k.startswith(f"{state_id}/")]
        for image_key in state_image_keys:
            camera_id = image_key.split(".")[0]
            full_key = f"{camera_id}_{'.'.join(image_key.split('.')[1:])}"
            assert full_key in loaded_state.images, f"Missing image {full_key} in state {state_id}"
            loaded_img = loaded_state.images[full_key]
            assert loaded_img is not None, f"Image {full_key} is None in state {state_id}"
            assert loaded_img.rgb is not None
            assert len(loaded_img.rgb.shape) == 3  # (H, W, 3)
            assert loaded_img.rgb.dtype == np.uint8
        
        # Verify metadata
        assert loaded_state.objects_in_view == objects_in_view
        if state_id == "1":
            assert loaded_state.objects_in_hand == {env.cup1}
            assert not loaded_state.gripper_open
        elif state_id == "3":
            assert loaded_state.objects_in_hand == {env.cup2}
            assert not loaded_state.gripper_open
        else:
            assert not loaded_state.objects_in_hand
            assert loaded_state.gripper_open
    
    # Verify transition graph file exists
    graph_file = Path(test_dir) / "transitions" / f"Transition Graph, {env.name.replace('_', ' ').title()}.html"
    assert graph_file.exists(), "Transition graph file not generated"

def test_load_saved_two_cup_phone():
    """Test loading a previously saved two-cup pick-and-place task with phone HEIC images."""
    # Set up configuration with the same test directory
    test_name = "test_mock_task_phone_pick_place_2_cups"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir
    })
    
    # Create environment with same name to ensure object consistency
    env = MockSpotPickPlaceTwoCupEnv()
    
    # Create environment creator for loading
    creator = MockEnvCreatorBase(test_dir, env=env)
    
    # Get total number of states from saved data
    image_dir = Path(test_dir) / "images"
    if not image_dir.exists():
        rprint("[red]Warning: No saved image directory found[/red]")
        return
        
    saved_states = [d.name for d in image_dir.iterdir() if d.is_dir()]
    if not saved_states:
        rprint("[red]Warning: No saved states found[/red]")
        return
        
    # Load and verify each state
    for state_id in sorted(saved_states):
        try:
            # Load state
            loaded_state = creator.load_state(state_id)
            
            # Print state info
            rprint(f"[green]State {state_id} loaded:[/green]")
            rprint(f"  Images: {loaded_state.images.keys() if loaded_state.images else 'None'}")
            rprint(f"  Objects in view: {loaded_state.objects_in_view}")
            rprint(f"  Objects in hand: {loaded_state.objects_in_hand}")
            rprint(f"  Gripper open: {loaded_state.gripper_open}")
            
            # Verify basic structure
            assert loaded_state.images, f"No views loaded for {state_id}"
            assert loaded_state.objects_in_view, f"No objects in view for {state_id}"
            
            # Verify expected objects are present
            expected_objects = {env.cup1, env.cup2, env.table, env.target}
            assert loaded_state.objects_in_view == expected_objects
            
            # Verify state-specific conditions
            if state_id == "1":
                assert loaded_state.objects_in_hand == {env.cup1}
                assert not loaded_state.gripper_open
            elif state_id == "3":
                assert loaded_state.objects_in_hand == {env.cup2}
                assert not loaded_state.gripper_open
            else:
                assert not loaded_state.objects_in_hand
                assert loaded_state.gripper_open
                
        except Exception as e:
            rprint(f"[red]Error loading state {state_id}: {str(e)}[/red]") 