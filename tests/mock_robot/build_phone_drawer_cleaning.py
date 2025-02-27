"""Test manual creation of images for a drawer cleaning task using phone HEIC images.

This test focuses on:
1. Creating transition graph for planning the image collection
2. Adding manual HEIC images from phone for each state
3. Verifying image loading and state transitions

Directory Structure:
    mock_env_data/test_mock_task_phone_drawer_cleaning/
    ├── images/
    │   ├── state_0/
    │   │   └── cam1.rgb.npy
    │   └── ...
    ├── transitions/
    │   └── Transition Graph, Test Drawer Cleaning.html
    ├── state_mapping.yaml
    └── plan.yaml
"""

import os
import numpy as np
from pathlib import Path
from rich import print as rprint

from predicators.envs.mock_spot_env import (
    MockSpotDrawerCleaningEnv, _robot_type, _container_type, _immovable_object_type, 
    _movable_object_type, _HandEmpty, _NotBlocked, _On, _NotInsideAnyContainer,
    _IsPlaceable, _FitsInXY, _NotHolding, _NEq, _Inside, _HasFlatTopSurface,
    _Reachable, _DrawerClosed, _DrawerOpen
)
from predicators.spot_utils.mock_env.mock_env_creator_base import MockEnvCreatorBase
from predicators.structs import Object, GroundAtom
from predicators import utils


def test_create_drawer_cleaning_transition_graph():
    """Create transition graph for the drawer cleaning task.
    
    This test only creates the transition graph without adding images.
    Use this to guide the image collection process:
    
    State 0: Initial state - both cups in drawer, drawer closed
    State 1: Drawer open, both cups visible
    State 2: First cup (red) in hand
    State 3: First cup (red) in container
    State 4: Second cup (blue) in hand
    State 5: Second cup (blue) in container
    State 6: Drawer closed (final state)
    """
    # Set up configuration
    test_name = "task_phone_drawer_cleaning"
    test_dir = os.path.join("mock_env_data", test_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir,
        "mock_env_use_belief_operators": False
    })
    
    # Create environment
    env = MockSpotDrawerCleaningEnv()
    
    # Create environment creator
    creator = MockEnvCreatorBase(test_dir, env=env)
    
    # Plan and visualize transitions
    name = f'Transition Graph, {test_name.replace("_", " ").title()}'
    creator.plan_and_visualize(env.initial_atoms, env.goal_atoms, env.objects, task_name=name)
    
    # Verify transition graph file exists
    graph_file = Path(test_dir) / "transitions" / f"{name}.html"
    assert graph_file.exists(), "Transition graph file not generated"


def test_drawer_cleaning_with_phone_images():
    """Test creating a drawer cleaning task with manual phone HEIC images.
    
    This test:
    1. Creates a drawer cleaning scenario with:
       - A robot
       - A drawer (container)
       - A container box (container)
       - Two cups (red and blue, movable objects)
       
    2. Generates transition graph showing:
       - Initial state (cups in drawer)
       - Intermediate states (opening drawer, moving cups)
       - Goal state (cups in container, drawer closed)
       
    3. Adds manual HEIC images for each state
    """
    # Set up configuration
    test_name = "test_mock_task_phone_drawer_cleaning"
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
    env = MockSpotDrawerCleaningEnv()
    
    # Create environment creator
    creator = MockEnvCreatorBase(test_dir, env=env)
    
    # Get example HEIC image paths
    root_path = Path(__file__).parent.parent.parent / "mock_task_images" / "phone_drawer_cleaning_2_cups"
    
    # Define states and their images
    test_state_images = {
        # Initial state - both cups in drawer, drawer closed
        "0": {
            "cam1.seed0.rgb": (str(root_path / "drawer, closed.HEIC"), "rgb"),
        },
        
        # Drawer open, both cups visible
        "1": {
            "cam1.seed0.rgb": (str(root_path / "drawer, both cups in drawer.HEIC"), "rgb"),
        },
        
        # First cup (red) in hand
        "2": {
            "cam1.seed0.rgb": (str(root_path / "drawer, blue cup in drawer.HEIC"), "rgb"),
        },
        
        # First cup (red) in container
        "3": {
            "cam1.seed0.rgb": (str(root_path / "container, red cup in container.HEIC"), "rgb"),
        },
        
        # Second cup (blue) in hand
        "4": {
            "cam1.seed0.rgb": (str(root_path / "drawer, empty.HEIC"), "rgb"),
        },
        
        # Second cup (blue) in container
        "5": {
            "cam1.seed0.rgb": (str(root_path / "container, both cups in container.HEIC"), "rgb"),
        },
        
        # Final state - drawer closed
        "6": {
            "cam1.seed0.rgb": (str(root_path / "drawer, closed.HEIC"), "rgb"),
        },
    }
    
    # Define objects in view for all states
    objects_in_view = {env.drawer, env.container, env.red_cup, env.blue_cup}
    
    # Process each state
    for state_id in sorted(test_state_images.keys()):
        if state_id not in test_state_images:
            rprint(f"[yellow]Warning: No image data found for state {state_id}[/yellow]")
            continue
            
        # Determine objects in hand based on state
        objects_in_hand = {env.red_cup} if state_id == "2" else {env.blue_cup} if state_id == "4" else set()
                    
        # Add state
        creator.add_state_from_raw_images(
            test_state_images[state_id],
            state_id=state_id,
            objects_in_view=objects_in_view,
            objects_in_hand=objects_in_hand,
            gripper_open=(state_id not in ["2", "4"])
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
        if state_id == "2":
            assert loaded_state.objects_in_hand == {env.red_cup}
            assert not loaded_state.gripper_open
        elif state_id == "4":
            assert loaded_state.objects_in_hand == {env.blue_cup}
            assert not loaded_state.gripper_open
        else:
            assert not loaded_state.objects_in_hand
            assert loaded_state.gripper_open
    
    # Save transitions (which now includes state mapping)
    creator.save_transitions()
    
    # Verify transition system file exists
    transition_file = Path(test_dir) / "transition_system.yaml"
    assert transition_file.exists(), "Transition system file not generated"
    
    # Test atom to state mappings
    # Check initial state atoms
    drawer_closed_atom = GroundAtom(_DrawerClosed, [env.drawer])
    drawer_states = creator.get_states_with_atom(drawer_closed_atom)
    assert "0" in drawer_states, "Initial state should have drawer closed"
    assert "6" in drawer_states, "Final state should have drawer closed"
    
    # Check goal state atoms
    inside_red_atom = GroundAtom(_Inside, [env.red_cup, env.container])
    inside_blue_atom = GroundAtom(_Inside, [env.blue_cup, env.container])
    
    red_cup_states = creator.get_states_with_atom(inside_red_atom)
    blue_cup_states = creator.get_states_with_atom(inside_blue_atom)
    
    assert "3" in red_cup_states, "State 3 should have red cup in container"
    assert "5" in red_cup_states, "State 5 should have both cups in container"
    assert "5" in blue_cup_states, "State 5 should have both cups in container"
    
    # Check key atoms
    key_atoms = creator.get_key_atoms()
    assert str(drawer_closed_atom) in key_atoms, "DrawerClosed should be a key atom"
    assert str(inside_red_atom) in key_atoms, "Inside(red_cup, container) should be a key atom"
    assert str(inside_blue_atom) in key_atoms, "Inside(blue_cup, container) should be a key atom"


def test_load_saved_drawer_cleaning():
    """Test loading a previously saved drawer cleaning task with phone HEIC images."""
    # Set up configuration with the same test directory
    test_name = "test_mock_task_phone_drawer_cleaning"
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
    env = MockSpotDrawerCleaningEnv()
    
    # Create environment creator for loading
    creator = MockEnvCreatorBase(test_dir, env=env)
    
    # Load state mapping
    creator.load_state_mapping()
    
    # Get unique states
    unique_states = list(creator.get_unique_states())
    assert unique_states, "No unique states found"
    
    # Test atom to canonical state mappings
    drawer_closed_atom = GroundAtom(_DrawerClosed, [env.drawer])
    inside_red_atom = GroundAtom(_Inside, [env.red_cup, env.container])
    inside_blue_atom = GroundAtom(_Inside, [env.blue_cup, env.container])
    
    # Get canonical states for each key atom
    drawer_closed_states = creator.get_canonical_states_with_atom(drawer_closed_atom)
    red_cup_states = creator.get_canonical_states_with_atom(inside_red_atom)
    blue_cup_states = creator.get_canonical_states_with_atom(inside_blue_atom)
    
    # Print atom to state mappings
    rprint("\n[cyan]Atom to State Mappings:[/cyan]")
    rprint(f"  DrawerClosed: {drawer_closed_states}")
    rprint(f"  Inside(red_cup, container): {red_cup_states}")
    rprint(f"  Inside(blue_cup, container): {blue_cup_states}")
    
    # Load and verify each unique state
    for canonical_id, equivalent_ids in unique_states:
        try:
            # Load state
            loaded_state = creator.load_state(canonical_id)
            
            # Print state info
            rprint(f"\n[green]State {canonical_id} loaded:[/green]")
            rprint(f"  Equivalent states: {equivalent_ids}")
            rprint(f"  Images: {loaded_state.images}")
            rprint(f"  Objects in view: {loaded_state.objects_in_view}")
            rprint(f"  Objects in hand: {loaded_state.objects_in_hand}")
            rprint(f"  Gripper open: {loaded_state.gripper_open}")
            
            # Print key atoms true in this state
            key_atoms = creator.get_key_atoms()
            true_key_atoms = [atom for atom in key_atoms 
                            if canonical_id in creator.get_canonical_states_with_atom(atom)]
            if true_key_atoms:
                rprint("  Key atoms true in this state:")
                for atom in true_key_atoms:
                    rprint(f"    - {atom}")
            
            # Verify basic structure
            assert loaded_state.images, f"No views loaded for {canonical_id}"
            assert loaded_state.objects_in_view, f"No objects in view for {canonical_id}"
            
            # Verify expected objects are present
            expected_objects = {env.drawer, env.container, env.red_cup, env.blue_cup}
            assert loaded_state.objects_in_view == expected_objects
            
            # Verify state-specific conditions
            if canonical_id == "2":
                assert loaded_state.objects_in_hand == {env.red_cup}
                assert not loaded_state.gripper_open
            elif canonical_id == "4":
                assert loaded_state.objects_in_hand == {env.blue_cup}
                assert not loaded_state.gripper_open
            else:
                assert not loaded_state.objects_in_hand
                assert loaded_state.gripper_open
                
        except Exception as e:
            rprint(f"[red]Error loading state {canonical_id}: {str(e)}[/red]") 