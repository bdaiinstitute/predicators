"""Test cases for the mock Spot perceiver.

This module contains tests for the MockSpotPerceiver class, which provides
simulated perception capabilities for testing robot behaviors without
requiring physical Spot robot hardware.

The tests verify:
1. Basic initialization and state management
2. RGBD image handling with camera context
3. Object tracking (in view and in hand)
4. Drawer observation and belief updates
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Set, Dict, Any, List, Optional
from rich import print
import PIL.Image

from predicators import utils
from predicators.perception.mock_spot_perceiver import MockSpotPerceiver
from predicators.spot_utils.perception.perception_structs import UnposedImageWithContext
from predicators.envs.mock_spot_env import (
    MockSpotPickPlaceTwoCupEnv, _robot_type, _container_type, _immovable_object_type, _movable_object_type,
    _Unknown_ContainerEmpty, _Known_ContainerEmpty, _BelieveTrue_ContainerEmpty,
    _BelieveFalse_ContainerEmpty, _DrawerOpen, _DrawerClosed, _Inside,
    _HandEmpty, _NotBlocked, _Reachable, _InHandView, _MockSpotObservation,
    get_vlm_predicates
)
from predicators.structs import Object, GroundAtom, State, VLMPredicate
from predicators.pretrained_model_interface import VisionLanguageModel
from predicators.spot_utils.perception.object_perception import get_vlm


class _DummyVLM(VisionLanguageModel):
    """A dummy VLM for testing."""

    def get_id(self) -> str:
        """Get the ID of this VLM."""
        return "dummy"

    def _sample_completions(self,
                          prompt: str,
                          imgs: Optional[List[PIL.Image.Image]],
                          temperature: float,
                          seed: int,
                          stop_token: Optional[str] = None,
                          num_completions: int = 1) -> List[str]:
        """Sample completions from the VLM."""
        completions = []
        for _ in range(num_completions):
            # Return some dummy predicates that should be true
            completion = "Inside(cup1, target): True.\nOn(cup2, table): True."
            completions.append(completion)
        return completions


def test_mock_spot_perceiver():
    """Tests for the MockSpotPerceiver class."""
    # Set up configuration
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "mock_env_use_belief_operators": True,
        "mock_env_vlm_eval_predicate": False,  # Disable VLM predicates for this test
    })

    # Initialize perceiver
    perceiver = MockSpotPerceiver()
    assert perceiver.get_name() == "mock_spot"

    # Create test objects
    robot = Object("robot", _robot_type)
    drawer = Object("drawer", _container_type)
    table = Object("table", _immovable_object_type)
    apple = Object("apple", _movable_object_type)

    # Create initial observation with drawer unknown
    atoms = {
        GroundAtom(_HandEmpty, [robot]),
        GroundAtom(_NotBlocked, [drawer]),
        GroundAtom(_Reachable, [robot, drawer]),
        GroundAtom(_InHandView, [robot, drawer]),
        GroundAtom(_DrawerOpen, [drawer]),
        GroundAtom(_Unknown_ContainerEmpty, [drawer])
    }
    # Convert atoms to dictionary
    atom_dict = {atom: True for atom in atoms}
    obs = _MockSpotObservation(
        images=None,
        gripper_open=True,
        objects_in_view=set(),
        objects_in_hand=set(),
        state_id="0",
        atom_dict={},
        non_vlm_atom_dict=atom_dict,
        vlm_atom_dict=None,
        vlm_predicates=None
    )

    # Update state through perceiver
    state = perceiver.step(obs)
    assert isinstance(state, State)
    print(state)
    
    # Create observation after finding drawer empty
    atoms = {
        GroundAtom(_HandEmpty, [robot]),
        GroundAtom(_NotBlocked, [drawer]),
        GroundAtom(_Reachable, [robot, drawer]),
        GroundAtom(_InHandView, [robot, drawer]),
        GroundAtom(_DrawerOpen, [drawer]),
        GroundAtom(_Known_ContainerEmpty, [drawer]),
        GroundAtom(_BelieveTrue_ContainerEmpty, [drawer])
    }
    # Convert atoms to dictionary
    atom_dict = {atom: True for atom in atoms}
    obs = _MockSpotObservation(
        images=None,
        gripper_open=True,
        objects_in_view=set(),
        objects_in_hand=set(),
        state_id="1",
        atom_dict={},
        non_vlm_atom_dict=atom_dict,
        vlm_atom_dict=None,
        vlm_predicates=None
    )

    # Update state through perceiver
    state = perceiver.step(obs)
    assert isinstance(state, State)
    print(state)

    # Create observation after finding objects in drawer
    atoms = {
        GroundAtom(_HandEmpty, [robot]),
        GroundAtom(_NotBlocked, [drawer]),
        GroundAtom(_Reachable, [robot, drawer]),
        GroundAtom(_InHandView, [robot, drawer]),
        GroundAtom(_DrawerOpen, [drawer]),
        GroundAtom(_Known_ContainerEmpty, [drawer]),
        GroundAtom(_BelieveFalse_ContainerEmpty, [drawer]),
        GroundAtom(_Inside, [apple, drawer])
    }
    # Convert atoms to dictionary
    atom_dict = {atom: True for atom in atoms}
    obs = _MockSpotObservation(
        images=None,
        gripper_open=True,
        objects_in_view=set(),
        objects_in_hand=set(),
        state_id="2",
        atom_dict={},
        non_vlm_atom_dict=atom_dict,
        vlm_atom_dict=None,
        vlm_predicates=None
    )

    # Update state through perceiver
    state = perceiver.step(obs)
    assert isinstance(state, State)
    print(state)
    
    # Use actual environment to test
    env = MockSpotPickPlaceTwoCupEnv()
    obs = env.reset('train', task_idx=0)
    print(type(obs))
    print(obs)
    state = perceiver.step(obs)
    print(state)
    
    # Test an action and step, then perceiver step
    # env.step()
    # state = perceiver.step(obs)
    # print(state)


def test_mock_spot_perceiver_vlm():
    """Tests for the MockSpotPerceiver class with VLM predicates."""
    # Set up configuration
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "mock_env_use_belief_operators": True,
        "mock_env_vlm_eval_predicate": True,  # Enable VLM predicates
        "spot_vlm_eval_predicate": True,  # Enable VLM evaluation
    })

    # Create environment and perceiver
    env = MockSpotPickPlaceTwoCupEnv()
    perceiver = MockSpotPerceiver()

    # Create mock images
    mock_rgb = np.ones((480, 640, 3), dtype=np.uint8)
    mock_depth = np.ones((480, 640), dtype=np.uint16)
    mock_images = {
        "cam1": UnposedImageWithContext(
            rgb=mock_rgb,
            depth=mock_depth,
            camera_name="cam1",
            image_rot=None
        )
    }

    # Get initial state and verify VLM predicates
    state = env.reset('train', task_idx=0)
    vlm_predicates = get_vlm_predicates()
    assert vlm_predicates is not None, "VLM predicates should not be None"
    assert len(vlm_predicates) > 0, "VLM predicates should be initialized"

    # Create test objects
    robot = Object("robot", _robot_type)
    container = Object("container", _container_type)

    # Create ground atoms
    hand_empty_atom = GroundAtom(_HandEmpty, [robot])
    drawer_closed_atom = GroundAtom(_DrawerClosed, [container])
    non_vlm_atoms = {
        hand_empty_atom: True,
        drawer_closed_atom: True,
    }

    # Create string representation for atom_dict
    atom_dict = {
        "HandEmpty(robot)": True,
        "DrawerClosed(container)": True,
    }

    # Create observation with VLM predicates
    obs = _MockSpotObservation(
        images=mock_images,
        gripper_open=True,
        objects_in_view={robot, container},
        objects_in_hand=set(),
        state_id="test_state",
        atom_dict=atom_dict,
        non_vlm_atom_dict=non_vlm_atoms,
        vlm_predicates=vlm_predicates,
        vlm_atom_dict={},  # Empty initially, will be populated by perceiver
    )

    # Update state with mock images
    state = perceiver.step(obs)
    assert state is not None, "State should not be None"
    assert state.vlm_predicates is not None, "VLM predicates should not be None"
    assert state.vlm_predicates == vlm_predicates, "VLM predicates should be preserved"
    assert state.vlm_atom_dict is not None, "VLM atom dict should not be None"
    assert len(state.vlm_atom_dict) > 0, "VLM atom dict should be populated"
    assert state.non_vlm_atom_dict is not None, "Non-VLM atom dict should not be None"
    assert len(state.non_vlm_atom_dict) > 0, "Non-VLM atom dict should be populated"

    # Print state for debugging
    print(f"VLM predicates: {state.vlm_predicates}")
    print(f"VLM atom dict: {state.vlm_atom_dict}")
    print(f"Non-VLM atom dict: {state.non_vlm_atom_dict}")
    print(f"Visible objects: {state.visible_objects}")
    print(f"Objects in view from obs: {obs.objects_in_view}")
    print(f"Objects in hand from obs: {obs.objects_in_hand}")

    # Verify specific predicates and atoms
    # robot = next((obj for obj in state if obj.type == _robot_type), None)
    # container = next((obj for obj in state if obj.type == _container_type), None)
    # assert robot is not None, "Robot object should exist"
    # assert container is not None, "Container object should exist"
    
    # Check for VLM predicates
    assert any(pred is not None and isinstance(pred, VLMPredicate) 
              for pred in state.vlm_predicates), \
        "Should have VLM predicates"
