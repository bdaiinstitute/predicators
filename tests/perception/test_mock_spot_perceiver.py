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
from typing import Set, Dict, Any
from rich import print

from predicators import utils
from predicators.perception.mock_spot_perceiver import MockSpotPerceiver
from predicators.spot_utils.perception.perception_structs import UnposedImageWithContext
from predicators.envs.mock_spot_env import (
    MockSpotPickPlaceTwoCupEnv, _robot_type, _container_type, _immovable_object_type, _movable_object_type,
    _Unknown_ContainerEmpty, _Known_ContainerEmpty, _BelieveTrue_ContainerEmpty,
    _BelieveFalse_ContainerEmpty, _DrawerOpen, _DrawerClosed, _Inside,
    _HandEmpty, _NotBlocked, _Reachable, _InHandView, _MockSpotObservation
)
from predicators.structs import Object, GroundAtom, State


def test_mock_spot_perceiver():
    """Tests for the MockSpotPerceiver class."""
    # Set up configuration
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "mock_env_use_belief_operators": True,
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
