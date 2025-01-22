"""Test suite for verifying planning in the mock Spot environment.

This test suite verifies that the mock Spot environment correctly handles VLM-based
perception and planning, similar to the real Spot environment but without requiring
physical robot interaction.
"""

import os
import tempfile
import shutil
from typing import Set

import numpy as np

from predicators import utils
from predicators.envs import create_new_env
from predicators.envs.mock_spot_env import MockSpotEnv
from predicators.perception.mock_spot_perceiver import MockSpotPerceiver, _MockSpotObservation
from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext
from predicators.settings import CFG
from predicators.structs import Predicate, Type, VLMPredicate, Object, GroundAtom, EnvironmentTask, State
from bosdyn.api.geometry_pb2 import FrameTreeSnapshot
from bosdyn.client.math_helpers import SE3Pose


def test_mock_spot_perceiver() -> None:
    """Test that the mock perceiver correctly handles VLM predicates."""
    # Create temp directory for test data
    temp_dir = tempfile.mkdtemp()
    try:
        # Set up mock environment with VLM predicates enabled
        utils.reset_config({
            "env": "mock_spot",
            "approach": "oracle",
            "num_test_tasks": 1,
            "mock_env_data_dir": temp_dir,
            "spot_vlm_eval_predicate": True  # Enable VLM predicate evaluation
        })

        # Create environment and perceiver
        env = create_new_env("mock_spot")
        perceiver = MockSpotPerceiver(data_dir=temp_dir)

        # Create mock types for testing
        obj_type = Type("object", ["x", "y", "z"])
        container_type = Type("container", ["x", "y", "z"], parent=obj_type)

        # Create VLM predicates for testing
        inside_pred = VLMPredicate(
            "Inside", [obj_type, container_type],
            prompt="This predicate is true if the first object is inside the second object (container)."
        )
        on_pred = VLMPredicate(
            "On", [obj_type, obj_type],
            prompt="This predicate is true if the first object is on top of the second object."
        )

        # Initialize perceiver with VLM predicates
        perceiver.update_state(
            gripper_open=True,
            objects_in_view=set(),
            objects_in_hand=set(),
            vlm_predicates={inside_pred, on_pred}
        )

        # Verify initial state
        obs = perceiver.get_observation()
        assert obs is not None
        assert obs.vlm_predicates == {inside_pred, on_pred}
        assert not obs.vlm_atom_dict  # Should be empty initially

        # Update state with mock images and verify VLM predicates
        rgb_image = np.zeros((100, 100, 3), dtype=np.uint8)
        depth_image = np.zeros((100, 100), dtype=np.uint16)
        perceiver.save_image(RGBDImageWithContext(
            rgb=rgb_image,
            depth=depth_image,
            camera_name="mock_camera",
            image_rot=0.0,
            world_tform_camera=SE3Pose(x=0.0, y=0.0, z=0.0, rot=np.eye(3)),
            depth_scale=1.0,
            transforms_snapshot=FrameTreeSnapshot(),
            frame_name_image_sensor="mock_camera",
            camera_model=None
        ))
        perceiver.update_state(
            gripper_open=True,
            objects_in_view=set(),
            objects_in_hand=set(),
            vlm_predicates={inside_pred, on_pred}
        )

        # Verify VLM predicates are preserved
        obs = perceiver.get_observation()
        assert obs is not None
        assert obs.vlm_predicates == {inside_pred, on_pred}

        # Reset should clear VLM state
        # Create test task
        robot = Object("robot", next(t for t in env.types if t.name == "robot"))
        cube = Object("cube", next(t for t in env.types if t.name == "movable_object"))
        target = Object("target", next(t for t in env.types if t.name == "immovable_object"))

        init_atoms = {
            GroundAtom(next(p for p in env.predicates if p.name == "HandEmpty"), [robot]),
            GroundAtom(next(p for p in env.predicates if p.name == "InView"), [robot, cube]),
            GroundAtom(next(p for p in env.predicates if p.name == "On"), [cube, target])
        }
        goal_atoms = {
            GroundAtom(next(p for p in env.predicates if p.name == "HandEmpty"), [robot]),
            GroundAtom(next(p for p in env.predicates if p.name == "On"), [cube, target])
        }

        task = EnvironmentTask(State(init_atoms), goal_atoms)
        perceiver.reset(task)
        obs = perceiver.get_observation()
        assert obs is not None
        assert not obs.vlm_predicates
        assert not obs.vlm_atom_dict

    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
