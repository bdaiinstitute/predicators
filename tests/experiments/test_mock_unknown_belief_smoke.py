"""Smoke test: mock env reset with Unknown seeds and dataset present."""

from pathlib import Path

import pytest

from predicators import utils
from predicators.envs.mock_spot_env import MockSpotCupEmptiness


@pytest.mark.smoke
def test_mock_cup_emptiness_unknown_seed_reset():
    """Ensure Unknown seeding doesn't crash mock env reset."""
    # Use the canonical mock dataset directory shipped with the repo.
    data_dir = Path("mock_env_data/MockSpotCupEmptiness")
    if not data_dir.exists():
        pytest.skip("mock_env_data/MockSpotCupEmptiness not available")

    utils.reset_config({
        "seed": 0,
        "approach": "oracle",
        "env": "mock_spot_cup_emptiness",
        "perceiver": "mock_spot_perceiver",
        "execution_monitor": "information_only",
        "spot_perception_refresh_observe_only": True,
        "spot_use_vlm_pointing": True,
        "spot_initial_unknown_predicates": ("ContainerEmpty",),
        "mock_env_data_dir": "mock_env_data",
        "bilevel_plan_without_sim": True,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
    })

    env = MockSpotCupEmptiness(use_gui=False)
    obs = env.reset(train_or_test="test", task_idx=0)
    assert obs is not None
