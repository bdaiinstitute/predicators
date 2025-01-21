"""Test loading saved mock environment data.

This test demonstrates:
1. Basic environment setup
2. Loading saved environment data
3. Verifying loaded state matches expectations
"""

import os
from predicators.envs.mock_spot_env import MockSpotPickPlaceTwoCupEnv
from predicators import utils
from predicators.settings import CFG

def test_load_mock_env():
    """Test loading a saved mock environment."""
    # Set up configuration
    env_name = "mock_spot_pick_place_two_cup"  # This matches the environment's name
    test_dir = os.path.join("mock_env_data", env_name)
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "mock_env_data_dir": test_dir
    })
    
    # Create environment - it should load from the path specified in the class
    env = MockSpotPickPlaceTwoCupEnv()
    
    # Verify environment loaded correctly
    assert env.name == env_name
    assert env.load_from_path == test_dir
    
    # Verify objects exist
    assert hasattr(env, "cup1")
    assert hasattr(env, "cup2")
    assert hasattr(env, "table")
    assert hasattr(env, "target")
    
    # Get initial observation
    obs = env._build_observation(None)
    
    # Verify observation has expected structure
    assert obs.state_id is not None
    assert obs.gripper_open is not None
    assert obs.objects_in_view is not None
    assert obs.objects_in_hand is not None
    
    # Verify VLM predicates are handled correctly
    if CFG.spot_vlm_eval_predicate:
        assert obs.vlm_predicates is not None
        assert obs.vlm_atom_dict is not None
    else:
        assert obs.vlm_predicates is None
        # When VLM is disabled, vlm_atom_dict is initialized as empty dict
        assert not obs.vlm_atom_dict  # Empty dict or None 