"""Test cases for mock Spot NSRTs and options."""

import pytest
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models.mock_spot_env.nsrts import MockSpotGroundTruthNSRTFactory
from predicators.ground_truth_models.mock_spot_env.options import MockSpotGroundTruthOptionFactory
from predicators.envs.mock_spot_env import TYPES, PREDICATES
from predicators.structs import State


def test_mock_spot_nsrts():
    """Tests for mock Spot NSRTs."""
    # Set up configuration
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
    })

    # Get NSRTs
    factory = MockSpotGroundTruthNSRTFactory()
    env_name = "mock_spot"
    
    # First get options since NSRTs need them
    option_factory = MockSpotGroundTruthOptionFactory()
    types_dict = {t.name: t for t in TYPES}
    predicates_dict = {p.name: p for p in PREDICATES}
    options = {
        o.name: o for o in option_factory.get_options(env_name, types_dict, predicates_dict,
                                                    Box(-1, 1, (3,)))
    }
    
    # Now get NSRTs
    nsrts = factory.get_nsrts(env_name, types_dict, predicates_dict, options)
    
    # Test basic properties
    assert len(nsrts) == 5  # number of operators in mock env
    
    # Test that each NSRT has the right components
    for nsrt in nsrts:
        # Each NSRT should have a corresponding option
        assert nsrt.option.name == nsrt.name
        assert nsrt.option in options.values()
        
        # Parameters should match between NSRT and option
        assert len(nsrt.parameters) == len(nsrt.option.types)
        for param, type_ in zip(nsrt.parameters, nsrt.option.types):
            assert param.type == type_
            
        # Should have some preconditions and effects
        assert nsrt.preconditions
        assert nsrt.add_effects or nsrt.delete_effects


def test_mock_spot_options():
    """Tests for mock Spot options."""
    # Set up configuration
    utils.reset_config({
        "env": "mock_spot",
        "approach": "oracle",
        "seed": 123,
    })

    # Get options
    factory = MockSpotGroundTruthOptionFactory()
    env_name = "mock_spot"
    types_dict = {t.name: t for t in TYPES}
    predicates_dict = {p.name: p for p in PREDICATES}
    options = factory.get_options(env_name, types_dict, predicates_dict, Box(-1, 1, (3,)))
    
    # Test basic properties
    assert len(options) == 5  # number of operators in mock env
    
    # Test each option
    for option in options:
        # Test policy
        # Create a dummy state since policy doesn't use it
        dummy_state = State({}, set(), {}, set())
        memory = {}  # not used by policy
        objects = []  # not used by policy
        params = np.zeros(1, dtype=np.float32)  # dummy params
        
        action = option.policy(dummy_state, memory, objects, params)
        
        # Action should be a dummy array with operator name in extra_info
        assert isinstance(action.arr, np.ndarray)
        assert action.arr.shape == (3,)  # matches env's action space
        assert np.all(action.arr == 0)  # dummy array of zeros
        assert action.extra_info["operator_name"] == option.name
        
        # Test initiable and terminal
        assert option.initiable(dummy_state, memory, objects, params)
        assert option.terminal(dummy_state, memory, objects, params) 