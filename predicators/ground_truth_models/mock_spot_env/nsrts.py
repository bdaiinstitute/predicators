"""Ground-truth NSRTs for the mock Spot environment.

Key implementation notes:
1. Unlike other environments that define their own operators (e.g., Sokoban), 
   we reuse the operators already defined in MockSpotEnv. This is because:
   - The operators are already well-defined in the environment
   - We want to maintain consistency with the environment's transition graph
   - The operators include all necessary parameters, preconditions, and effects

2. We use dummy samplers (null_sampler) for all operators because:
   - The mock environment doesn't need actual sampling
   - State transitions are defined by the graph structure
   - The environment's add_state() and add_transition() methods handle state management

3. The NSRTs are created by directly converting STRIPSOperators to NSRTs using 
   make_nsrt(), which preserves all the operator's properties while adding:
   - The corresponding ParameterizedOption from the options dict
   - The operator's parameters as option variables
   - A dummy sampler that doesn't actually need to generate samples
"""

from typing import Dict, Set, Sequence

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, Object, ParameterizedOption, \
    Predicate, State, Type
from predicators.utils import null_sampler
from predicators.envs import get_or_create_env
from predicators.envs.mock_spot_env import MockSpotEnv


class MockSpotGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the mock Spot environment.
    
    This factory creates NSRTs by reusing operators from the MockSpotEnv.
    Unlike environments that need complex sampling (e.g., Spot), we use
    dummy samplers since state transitions are graph-based.
    """

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"mock_spot", "mock_spot_pick_place_two_cup", "mock_spot_drawer_cleaning",  "mock_spot_sort_weight", "mock_spot_cup_emptiness"}

    @classmethod
    def get_nsrts(cls, env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        """Get all ground truth NSRTs for the mock Spot environment.
        
        This implementation:
        1. Gets the environment instance using get_or_create_env
        2. Converts each STRIPSOperator to an NSRT using make_nsrt()
        3. Uses null_sampler for all operators since we don't need sampling
        4. Pairs each operator with its corresponding option from options dict
        
        Args:
            env_name: Name of the environment (should be "mock_spot")
            types: Dictionary mapping type names to Type objects
            predicates: Dictionary mapping predicate names to Predicate objects
            options: Dictionary mapping option names to ParameterizedOption objects
        
        Returns:
            A set of NSRTs created from the environment's operators
        """
        env = get_or_create_env(env_name)
        assert isinstance(env, MockSpotEnv)
        
        nsrts = set()
        
        # Create NSRTs from operators
        for strips_op in env.strips_operators:
            option = options[strips_op.name]
            nsrt = strips_op.make_nsrt(
                option=option,
                option_vars=strips_op.parameters,
                sampler=null_sampler,  # Use dummy sampler for all operators
            )
            nsrts.add(nsrt)
            
        return nsrts 