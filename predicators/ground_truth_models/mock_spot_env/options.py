"""Ground-truth options for the mock Spot environment.

Key implementation notes:
1. Action representation differs from other environments:
   - Sokoban uses one-hot vectors to encode discrete actions
   - Spot uses continuous parameters for robot control
   - Mock Spot uses operator names stored in action's extra_info field because:
     * We don't need actual continuous control
     * The environment only needs to know which operator to apply
     * The transition graph handles state changes

2. Option parameters are simplified:
   - Uses a zero-dimensional parameter space since we don't need parameters
   - All options are always initiable and terminal
   - The policy ignores state/memory/objects/params and just returns the operator name

3. Options are created to match the environment's operators:
   - One option per STRIPSOperator in MockSpotEnv
   - Option names match operator names exactly
   - Option types match operator parameter types
"""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type
from predicators.envs import get_or_create_env
from predicators.envs.mock_spot_env import MockSpotEnv


class MockSpotGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the mock Spot environment.
    
    This factory creates simplified options that store operator names in
    action's extra_info field instead of using actual continuous parameters
    or one-hot encodings.
    """

    @classmethod
    def get_env_names(cls) -> Set[str]:
        """Get the env names that this factory builds options for."""
        return {"mock_spot", "mock_spot_pick_place_two_cup"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                   predicates: Dict[str, Predicate],
                   action_space: Box) -> Set[ParameterizedOption]:
        """Get all ground truth options for the mock Spot environment.
        
        This implementation:
        1. Gets the environment instance using get_or_create_env
        2. Creates one option per operator with matching name and types
        3. Uses zero-dimensional parameter space since we don't need parameters
        4. Creates policies that store operator names in action's extra_info
        
        Args:
            env_name: Name of the environment (should be "mock_spot")
            types: Dictionary mapping type names to Type objects
            predicates: Dictionary mapping predicate names to Predicate objects
            action_space: The environment's action space (not actually used)
        
        Returns:
            A set of ParameterizedOptions matching the environment's operators
        """
        env = get_or_create_env(env_name)
        assert isinstance(env, MockSpotEnv)
        
        options: Set[ParameterizedOption] = set()
        
        # Create an option for each operator
        for strips_op in env.strips_operators:
            policy = cls._create_policy(strips_op.name)
            option = ParameterizedOption(
                name=strips_op.name,
                types=[p.type for p in strips_op.parameters],  # Changed to list instead of tuple
                params_space=Box(0, 1, (0,)),  # Zero-dimensional parameter space
                policy=policy,
                initiable=lambda s, m, o, p: True,  # Always initiable
                terminal=lambda s, m, o, p: True,  # Always terminal
            )
            options.add(option)
            
        return options

    @classmethod
    def _create_policy(cls, operator_name: str) -> ParameterizedPolicy:
        """Create a policy that stores the operator name in action's extra_info.
        
        Unlike Sokoban's policies that return one-hot vectors or Spot's policies
        that return continuous control parameters, this policy simply stores the
        operator name in the action's extra_info field. The actual action array
        is a dummy array of zeros.
        
        Args:
            operator_name: Name of the operator this policy is for
            
        Returns:
            A policy function that creates actions with operator names in extra_info
        """
        def policy(state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> Action:
            del state, memory, params  # unused
            # Create a dummy action array but store operator name in extra_info
            arr = np.zeros(0, dtype=np.float32)  # Zero-dimensional array to match params_space
            return Action(arr, extra_info={
                "operator_name": operator_name,
                # "objects": {obj.name: obj for obj in objects}
                "objects": list(objects)  # Store objects as a list instead of dict
            })
            
        return policy 