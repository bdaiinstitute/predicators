"""Ground-truth options for the LIS thesis sweep environment."""

from typing import Dict, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class LISThesisSweepGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the LIS thesis sweep environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"lis_thesis_sweep"}

    @classmethod
    def get_options(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            action_space: Box) -> Set[ParameterizedOption]:  # pragma: no cover

        del env_name, predicates  # unused.

        object_type = types["object"]
        tool_type = types["tool"]
        surface_type = types["surface"]
        toy_type = types["toy"]
        container_type = types["container"]

        # Move to an object location
        Move = utils.SingletonParameterizedOption(
            "move",
            cls._create_dummy_policy(action_space),
            types=[object_type])

        # Pick up an object
        Pick = utils.SingletonParameterizedOption(
            "pick",
            cls._create_dummy_policy(action_space),
            types=[object_type])

        # Sweep a surface with held tool
        Sweep = utils.SingletonParameterizedOption(
            "sweep",
            cls._create_dummy_policy(action_space),
            types=[surface_type])

        # Place object on surface
        PlaceOn = utils.SingletonParameterizedOption(
            "place_on",
            cls._create_dummy_policy(action_space),
            types=[object_type, surface_type])

        # Place object in container
        PlaceIn = utils.SingletonParameterizedOption(
            "place_in",
            cls._create_dummy_policy(action_space),
            types=[toy_type, container_type])

        return {Move, Pick, Sweep, PlaceOn, PlaceIn}

    @classmethod
    def _create_dummy_policy(
            cls, action_space: Box) -> ParameterizedPolicy:  # pragma: no cover
        del action_space  # unused

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params
            raise ValueError("Shouldn't be attempting to run this policy!")

        return policy
