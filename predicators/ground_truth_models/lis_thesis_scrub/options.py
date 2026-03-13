"""Ground-truth options for the LIS thesis scrub environment."""

from typing import Dict, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class LISThesisScrubGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the LIS thesis scrub environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"lis_thesis_scrub"}

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
        container_type = types["container"]
        furniture_type = types["furniture"]

        # Move to an object location
        Move = utils.SingletonParameterizedOption(
            "move",
            cls._create_dummy_policy(action_space),
            types=[object_type])

        # Pick up an object
        Pick = utils.SingletonParameterizedOption(
            "pick",
            cls._create_dummy_policy(action_space),
            types=[tool_type])

        # Unblock access to surface
        Unblock = utils.SingletonParameterizedOption(
            "unblock",
            cls._create_dummy_policy(action_space),
            types=[furniture_type, surface_type])

        # Block access to surface
        Block = utils.SingletonParameterizedOption(
            "block",
            cls._create_dummy_policy(action_space),
            types=[furniture_type, surface_type])

        # Dump contents from container
        Dump = utils.SingletonParameterizedOption(
            "dump",
            cls._create_dummy_policy(action_space),
            types=[container_type, tool_type])

        # Scrub a surface
        Scrub = utils.SingletonParameterizedOption(
            "scrub",
            cls._create_dummy_policy(action_space),
            types=[tool_type, surface_type])

        # Place object in container
        PlaceIn = utils.SingletonParameterizedOption(
            "place_in",
            cls._create_dummy_policy(action_space),
            types=[tool_type, container_type])

        return {Move, Pick, Unblock, Block, Dump, Scrub, PlaceIn}

    @classmethod
    def _create_dummy_policy(
            cls, action_space: Box) -> ParameterizedPolicy:  # pragma: no cover
        del action_space  # unused

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params
            raise ValueError("Shouldn't be attempting to run this policy!")

        return policy
