"""Ground-truth options for the LIS thesis pick-place environment."""

from typing import Dict, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class LISThesisPickPlaceGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the LIS thesis pick-place environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"lis_thesis_pickplace"}

    @classmethod
    def get_options(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            action_space: Box) -> Set[ParameterizedOption]:  # pragma: no cover

        del env_name, predicates  # unused.

        object_type = types["object"]
        ball_type = types["ball"]
        table_type = types["table"]

        # Move to an object location
        Move = utils.SingletonParameterizedOption(
            # variables: [object to move to]
            # params: []
            "move",
            cls._create_dummy_policy(action_space),
            types=[object_type])

        # Pick up a ball
        Pick = utils.SingletonParameterizedOption(
            # variables: [ball to pick]
            # params: []
            "pick",
            cls._create_dummy_policy(action_space),
            types=[ball_type])

        # Place object on table
        PlaceOn = utils.SingletonParameterizedOption(
            # variables: [object to place, table to place on]
            # params: []
            "place_on",
            cls._create_dummy_policy(action_space),
            types=[ball_type, table_type])

        return {Move, Pick, PlaceOn}

    @classmethod
    def _create_dummy_policy(
            cls, action_space: Box) -> ParameterizedPolicy:  # pragma: no cover
        del action_space  # unused

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params
            raise ValueError("Shouldn't be attempting to run this policy!")

        return policy
