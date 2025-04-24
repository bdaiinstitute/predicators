"""Ground-truth options for the spot_vlm_juice_making_invention environment."""

from typing import Dict, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class SpotJuiceMakingGroundTruthOptionsFactory(GroundTruthOptionFactory
                                                       ):
    """Ground-truth options for the tea making environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "spot_vlm_juice_making_human_invention_env"
        }

    @classmethod
    def get_options(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            action_space: Box) -> Set[ParameterizedOption]:  # pragma: no cover

        del env_name, predicates  # unused.

        robot_type = types["robot"]
        movable_type = types["movable_object"]
        container_type = types["container"]
        immovable_type = types["immovable_object"]
        juicer_type = types["juicer"]

        PickContainer = utils.SingletonParameterizedOption(
            "PickContainer",
            cls._create_dummy_policy(action_space),
            types=[robot_type, container_type])
        PickMovable = utils.SingletonParameterizedOption(
            "PickMovable",
            cls._create_dummy_policy(action_space),
            types=[robot_type, movable_type])
        PlaceOnLeft = utils.SingletonParameterizedOption(
            "PlaceInsideWasteValveRegion",
            cls._create_dummy_policy(action_space),
            types=[robot_type, container_type, juicer_type])
        PlaceOnRight = utils.SingletonParameterizedOption(
            "PlaceInsideJuiceValveRegion",
            cls._create_dummy_policy(action_space),
            types=[robot_type, container_type, juicer_type])
        PlaceInside = utils.SingletonParameterizedOption(
            "PlaceInside",
            cls._create_dummy_policy(action_space),
            types=[robot_type, movable_type, immovable_type])
        DumpFromOneIntoOther = utils.SingletonParameterizedOption(
            "DumpFromOneIntoOther",
            cls._create_dummy_policy(action_space),
            types=[robot_type, container_type, container_type])
        CloseLid = utils.SingletonParameterizedOption(
            "CloseLid",
            cls._create_dummy_policy(action_space),
            types=[robot_type, juicer_type])
        RunMachine = utils.SingletonParameterizedOption(
            "TurnOnAndRunMachine",
            cls._create_dummy_policy(action_space),
            types=[robot_type, juicer_type, container_type, container_type])
        return {
            PickContainer,
            PickMovable,
            PlaceOnLeft,
            PlaceOnRight,
            DumpFromOneIntoOther,
            PlaceInside,
            CloseLid,
            RunMachine,
        }

    @classmethod
    def _create_dummy_policy(
            cls, action_space: Box) -> ParameterizedPolicy:  # pragma: no cover
        del action_space  # unused

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params
            raise ValueError("Shouldn't be attempting to run this policy!")

        return policy
