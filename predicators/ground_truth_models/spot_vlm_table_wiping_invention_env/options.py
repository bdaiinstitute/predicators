"""Ground-truth options for the spot_vlm_table_wiping_invention_env
environment."""

from typing import Dict, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class SpotTableWipingInventionGroundTruthOptionFactory(GroundTruthOptionFactory
                                                       ):
    """Ground-truth options for the tea making environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"spot_vlm_table_wiping_invention_env", "spot_vlm_table_wiping_human_invention_env"}

    @classmethod
    def get_options(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            action_space: Box) -> Set[ParameterizedOption]:  # pragma: no cover

        del env_name, predicates  # unused.

        robot_type = types["robot"]
        movable_type = types["movable_object"]
        immovable_type = types["immovable_object"]
        table_type = types["table"]
        trash_can_type = types["trash_can"]

        MoveToHandViewObject = utils.SingletonParameterizedOption(
            "MoveToHandViewObject",
            cls._create_dummy_policy(action_space),
            types=[robot_type, movable_type])
        PickFromTop = utils.SingletonParameterizedOption(
            "PickFromTop",
            cls._create_dummy_policy(action_space),
            types=[robot_type, movable_type, immovable_type])
        MoveToReachObject = utils.SingletonParameterizedOption(
            "MoveToReachObject",
            cls._create_dummy_policy(action_space),
            types=[robot_type, immovable_type])
        PlaceInside = utils.SingletonParameterizedOption(
            "PlaceInside",
            cls._create_dummy_policy(action_space),
            types=[robot_type, movable_type, immovable_type])
        PickFromFloor = utils.SingletonParameterizedOption(
            "PickFromFloor",
            cls._create_dummy_policy(action_space),
            types=[robot_type, movable_type])
        WipeAndContinueHoldingEraser = utils.SingletonParameterizedOption(
            "WipeAndContinueHoldingEraser",
            cls._create_dummy_policy(action_space),
            types=[robot_type, movable_type, table_type])
        DumpContentsOntoFloor = utils.SingletonParameterizedOption(
            "DumpContentsOntoFloor",
            cls._create_dummy_policy(action_space),
            types=[robot_type, trash_can_type])
        return {
            MoveToHandViewObject, PickFromTop, MoveToReachObject, PlaceInside,
            PickFromFloor, WipeAndContinueHoldingEraser, DumpContentsOntoFloor
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
