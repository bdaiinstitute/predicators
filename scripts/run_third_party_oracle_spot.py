"""Example script for defining a new spot environment and a new NSRT for that
environment, and running an oracle approach, without modifying predicators."""

from typing import Dict, Sequence, Set

from bosdyn.client import math_helpers
from gym.spaces import Box

from predicators import utils
from predicators.approaches.oracle_approach import OracleApproach
from predicators.approaches.spot_wrapper_approach import SpotWrapperApproach
from predicators.cogman import CogMan
from predicators.envs.spot_env import SpotRearrangementEnv, \
    _base_object_type, _Blocking, _create_operators, _HandEmpty, _Holding, \
    _immovable_object_type, _InView, _movable_object_type, _NotBlocked, _On, \
    _Reachable, _robot_type
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models.spot.nsrts import \
    _move_to_view_object_sampler
from predicators.ground_truth_models.spot.options import \
    _OPERATOR_NAME_TO_PARAM_SPACE, _OPERATOR_NAME_TO_POLICY
from predicators.option_model import _OracleOptionModel
from predicators.perception.spot_perceiver import SpotPerceiver
from predicators.spot_utils.perception.object_detection import \
    KnownStaticObjectDetectionID, LanguageObjectDetectionID, \
    ObjectDetectionID
from predicators.spot_utils.utils import load_spot_metadata
from predicators.structs import Action, Array, EnvironmentTask, \
    GoalDescription, GroundAtom, Object, ParameterizedOption, Predicate, \
    State, STRIPSOperator, Type

###############################################################################
#                         Custom Environment Definition                       #
###############################################################################

# Borrow some of the operators from the base spot environment.
_OP_TO_NAME = {o.name: o for o in _create_operators()}
_MoveToViewObjectOperator = _OP_TO_NAME["MoveToViewObject"]

# Create new options and operators.
_OldPickObjectFromTopOperator = _OP_TO_NAME["PickObjectFromTop"]
_CustomPickObjectFromTopOperator = STRIPSOperator(
    "CustomPickObjectFromTop",
    # These can all be changed, but for now, we'll just use them as is.
    _OldPickObjectFromTopOperator.parameters,
    _OldPickObjectFromTopOperator.preconditions,
    _OldPickObjectFromTopOperator.add_effects,
    _OldPickObjectFromTopOperator.delete_effects,
    _OldPickObjectFromTopOperator.ignore_effects)


class CustomSpotGraspEnv(SpotRearrangementEnv):
    """A custom environment where the only task is to grasp a soda can."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        self._strips_operators = {
            _MoveToViewObjectOperator,
            _CustomPickObjectFromTopOperator,
        }

    @classmethod
    def get_name(cls) -> str:
        return "custom_spot_soda_grasp_env"

    @property
    def types(self) -> Set[Type]:
        return {
            _robot_type,
            _base_object_type,
            _movable_object_type,
            _immovable_object_type,
        }

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            _On,
            _HandEmpty,
            _Holding,
            _Reachable,
            _InView,
            _Blocking,
            _NotBlocked,
        }

    @property
    def percept_predicates(self) -> Set[Predicate]:
        """The predicates that are NOT stored in the simulator state."""
        return {
            _HandEmpty,
            _Holding,
            _On,
            _InView,
            _Reachable,
            _Blocking,
            _NotBlocked,
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return self.predicates

    @property
    def _detection_id_to_obj(self) -> Dict[ObjectDetectionID, Object]:

        detection_id_to_obj: Dict[ObjectDetectionID, Object] = {}

        soda_can = Object("soda_can", _movable_object_type)
        soda_can_detection = LanguageObjectDetectionID("soda can")
        detection_id_to_obj[soda_can_detection] = soda_can

        known_immovables = load_spot_metadata()["known-immovable-objects"]
        for obj_name, obj_pos in known_immovables.items():
            obj = Object(obj_name, _immovable_object_type)
            pose = math_helpers.SE3Pose(obj_pos["x"],
                                        obj_pos["y"],
                                        obj_pos["z"],
                                        rot=math_helpers.Quat())
            detection_id = KnownStaticObjectDetectionID(obj_name, pose)
            detection_id_to_obj[detection_id] = obj

        return detection_id_to_obj

    def _get_initial_nonpercept_atoms(self) -> Set[GroundAtom]:
        return set()

    def _generate_goal_description(self) -> GoalDescription:
        return "pick up the soda can"


###############################################################################
#                       Custom Option / NSRT Definitions                      #
###############################################################################

# MoveToViewObject (borrowed from existing)
types = [p.type for p in _MoveToViewObjectOperator.parameters]
policy = _OPERATOR_NAME_TO_POLICY[_MoveToViewObjectOperator.name]
params_space = _OPERATOR_NAME_TO_PARAM_SPACE[_MoveToViewObjectOperator.name]
_MoveToViewObjectOption = utils.SingletonParameterizedOption(
    _MoveToViewObjectOperator.name, policy, types, params_space)
_MoveToViewObjectNSRT = _MoveToViewObjectOperator.make_nsrt(
    option=_MoveToViewObjectOption,
    option_vars=_MoveToViewObjectOperator.parameters,
    sampler=_move_to_view_object_sampler,
)

# PickObjectFromTop (created anew, with some pieces borrowed from existing)
types = [p.type for p in _CustomPickObjectFromTopOperator.parameters]


def _execute_custom_pick_from_top_on_robot(object_name: str) -> None:
    # TODO: add whatever code here
    print(f"I'm executing a pick on {object_name}!")


def custom_pick_from_top_policy(state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> Action:
    """A custom implementation of pick from top."""
    del memory, params  # not used, probably

    name = _CustomPickObjectFromTopOperator.name
    target_obj_idx = 1
    target_obj = objects[target_obj_idx]

    return utils.create_spot_env_action(
        name, objects, _execute_custom_pick_from_top_on_robot,
        (target_obj.name, ))


custom_pick_from_top_params_space = Box(0, 1, (0, ))  # not parameterized
_CustomPickObjectFromTopOption = utils.SingletonParameterizedOption(
    _CustomPickObjectFromTopOperator.name, custom_pick_from_top_policy, types,
    custom_pick_from_top_params_space)
_CustomPickObjectFromTopNSRT = _CustomPickObjectFromTopOperator.make_nsrt(
    option=_CustomPickObjectFromTopOption,
    option_vars=_CustomPickObjectFromTopOperator.parameters,
    sampler=utils.null_sampler,
)


def _main() -> None:
    args = utils.parse_args()
    utils.update_config(args)

    # Create the environment.
    env = CustomSpotGraspEnv()

    # Create the options and NSRTs.
    options: Set[ParameterizedOption] = {
        _MoveToViewObjectOption, _CustomPickObjectFromTopOption
    }
    nsrts = {_MoveToViewObjectNSRT, _CustomPickObjectFromTopNSRT}

    # Create the option model (shouldn't actually get used for now).
    option_model = _OracleOptionModel(options, env.simulate)

    # Create oracle approach.
    train_tasks = [t.task for t in env.get_train_tasks()]
    base_approach = OracleApproach(env.predicates,
                                   options,
                                   env.types,
                                   env.action_space,
                                   train_tasks,
                                   nsrts=nsrts,
                                   option_model=option_model)
    # Wrap the approach in a spot wrapper.
    approach = SpotWrapperApproach(base_approach, env.predicates, options,
                                   env.types, env.action_space, train_tasks)

    # Create perceiver, execution monitor, cogman.
    perceiver = SpotPerceiver()
    execution_monitor = create_execution_monitor("trivial")
    cogman = CogMan(approach, perceiver, execution_monitor)

    # Reset the environment.
    test_tasks = env.get_test_tasks()
    obs = env.reset("test", 0)
    goal_description = test_tasks[0].goal_description
    env_task = EnvironmentTask(obs, goal_description)

    # Reset cogman.
    cogman.reset(env_task)

    # Run cogman.
    max_steps = 10
    for _ in range(max_steps):
        if env.goal_reached():
            print("Goal reached!")
            break
        act = cogman.step(obs)
        if act is None:
            # I don't expect this to happen.
            print("Cogman terminated without goal reached.")
            break
        obs = env.step(act)
    else:
        print(f"Reached max steps ({max_steps}), goal not reached")


if __name__ == "__main__":
    _main()
