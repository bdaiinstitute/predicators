"""A bunch of environments useful for testing VLM-based Predicate Invention.

Will likely be updated and potentially split into separate files in the
future.
"""

from typing import List, Optional, Sequence, Set

import matplotlib
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.envs.spot_env import _get_vlm_query_str
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Task, Type

DUMMY_GOAL_OBJ_NAME = "dummy_goal_obj"  # used in VLM parsing as well.


class VLMPredicateEnv(BaseEnv):
    """Environments that use VLM Predicates.

    Note that no simulate function or ground truth model is implemented
    for these yet. These are forthcoming.
    """

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._object_type = Type("object", [])
        self._goal_object_type = Type("goal_object", ["goal_true"],
                                      self._object_type)

        # Predicates
        self._DummyGoal = Predicate("DummyGoal", [self._goal_object_type],
                                    self._Dummy_Goal_holds)

    def simulate(self, state: State, action: Action) -> State:
        raise ValueError("Simulate shouldn't be getting called!")

    @property
    def types(self) -> Set[Type]:
        return {self._object_type, self._goal_object_type}

    def _Dummy_Goal_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "goal_true") > 0.5

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._DummyGoal}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._DummyGoal}

    @property
    def action_space(self) -> Box:
        return Box(low=0.0, high=1.0, shape=(0, ), dtype=np.float32)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise ValueError("shouldn't be trying to render env at any point!")

    def _get_tasks(
        self, num: int, rng: np.random.Generator
    ) -> List[EnvironmentTask]:  # pragma: no cover.
        del num, rng
        raise NotImplementedError("Override!")


class IceTeaMakingEnv(VLMPredicateEnv):
    """A (simplified) version of a tea-making task that's closer to pick-and-
    place than real tea-making."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Env-specific types.
        self._teabag_type = Type("teabag", [], self._object_type)
        self._spoon_type = Type("spoon", [], self._object_type)
        self._cup_type = Type("cup", [], self._object_type)
        self._plate_type = Type("plate", [], self._object_type)
        self._hand_type = Type("hand", [], self._object_type)

    @classmethod
    def get_name(cls) -> str:
        return "ice_tea_making"

    @property
    def types(self) -> Set[Type]:
        return super().types | {
            self._teabag_type, self._spoon_type, self._cup_type,
            self._plate_type, self._hand_type
        }

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        del rng  # unused.
        dummy_goal_obj = Object(DUMMY_GOAL_OBJ_NAME, self._goal_object_type)
        teabag_obj = Object("teabag", self._teabag_type)
        spoon_obj = Object("spoon", self._spoon_type)
        cup_obj = Object("cup", self._cup_type)
        plate_obj = Object("plate", self._plate_type)
        hand_obj = Object("hand", self._hand_type)
        init_state = State({
            dummy_goal_obj: np.array([0.0]),
            teabag_obj: np.array([]),
            plate_obj: np.array([]),
            spoon_obj: np.array([]),
            cup_obj: np.array([]),
            hand_obj: np.array([])
        })
        return [
            EnvironmentTask(
                init_state,
                set([GroundAtom(self._DummyGoal, [dummy_goal_obj])]))
            for _ in range(num)
        ]

    def get_vlm_debug_atom_strs(self,
                                train_tasks: List[Task]) -> List[List[str]]:
        del train_tasks
        atom_strs = set([
            "hand_grasping_spoon(hand, spoon)",
            "hand_grasping_teabag(hand, teabag)", "spoon_in_cup(spoon, cup)",
            "spoon_on_plate(spoon, plate)", "teabag_in_cup(teabag, cup)",
            "teabag_on_plate(teabag, plate)"
        ])
        return [[a] for a in atom_strs]


class SpotVLMTableWipingInventionEnv(VLMPredicateEnv):
    """An env that is intended to be the same as the
    'spot_vlm_table_wiping_env' defined in spot_env.py, but useful for actual
    predicate invention (the env in spot_env.py requires using a spot_wrapper
    approach...)."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # Env-specific types.
        self._robot_type = Type("robot", [], self._object_type)
        self._movable_object_type = Type("movable_object", [],
                                         self._object_type)
        self._immovable_object_type = Type("immovable_object", [],
                                           self._object_type)
        self._table_type = Type("table", [], self._immovable_object_type)
        self._trash_can_type = Type("trash_can", [],
                                    self._immovable_object_type)
        self._VLMIn = utils.create_vlm_predicate(
            "InsideContainer",
            [self._movable_object_type, self._trash_can_type],
            lambda o: _get_vlm_query_str("InsideContainer", o))
        self._TableWiped = utils.create_vlm_predicate(
            "WipedOfMarkerScribbles", [self._table_type],
            lambda o: _get_vlm_query_str("WipedOfMarkerScribbles", o))

    @classmethod
    def get_name(cls) -> str:
        return "spot_vlm_table_wiping_invention_env"

    @property
    def types(self) -> Set[Type]:
        return super().types | {
            self._robot_type, self._table_type, self._object_type,
            self._movable_object_type, self._immovable_object_type,
            self._trash_can_type
        }

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        del rng  # unused.
        spot_obj = Object("spot", self._robot_type)
        table_obj = Object("child_play_table", self._table_type)
        apple_obj = Object("apple", self._movable_object_type)
        green_block_obj = Object("green_block", self._movable_object_type)
        orange_block_obj = Object("orange_block", self._movable_object_type)
        spam_tin_obj = Object("spam_tin", self._movable_object_type)
        trash_can_obj = Object("seethru_plastic_dustbin", self._trash_can_type)
        duster_obj = Object("furry_green_eraser", self._movable_object_type)
        cup_obj = Object("red_drink_cup", self._movable_object_type)
        carboard_recycling_bin = Object("cardboard_recycling_bin",
                                        self._trash_can_type)

        ret_tasks = []
        for i in range(num):
            init_state_dict = {
                spot_obj: np.array([]),
                trash_can_obj: np.array([]),
            }
            if i in [0, 3]:
                init_state_dict.update({
                    table_obj: np.array([]),
                    duster_obj: np.array([]),
                    apple_obj: np.array([]),
                    cup_obj: np.array([]),
                })
                goal = {
                    GroundAtom(self._TableWiped, [table_obj]),
                }
            elif i == 1:
                init_state_dict.update({
                    table_obj: np.array([]),
                    duster_obj: np.array([]),
                    apple_obj: np.array([]),
                    cup_obj: np.array([]),
                })
                goal = {
                    GroundAtom(self._VLMIn, [apple_obj, trash_can_obj]),
                    GroundAtom(self._TableWiped, [table_obj]),
                }
            elif i == 2:
                init_state_dict.update({
                    table_obj: np.array([]),
                    duster_obj: np.array([]),
                    apple_obj: np.array([]),
                })
                init_state_dict[green_block_obj] = np.array([])
                goal = {
                    GroundAtom(self._TableWiped, [table_obj]),
                }
            elif i == 4:
                init_state_dict.update({
                    table_obj: np.array([]),
                    duster_obj: np.array([]),
                    apple_obj: np.array([]),
                })
                init_state_dict[green_block_obj] = np.array([])
                goal = {
                    GroundAtom(self._VLMIn, [apple_obj, trash_can_obj]),
                    GroundAtom(self._TableWiped, [table_obj]),
                }
            elif i == 5:
                init_state_dict.update({
                    green_block_obj: np.array([]),
                    carboard_recycling_bin: np.array([]),
                })
                goal = {
                    GroundAtom(self._VLMIn,
                               [green_block_obj, carboard_recycling_bin])
                }
            elif i == 6:
                init_state_dict.update({
                    orange_block_obj: np.array([]),
                    carboard_recycling_bin: np.array([]),
                })
                goal = {
                    GroundAtom(self._VLMIn,
                               [orange_block_obj, carboard_recycling_bin])
                }
            elif i == 7:
                init_state_dict.update({
                    spam_tin_obj: np.array([]),
                    carboard_recycling_bin: np.array([]),
                })
                goal = {
                    GroundAtom(self._VLMIn,
                               [spam_tin_obj, carboard_recycling_bin])
                }
            else:
                raise NotImplementedError(
                    "Shouldn't be getting here! i = {}".format(i))

            ret_tasks.append(EnvironmentTask(State(init_state_dict), goal))
        return ret_tasks

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._VLMIn, self._TableWiped}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._VLMIn, self._TableWiped}

    def get_vlm_debug_atom_strs(self,
                                train_tasks: List[Task]) -> List[List[str]]:
        del train_tasks
        atom_strs = set([
            "inAir(apple)", "onTable(apple)", "onFloor(furry_green_eraser)",
            "canBeUsedForErasing(furry_green_eraser)",
            "noObjectsOntopTable(child_play_table)"
        ])
        return [[a] for a in atom_strs]


class SpotVLMTableWipingHumanInventionEnv(VLMPredicateEnv):
    """An env that is the same as the above, except intended for invention from
    human demos!"""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # Env-specific types.
        self._robot_type = Type("robot", [], self._object_type)
        self._movable_object_type = Type("movable_object", [],
                                         self._object_type)
        self._immovable_object_type = Type("immovable_object", [],
                                           self._object_type)
        self._table_type = Type("table", [], self._immovable_object_type)
        self._trash_can_type = Type("trash_can", [],
                                    self._immovable_object_type)
        self._VLMIn = utils.create_vlm_predicate(
            "InsideContainer",
            [self._movable_object_type, self._trash_can_type],
            lambda o: _get_vlm_query_str("InsideContainer", o))
        self._TableWiped = utils.create_vlm_predicate(
            "WipedOfMarkerScribbles", [self._table_type],
            lambda o: _get_vlm_query_str("WipedOfMarkerScribbles", o))

    @classmethod
    def get_name(cls) -> str:
        return "spot_vlm_table_wiping_human_invention_env"

    @property
    def types(self) -> Set[Type]:
        return super().types | {
            self._robot_type, self._table_type, self._object_type,
            self._movable_object_type, self._immovable_object_type,
            self._trash_can_type
        }

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        del rng  # unused.
        hand_obj = Object("hand", self._robot_type)
        table_obj = Object("child_play_table", self._table_type)
        apple_obj = Object("apple", self._movable_object_type)
        green_block_obj = Object("green_block", self._movable_object_type)
        orange_block_obj = Object("orange_block", self._movable_object_type)
        spam_tin_obj = Object("spam_tin", self._movable_object_type)
        trash_can_obj = Object("seethru_plastic_dustbin", self._trash_can_type)
        duster_obj = Object("furry_green_eraser", self._movable_object_type)
        cup_obj = Object("red_drink_cup", self._movable_object_type)
        carboard_recycling_bin = Object("cardboard_recycling_bin",
                                        self._trash_can_type)

        ret_tasks = []
        for i in range(num):
            init_state_dict = {
                hand_obj: np.array([]),
                trash_can_obj: np.array([]),
            }
            if i in [0, 2]:
                init_state_dict.update({
                    table_obj: np.array([]),
                    duster_obj: np.array([]),
                    apple_obj: np.array([]),
                    cup_obj: np.array([]),
                })
                goal = {
                    GroundAtom(self._TableWiped, [table_obj]),
                }
            # elif i == 1:
            #     init_state_dict.update({
            #         table_obj: np.array([]),
            #         duster_obj: np.array([]),
            #         apple_obj: np.array([]),
            #         cup_obj: np.array([]),
            #     })
            #     goal = {
            #         GroundAtom(self._VLMIn, [apple_obj, trash_can_obj]),
            #         GroundAtom(self._TableWiped, [table_obj]),
            #     }
            elif i == 1:
                init_state_dict.update({
                    table_obj: np.array([]),
                    duster_obj: np.array([]),
                    apple_obj: np.array([]),
                })
                init_state_dict[green_block_obj] = np.array([])
                goal = {
                    GroundAtom(self._TableWiped, [table_obj]),
                }
            elif i == 3:
                init_state_dict.update({
                    table_obj: np.array([]),
                    duster_obj: np.array([]),
                    apple_obj: np.array([]),
                })
                init_state_dict[green_block_obj] = np.array([])
                goal = {
                    GroundAtom(self._VLMIn, [apple_obj, trash_can_obj]),
                    GroundAtom(self._TableWiped, [table_obj]),
                }
            elif i == 4:
                init_state_dict.update({
                    green_block_obj: np.array([]),
                    carboard_recycling_bin: np.array([]),
                })
                goal = {
                    GroundAtom(self._VLMIn,
                               [green_block_obj, carboard_recycling_bin])
                }
            # elif i == 5:
            #     init_state_dict.update({
            #         orange_block_obj: np.array([]),
            #         carboard_recycling_bin: np.array([]),
            #     })
            #     goal = {
            #         GroundAtom(self._VLMIn,
            #                    [orange_block_obj, carboard_recycling_bin])
            #     }
            elif i == 5:
                init_state_dict.update({
                    spam_tin_obj: np.array([]),
                    carboard_recycling_bin: np.array([]),
                })
                goal = {
                    GroundAtom(self._VLMIn,
                               [spam_tin_obj, carboard_recycling_bin])
                }
            else:
                raise NotImplementedError(
                    "Shouldn't be getting here! i = {}".format(i))

            ret_tasks.append(EnvironmentTask(State(init_state_dict), goal))

        return ret_tasks

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._VLMIn, self._TableWiped}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._VLMIn, self._TableWiped}

    def get_vlm_debug_atom_strs(self,
                                train_tasks: List[Task]) -> List[List[str]]:
        del train_tasks
        atom_strs = set([
            "inAir(apple)", "onTable(apple)", "onFloor(furry_green_eraser)",
            "canBeUsedForErasing(furry_green_eraser)",
            "noObjectsOntopTable(child_play_table)"
        ])
        return [[a] for a in atom_strs]


class SpotVLMJuiceMakingHumanInventionEnv(VLMPredicateEnv):
    """An env that involves making juice using juicer and cup."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # Env-specific types.
        self._robot_type = Type("robot", [], self._object_type)
        self._movable_object_type = Type("movable_object", [],
                                         self._object_type)
        self._container_type = Type("container", [], self._movable_object_type)
        self._immovable_object_type = Type("immovable_object", [],
                                           self._object_type)
        self._juicer_type = Type("juicer", [], self._immovable_object_type)
        self._JuiceIn = utils.create_vlm_predicate(
            "JuiceInCup", [self._container_type],
            lambda o: _get_vlm_query_str("JuiceInside", o))
        self._Empty = utils.create_vlm_predicate(
            "Empty", [self._container_type],
            lambda o: _get_vlm_query_str("Empty", o))

    @classmethod
    def get_name(cls) -> str:
        return "spot_vlm_juice_making_human_invention_env"

    @property
    def types(self) -> Set[Type]:
        return super().types | {
            self._robot_type, self._object_type,
            self._movable_object_type, self._immovable_object_type, self._juicer_type,
            self._container_type
            }

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        del rng  # unused.
        hand_obj = Object("hand", self._robot_type)
        apple_obj = Object("apple", self._movable_object_type)
        juicer_obj = Object("juice_machine", self._juicer_type)
        orange_obj = Object("orange", self._movable_object_type)
        juice_cup_obj = Object("juice_cup", self._container_type)
        waste_cup_obj = Object("waste_cup", self._container_type)
        bowl_obj = Object("bowl", self._container_type)
        paper_cup_obj = Object("paper_cup", self._container_type)
        green_block_obj = Object("green_block", self._movable_object_type)
        glass_cup_obj = Object("glass_cup", self._container_type)
        spam_object = Object("spam_tin", self._movable_object_type)

        ret_tasks = []
        for i in range(num):
            init_state_dict = {
                hand_obj: np.array([]),
            }
            if i in [0, 1]:
                init_state_dict.update({
                    apple_obj: np.array([]),
                    orange_obj: np.array([]),
                    juicer_obj: np.array([]),
                    waste_cup_obj: np.array([]),
                    juice_cup_obj: np.array([]),
                    green_block_obj: np.array([]),
                    bowl_obj: np.array([]),
                    spam_object: np.array([]),
                })
                goal = {
                    GroundAtom(self._JuiceIn, [juice_cup_obj]),
                }
            elif i == 2:
                init_state_dict.update({
                    orange_obj: np.array([]),
                    juicer_obj: np.array([]),
                    waste_cup_obj: np.array([]),
                    juice_cup_obj: np.array([]),
                    spam_object: np.array([]),
                    bowl_obj: np.array([]),
                })
                goal = {
                    GroundAtom(self._JuiceIn, [juice_cup_obj]),
                }
            elif i == 3:
                init_state_dict.update({
                    paper_cup_obj: np.array([]),
                    bowl_obj: np.array([]),
                })
                goal = {
                    GroundAtom(self._Empty, [paper_cup_obj])
                }
            elif i == 4:
                init_state_dict.update({
                    glass_cup_obj: np.array([]),
                    bowl_obj: np.array([]),
                    paper_cup_obj: np.array([]),
                })
                goal = {
                    GroundAtom(self._Empty, [paper_cup_obj])
                }
            else:
                raise NotImplementedError(
                    "Shouldn't be getting here! i = {}".format(i))

            ret_tasks.append(EnvironmentTask(State(init_state_dict), goal))

        return ret_tasks

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._JuiceIn, self._Empty}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._JuiceIn, self._Empty}

    def get_vlm_debug_atom_strs(self,
                                train_tasks: List[Task]) -> List[List[str]]:
        del train_tasks
        atom_strs = set([
            "LidClosed(juice_machine)",
            "MachineTurnedOn(juice_machine)",
            "FruitInsideJuicer(juice_machine, orange)",
            "NotEmpty(waste_cup)",
            "UnderJuiceValve(juice_machine, juice_cup)",
            "UnderWasteValve(juice_machine, waste_cup)",
        ])
        return [[a] for a in atom_strs]