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
        self._movable_object_type = Type("movable", [],
                                         self._object_type)
        self._immovable_object_type = Type("immovable", [],
                                           self._object_type)
        self._table_type = Type("table", [], self._immovable_object_type)
        self._trash_can_type = Type("trashcan", [],
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
        self._movable_object_type = Type("movable", [],
                                         self._object_type)
        self._immovable_object_type = Type("immovable", [],
                                           self._object_type)
        self._table_type = Type("table", [], self._immovable_object_type)
        self._trash_can_type = Type("trashcan", [],
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


class LISThesisPickPlaceEnv(VLMPredicateEnv):
    """Environment for LIS thesis pick-and-place task with Spot robot.

    This is for predicate invention from VLM demos involving picking
    and placing balls on tables.
    """

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # Env-specific types.
        self._robot_type = Type("robot", [], self._object_type)
        self._ball_type = Type("ball", [], self._object_type)
        self._table_type = Type("table", [], self._object_type)

        # VLM-based goal predicate: OnTable(ball, table)
        def _get_vlm_query_str(pred_name: str, objects) -> str:
            return pred_name + "(" + ", ".join(
                str(obj.name) for obj in objects) + ")"

        self._OnTable = utils.create_vlm_predicate(
            "OnTable",
            [self._ball_type, self._table_type],
            lambda o: _get_vlm_query_str("OnTable", o))

    @classmethod
    def get_name(cls) -> str:
        return "lis_thesis_pickplace"

    @property
    def types(self) -> Set[Type]:
        return super().types | {
            self._robot_type,
            self._ball_type,
            self._table_type,
        }

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._OnTable}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._OnTable}

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        del rng  # unused.
        # Define objects matching the demo
        spot_obj = Object("spot", self._robot_type)
        tenis_ball_obj = Object("tenis_ball", self._ball_type)
        red_ball_obj = Object("red_ball", self._ball_type)
        yellow_table_obj = Object("yellow_table", self._table_type)

        ret_tasks = []
        for _ in range(num):
            init_state_dict = {
                spot_obj: np.array([]),
                tenis_ball_obj: np.array([]),
                red_ball_obj: np.array([]),
                yellow_table_obj: np.array([]),
            }
            # Goal: both balls should be on the yellow table
            goal = {
                GroundAtom(self._OnTable, [tenis_ball_obj, yellow_table_obj]),
                GroundAtom(self._OnTable, [red_ball_obj, yellow_table_obj]),
            }
            ret_tasks.append(EnvironmentTask(State(init_state_dict), goal))
        return ret_tasks

    def get_vlm_debug_atom_strs(self,
                                train_tasks: List[Task]) -> List[List[str]]:
        """Debug atoms that might be relevant for this domain."""
        del train_tasks
        atom_strs = set([
            "holding(tenis_ball)",
            "holding(red_ball)",
            "OnTable(tenis_ball, yellow_table)",
            "OnTable(red_ball, yellow_table)",
            "robot_at(spot, yellow_table)",
            "robot_at(spot, tenis_ball)",
            "robot_at(spot, red_ball)",
        ])
        return [[a] for a in atom_strs]


class LISThesisSweepEnv(VLMPredicateEnv):
    """Environment for LIS thesis sweep task with Spot robot.

    Goal: sweep table and put toys in the plastic bin.
    """

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # Env-specific types.
        self._robot_type = Type("robot", [], self._object_type)
        self._tool_type = Type("tool", [], self._object_type)
        self._surface_type = Type("surface", [], self._object_type)
        self._toy_type = Type("toy", [], self._object_type)
        self._container_type = Type("container", [], self._object_type)

        # VLM-based goal predicates
        def _get_vlm_query_str(pred_name: str, objects) -> str:
            return pred_name + "(" + ", ".join(
                str(obj.name) for obj in objects) + ")"

        self._InContainer = utils.create_vlm_predicate(
            "InContainer",
            [self._toy_type, self._container_type],
            lambda o: _get_vlm_query_str("InContainer", o))

        self._TableSwept = utils.create_vlm_predicate(
            "TableSwept",
            [self._surface_type],
            lambda o: _get_vlm_query_str("TableSwept", o))

    @classmethod
    def get_name(cls) -> str:
        return "lis_thesis_sweep"

    @property
    def types(self) -> Set[Type]:
        return super().types | {
            self._robot_type,
            self._tool_type,
            self._surface_type,
            self._toy_type,
            self._container_type,
        }

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._InContainer, self._TableSwept}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._InContainer, self._TableSwept}

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        del rng  # unused.
        # Define objects matching the demo
        spot_obj = Object("spot", self._robot_type)
        squeegee_obj = Object("squeegee", self._tool_type)
        wooden_table_obj = Object("wooden_table", self._surface_type)
        floor_obj = Object("floor", self._surface_type)
        toy_caterpillar_obj = Object("toy_caterpillar", self._toy_type)
        toy_elephant_obj = Object("toy_elephant", self._toy_type)
        toy_frog_obj = Object("toy_frog", self._toy_type)
        plastic_bin_obj = Object("plastic_bin", self._container_type)

        ret_tasks = []
        for _ in range(num):
            init_state_dict = {
                spot_obj: np.array([]),
                squeegee_obj: np.array([]),
                wooden_table_obj: np.array([]),
                floor_obj: np.array([]),
                toy_caterpillar_obj: np.array([]),
                toy_elephant_obj: np.array([]),
                toy_frog_obj: np.array([]),
                plastic_bin_obj: np.array([]),
            }
            # Goal: all three toys in the plastic bin
            goal = {
                GroundAtom(self._InContainer,
                           [toy_caterpillar_obj, plastic_bin_obj]),
                GroundAtom(self._InContainer,
                           [toy_elephant_obj, plastic_bin_obj]),
                GroundAtom(self._InContainer,
                           [toy_frog_obj, plastic_bin_obj]),
            }
            ret_tasks.append(EnvironmentTask(State(init_state_dict), goal))
        return ret_tasks

    def get_vlm_debug_atom_strs(self,
                                train_tasks: List[Task]) -> List[List[str]]:
        """Debug atoms that might be relevant for this domain."""
        del train_tasks
        atom_strs = set([
            "holding(squeegee)",
            "holding(toy_caterpillar)",
            "holding(toy_elephant)",
            "holding(toy_frog)",
            "InContainer(toy_caterpillar, plastic_bin)",
            "InContainer(toy_elephant, plastic_bin)",
            "InContainer(toy_frog, plastic_bin)",
            "TableSwept(wooden_table)",
            "OnSurface(squeegee, floor)",
            "OnSurface(squeegee, wooden_table)",
        ])
        return [[a] for a in atom_strs]


class LISThesisScrubEnv(VLMPredicateEnv):
    """Environment for LIS thesis scrub task with Spot robot.

    Goal: clean table and put scrub_sponge back in orange_bucket.
    """

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # Env-specific types.
        self._robot_type = Type("robot", [], self._object_type)
        self._tool_type = Type("tool", [], self._object_type)
        self._surface_type = Type("surface", [], self._object_type)
        self._container_type = Type("container", [], self._object_type)
        self._furniture_type = Type("furniture", [], self._object_type)

        # VLM-based goal predicates
        def _get_vlm_query_str(pred_name: str, objects) -> str:
            return pred_name + "(" + ", ".join(
                str(obj.name) for obj in objects) + ")"

        self._InContainer = utils.create_vlm_predicate(
            "InContainer",
            [self._tool_type, self._container_type],
            lambda o: _get_vlm_query_str("InContainer", o))

        self._TableClean = utils.create_vlm_predicate(
            "TableClean",
            [self._surface_type],
            lambda o: _get_vlm_query_str("TableClean", o))

    @classmethod
    def get_name(cls) -> str:
        return "lis_thesis_scrub"

    @property
    def types(self) -> Set[Type]:
        return super().types | {
            self._robot_type,
            self._tool_type,
            self._surface_type,
            self._container_type,
            self._furniture_type,
        }

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._InContainer, self._TableClean}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._InContainer, self._TableClean}

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        del rng  # unused.
        # Define objects matching the demo
        spot_obj = Object("spot", self._robot_type)
        blue_chair_obj = Object("blue_chair", self._furniture_type)
        wooden_table_obj = Object("wooden_table", self._surface_type)
        orange_bucket_obj = Object("orange_bucket", self._container_type)
        scrub_sponge_obj = Object("scrub_sponge", self._tool_type)

        ret_tasks = []
        for _ in range(num):
            init_state_dict = {
                spot_obj: np.array([]),
                blue_chair_obj: np.array([]),
                wooden_table_obj: np.array([]),
                orange_bucket_obj: np.array([]),
                scrub_sponge_obj: np.array([]),
            }
            # Goal: clean table and scrub_sponge in orange_bucket
            goal = {
                GroundAtom(self._TableClean, [wooden_table_obj]),
                GroundAtom(self._InContainer,
                           [scrub_sponge_obj, orange_bucket_obj]),
            }
            ret_tasks.append(EnvironmentTask(State(init_state_dict), goal))
        return ret_tasks

    def get_vlm_debug_atom_strs(self,
                                train_tasks: List[Task]) -> List[List[str]]:
        """Debug atoms that might be relevant for this domain."""
        del train_tasks
        atom_strs = set([
            "holding(scrub_sponge)",
            "InContainer(scrub_sponge, orange_bucket)",
            "TableClean(wooden_table)",
            "Blocking(blue_chair, wooden_table)",
            "TableAccessible(wooden_table)",
        ])
        return [[a] for a in atom_strs]