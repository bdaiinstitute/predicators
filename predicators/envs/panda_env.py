"""Basic environment for Panda arm robot."""

import abc
import functools
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Collection, Dict, Iterator, List, \
    Optional, Sequence, Set, Tuple

import matplotlib
import numpy as np
import numpy as np
from bosdyn.api.geometry_pb2 import FrameTreeSnapshot
from bosdyn.client import math_helpers
from numpy.typing import NDArray
from scipy import ndimage

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.spot_utils.perception.perception_structs import RGBDImageWithContext
from predicators.structs import Action, EnvironmentTask, GoalDescription, \
    GroundAtom, LiftedAtom, Object, Observation, Predicate, \
    SpotActionExtraInfo, State, STRIPSOperator, Type, Variable, Any


@dataclass(frozen=True)
class _PandaObservation:
    """An observation for a PandaEnv. Adapted from _SpotObservation."""
    # Camera name to image
    # TODO this is from spot_utils
    images: Dict[str, RGBDImageWithContext]
    # Objects that are seen in the current image and their positions in world
    objects_in_view: Dict[Object, math_helpers.SE3Pose]
    # Objects seen only by the hand camera
    objects_in_hand_view: Set[Object]
    # Objects seen by any camera except the back camera
    objects_in_any_view_except_back: Set[Object]
    # Expose the robot object.
    robot: Object
    # Status of the robot gripper.
    gripper_open_percentage: float
    # Robot SE3 Pose
    robot_pos: math_helpers.SE3Pose
    # Ground atoms without ground-truth classifiers
    # A placeholder until all predicates have classifiers
    nonpercept_atoms: Set[GroundAtom]
    nonpercept_predicates: Set[Predicate]


class _PartialPerceptionState(State):
    """Some continuous object features, and ground atoms in simulator_state.

    The main idea here is that we have some predicates with actual
    classifiers implemented, but not all.

    NOTE: these states are only created in the perceiver, but they are used
    in the classifier definitions for the dummy predicates
    """

    @property
    def _simulator_state_predicates(self) -> Set[Predicate]:
        assert isinstance(self.simulator_state, Dict)
        return self.simulator_state["predicates"]

    @property
    def _simulator_state_atoms(self) -> Set[GroundAtom]:
        assert isinstance(self.simulator_state, Dict)
        return self.simulator_state["atoms"]

    def simulator_state_atom_holds(self, atom: GroundAtom) -> bool:
        """Check whether an atom holds in the simulator state."""
        assert atom.predicate in self._simulator_state_predicates
        return atom in self._simulator_state_atoms

    def allclose(self, other: State) -> bool:
        if self.simulator_state != other.simulator_state:
            return False
        return self._allclose(other)

    def copy(self) -> State:
        state_copy = {o: self._copy_state_value(self.data[o]) for o in self}
        sim_state_copy = {
            "predicates": self._simulator_state_predicates.copy(),
            "atoms": self._simulator_state_atoms.copy()
        }
        return _PartialPerceptionState(state_copy,
                                       simulator_state=sim_state_copy)


def _create_dummy_predicate_classifier(
        pred: Predicate) -> Callable[[State, Sequence[Object]], bool]:

    def _classifier(s: State, objs: Sequence[Object]) -> bool:
        assert isinstance(s, _PartialPerceptionState)
        atom = GroundAtom(pred, objs)
        return s.simulator_state_atom_holds(atom)

    return _classifier


@functools.lru_cache(maxsize=None)
def get_robot(
) -> Any:
    """Create the robot only once.

    If we are doing a dry run, return dummy Nones for each component.
    """
    # TODO: this is a placeholder for the robot object
    return 0


# Robot type; adapted from `spot_utils/utils.py`
_robot_type = Type(
    "robot",
    ["gripper_open_percentage", "x", "y", "z", "qw", "qx", "qy", "qz"])


class PandaRearrangementEnv(BaseEnv):
    """Basic environment for Panda arm robot.

    This is adapted from the Spot environment.
    """

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        if not ("panda_wrapper" in CFG.approach and "panda_wrapper" in CFG.approach):
            # TODO: placeholder for Panda wrapper
            pass

        robot = get_robot()
        self._robot = robot

        # Note that we need to include the operators in this
        # class because they're used to update the symbolic
        # parts of the state during execution.
        self._strips_operators: Set[STRIPSOperator] = set()
        self._current_task_goal_reached = False
        self._last_action: Optional[Action] = None

        # Create constant objects.
        self._spot_object = Object("robot", _robot_type)

        # For noisy simulation in dry runs.
        self._noise_rng = np.random.default_rng(CFG.seed)

        # Used for the move-related hacks in step().
        self._last_known_object_poses: Dict[Object, math_helpers.SE3Pose] = {}
