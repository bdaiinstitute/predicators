"""An execution monitor that leverages knowledge of the high-level plan to only
suggest replanning when the expected atoms check is not met."""

import logging

from predicators.execution_monitoring.base_execution_monitor import \
    BaseExecutionMonitor
from predicators.settings import CFG
from predicators.structs import State


class ExpectedAtomsExecutionMonitor(BaseExecutionMonitor):
    """An execution monitor that only suggests replanning when we're doing
    bilevel planning and the expected atoms check fails."""

    @classmethod
    def get_name(cls) -> str:
        return "expected_atoms"

    def step(self, state: State) -> bool:
        # This monitor only makes sense to use with an oracle
        # bilevel planning approach.
        assert "oracle" in CFG.approach or "active_sampler" in CFG.approach \
            or "maple_q" in CFG.approach
        # If the approach info is empty, don't replan.
        if not self._approach_info:  # pragma: no cover
            return False
        next_expected_atoms = self._approach_info[0]
        assert isinstance(next_expected_atoms, set)
        self._curr_plan_timestep += 1
        # If the expected atoms are a subset of the current atoms, then
        # we don't have to replan.
        unsat_atoms = {a for a in next_expected_atoms if not a.holds(state)}
        # Check goal
        assert self.perceiver is not None and self.env_task is not None
        new_goal = self.perceiver._create_goal(state, self.env_task.goal_description)
        if new_goal != self._curr_goal:
            logging.info(
                "Expected atoms execution monitor triggered replanning "
                "because the goal has changed.")
            logging.info(f"Old goal: {self._curr_goal}")
            logging.info(f"New goal: {new_goal}")
            self._curr_goal = new_goal
        if not unsat_atoms:
            return False
        logging.info(
            "Expected atoms execution monitor triggered replanning "
            f"because of these atoms: {unsat_atoms}")  # pragma: no cover
        return True  # pragma: no cover
