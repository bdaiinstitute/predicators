"""An approach that just executes random options."""

from typing import Callable
import logging

from predicators import utils
from predicators.approaches import ApproachFailure, BaseApproach
from predicators.structs import Action, State, Task


class RandomOptionsApproach(BaseApproach):
    """Samples random options (and random parameters for those options)."""

    @classmethod
    def get_name(cls) -> str:
        return "random_options"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        # Add debugging
        logging.info(f"Available options: {self._initial_options}")
        logging.info(f"Initial state: {task.init}")
        logging.info(f"Goal: {task.goal}")

        def fallback_policy(state: State) -> Action:
            # Add debugging
            logging.error("Failed to sample valid option!")
            # raise ApproachFailure("Random option sampling failed!")

        return utils.create_random_option_policy(self._initial_options,
                                                 self._rng, fallback_policy)
