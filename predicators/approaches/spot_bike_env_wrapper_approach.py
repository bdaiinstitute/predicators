"""An approach that "wraps" a base approach for the SpotBikeEnv. The wrapper
always executes a "find" action when receiving a state with unknown tool
position. Otherwise it passes control to the regular approach. The way to use
this environment from the command-line is using the flag.

    --approach 'spot_bike_wrapper[base approach name]'

e.g.

    --approach 'spot_bike_wrapper[oracle]'
"""

from typing import Callable, Optional

import numpy as np

from predicators.approaches import BaseApproachWrapper
from predicators.structs import Action, State, Task


class NoisyButtonWrapperApproach(BaseApproachWrapper):
    """Always "find" when the button position is unknown."""

    @classmethod
    def get_name(cls) -> str:
        return "spot_bike_wrapper"

    @property
    def is_learning_based(self) -> bool:
        return self._base_approach.is_learning_based

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:

        # Maintain policy from the base approach.
        base_approach_policy: Optional[Callable[[State], Action]] = None

        def _policy(state: State) -> Action:
            nonlocal base_approach_policy
            objs = list(state)
            for obj in objs:
                if "position_known" in obj.type.feature_names and state.get(obj, "position_known") < 0.5:
                    # TODO: return a Find action for the particular object.
                    # Reset the base approach policy.
                    import ipdb; ipdb.set_trace()
                    base_approach_policy = None
            if base_approach_policy is None:
                cur_task = Task(state, task.goal)
                base_approach_policy = self._base_approach.solve(
                    cur_task, timeout)
            # Use the base policy.
            return base_approach_policy(state)

        return _policy
