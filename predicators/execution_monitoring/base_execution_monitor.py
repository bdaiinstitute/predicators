"""Base class for execution monitors."""

import abc
from typing import Any, List, Optional

from predicators.structs import State, Task


class BaseExecutionMonitor(abc.ABC):
    """An execution monitor consumes states and decides whether to replan."""

    def __init__(self) -> None:
        self._approach_info: List[Any] = []
        self._pending_entries: List[Any] = []
        self._entry_for_next_check: Optional[Any] = None
        self._plan_metadata: Optional[Any] = None
        self._curr_plan_timestep = 0
        # Skip the very first monitor check so we do not evaluate the plan
        # before the agent executes any action.
        self._skip_next_entry_check = True

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this execution monitor."""

    def reset(self, task: Task) -> None:
        """Reset after replanning."""
        del task  # unused
        self._curr_plan_timestep = 0
        self._approach_info = []
        self._pending_entries = []
        self._entry_for_next_check = None
        self._plan_metadata = None
        self._skip_next_entry_check = True

    @abc.abstractmethod
    def step(self, state: State) -> bool:
        """Return true if the agent should replan."""

    def update_approach_info(self, info: List[Any]) -> None:
        """Update internal info received from approach."""
        self._approach_info = list(info)
        self._pending_entries = list(info)
        self._plan_metadata = None
        self._prepare_next_entry()

    def _prepare_next_entry(self) -> None:
        """Advance to the next monitor entry that contains expected atoms."""
        self._entry_for_next_check = None
        while self._pending_entries:
            entry = self._pending_entries.pop(0)
            if isinstance(entry, dict) and "expected_atoms" not in entry:
                # Treat as plan metadata; keep latest copy.
                self._plan_metadata = entry
                continue
            self._entry_for_next_check = entry
            break
