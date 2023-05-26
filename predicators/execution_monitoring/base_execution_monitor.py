"""Base class for execution monitors."""

import abc
from typing import Any, List

from predicators.structs import State, Task


class BaseExecutionMonitor(abc.ABC):
    """An execution monitor consumes states and decides whether to replan."""

    def __init__(self):
        self._approach_info: List[Any] = []

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this execution monitor."""

    @abc.abstractmethod
    def reset(self, task: Task) -> None:
        """Reset after replanning."""

    @abc.abstractmethod
    def step(self, state: State) -> bool:
        """Return true if the agent should replan."""

    def update_approach_info(self, info: List[Any]) -> None:
        """Update internal info received from approach."""
        self._approach_info = info
