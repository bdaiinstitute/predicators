"""Monitor that decides when Spot should run perception updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from predicators.settings import CFG


@dataclass(frozen=True)
class PerceptionDecision:
    """Single-step perception request."""
    refresh_full: bool
    reuse_cache: bool
    trigger_pointing: bool
    reason: str = ""

    @property
    def requires_observation(self) -> bool:
        return self.refresh_full


class PerceptionMonitor:
    """Small helper that mirrors the execution monitor for perception cadence."""

    def __init__(self) -> None:
        self._initial_scan_done = True

    def reset(self, *, initial_scan_done: bool = True) -> None:
        self._initial_scan_done = initial_scan_done

    def mark_refresh_complete(self) -> None:
        self._initial_scan_done = True

    def decide(self, *, has_cached_observation: bool,
               is_information_gathering: bool,
               is_hand_view_action: bool) -> PerceptionDecision:
        """Compute the perception request for this env step."""
        trigger_pointing = is_hand_view_action and CFG.spot_use_vlm_pointing

        if not self._initial_scan_done:
            return PerceptionDecision(refresh_full=True,
                                      reuse_cache=False,
                                      trigger_pointing=trigger_pointing,
                                      reason="initial perception scan")

        if is_hand_view_action:
            # Always refresh when we rotate the hand camera to observe a target.
            return PerceptionDecision(refresh_full=True,
                                      reuse_cache=False,
                                      trigger_pointing=trigger_pointing,
                                      reason="hand-view observation")

        if is_information_gathering:
            return PerceptionDecision(refresh_full=True,
                                      reuse_cache=False,
                                      trigger_pointing=trigger_pointing,
                                      reason="information-gathering operator")

        if CFG.spot_perception_refresh_observe_only:
            if has_cached_observation:
                return PerceptionDecision(refresh_full=False,
                                          reuse_cache=True,
                                          trigger_pointing=trigger_pointing,
                                          reason="observe-only reuse")
            # No cache available; fall back to a full refresh.
            return PerceptionDecision(refresh_full=True,
                                      reuse_cache=False,
                                      trigger_pointing=trigger_pointing,
                                      reason="observe-only but cache empty")

        if CFG.spot_skip_perception_for_manipulation and has_cached_observation:
            return PerceptionDecision(refresh_full=False,
                                      reuse_cache=True,
                                      trigger_pointing=trigger_pointing,
                                      reason="manipulation skip w/ cache")

        return PerceptionDecision(refresh_full=True,
                                  reuse_cache=False,
                                  trigger_pointing=trigger_pointing,
                                  reason="default refresh")
