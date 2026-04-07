"""Tests for the perception monitor helper."""

from predicators.perception.perception_monitor import PerceptionMonitor
from predicators.settings import CFG
from predicators.utils import reset_config


def test_perception_monitor_initial_scan_and_default_refresh():
    """Ensure the monitor refreshes until the initial scan finishes."""
    reset_config()
    monitor = PerceptionMonitor()
    monitor.reset(initial_scan_done=False)
    decision = monitor.decide(has_cached_observation=False,
                              is_information_gathering=False,
                              is_hand_view_action=False)
    assert decision.refresh_full
    assert not decision.reuse_cache
    monitor.mark_refresh_complete()
    decision = monitor.decide(has_cached_observation=True,
                              is_information_gathering=False,
                              is_hand_view_action=False)
    assert decision.refresh_full  # legacy behaviour (no gating flags)


def test_perception_monitor_observe_only_reuse():
    """Observe-only flag should reuse cache when available."""
    reset_config({"spot_perception_refresh_observe_only": True})
    monitor = PerceptionMonitor()
    decision = monitor.decide(has_cached_observation=True,
                              is_information_gathering=False,
                              is_hand_view_action=False)
    assert not decision.refresh_full
    assert decision.reuse_cache

    decision = monitor.decide(has_cached_observation=False,
                              is_information_gathering=False,
                              is_hand_view_action=False)
    assert decision.refresh_full
    assert not decision.reuse_cache


def test_perception_monitor_hand_view_pointing():
    """Hand-view actions should refresh and trigger pointing."""
    reset_config({
        "spot_use_vlm_pointing": True,
        "spot_perception_refresh_observe_only": True
    })
    monitor = PerceptionMonitor()
    decision = monitor.decide(has_cached_observation=True,
                              is_information_gathering=False,
                              is_hand_view_action=True)
    assert decision.refresh_full
    assert decision.trigger_pointing
