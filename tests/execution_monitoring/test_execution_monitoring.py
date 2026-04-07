"""Tests for execution monitors."""

import pytest

from predicators.execution_monitoring import create_execution_monitor
from predicators.execution_monitoring.expected_atoms_monitor import \
    ExpectedAtomsExecutionMonitor, InformationOnlyExecutionMonitor
from predicators.execution_monitoring.mpc_execution_monitor import \
    MpcExecutionMonitor
from predicators.execution_monitoring.trivial_execution_monitor import \
    TrivialExecutionMonitor
from predicators import utils
from predicators.settings import CFG
from predicators.structs import GroundAtom, Object, Predicate, State, Type


def test_create_execution_monitor():
    """Tests for create_execution_monitor()."""
    exec_monitor = create_execution_monitor("trivial")
    assert isinstance(exec_monitor, TrivialExecutionMonitor)

    exec_monitor = create_execution_monitor("mpc")
    assert isinstance(exec_monitor, MpcExecutionMonitor)

    exec_monitor = create_execution_monitor("expected_atoms")
    assert isinstance(exec_monitor, ExpectedAtomsExecutionMonitor)

    exec_monitor = create_execution_monitor("information_only")
    assert isinstance(exec_monitor, InformationOnlyExecutionMonitor)

    with pytest.raises(NotImplementedError) as e:
        create_execution_monitor("not a real monitor")
    assert "Unrecognized execution monitor" in str(e)


def test_information_only_monitor_behavior():
    """Ensure expected-atoms monitors parse rich info correctly."""
    CFG.approach = "oracle"

    dummy_type = Type("dummy", ["feat"])
    obj = Object("obj", dummy_type)

    def _true_classifier(state: State, objs) -> bool:  # pragma: no cover
        return True

    def _false_classifier(state: State, objs) -> bool:  # pragma: no cover
        return False

    sat_pred = Predicate("Sat", [dummy_type], _true_classifier)
    unsat_pred = Predicate("Unsat", [dummy_type], _false_classifier)
    sat_atom = GroundAtom(sat_pred, [obj])
    unsat_atom = GroundAtom(unsat_pred, [obj])
    state = utils.create_state_from_dict({obj: {"feat": 0.0}})

    expected_monitor = ExpectedAtomsExecutionMonitor()
    expected_monitor.update_approach_info([{
        "current_nsrt_plan": [],
    }])
    assert not expected_monitor.step(state)

    expected_monitor.update_approach_info([{
        "expected_atoms": {sat_atom},
        "is_information_gathering": True,
    }])
    assert not expected_monitor.step(state)

    info_monitor = InformationOnlyExecutionMonitor()
    info_monitor.update_approach_info([{
        "expected_atoms": {unsat_atom},
        "is_information_gathering": False,
    }])
    assert not info_monitor.step(state)

    info_monitor.update_approach_info([{
        "expected_atoms": {unsat_atom},
        "is_information_gathering": True,
    }])
    assert info_monitor.step(state)
