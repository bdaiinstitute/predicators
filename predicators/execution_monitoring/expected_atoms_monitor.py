"""Execution monitor variants that rely on expected atoms from the plan."""

import logging
from typing import Optional, Set, Tuple

from predicators import utils
from predicators.execution_monitoring.base_execution_monitor import \
    BaseExecutionMonitor
from predicators.settings import CFG
from predicators.structs import GroundAtom, State, VLMPredicate


class ExpectedAtomsExecutionMonitor(BaseExecutionMonitor):
    """Suggests replanning when the expected atoms check fails."""

    @classmethod
    def get_name(cls) -> str:
        return "expected_atoms"

    def _get_expected_atoms_entry(
            self) -> Optional[Tuple[Set[GroundAtom], bool]]:
        """Return the expected atoms and whether the step is info-gathering."""
        entry = self._entry_for_next_check
        if entry is None:
            return None
        if isinstance(entry, dict):
            expected_atoms = entry.get("expected_atoms")
            if expected_atoms is None:
                return None
            is_info = entry.get("is_information_gathering", True)
            return expected_atoms, is_info
        if isinstance(entry, set):
            return entry, True
        return None

    def _evaluate_expected_atoms(self, state: State,
                                 expected_atoms: Set[GroundAtom]) -> bool:
        """Run the expected-atoms check and return whether to replan."""
        next_expected_vlm_atoms = {
            atom for atom in expected_atoms
            if isinstance(atom.predicate, VLMPredicate)
        }
        non_vlm_unsat_atoms = {
            atom
            for atom in expected_atoms - next_expected_vlm_atoms
            if not atom.holds(state)
        }
        vlm_unsat_atoms = set()
        if next_expected_vlm_atoms:
            vlm_unsat_atoms = expected_atoms - (
                utils.query_vlm_for_atom_vals(next_expected_vlm_atoms, state))
        unsat_atoms = non_vlm_unsat_atoms | vlm_unsat_atoms
        if unsat_atoms:
            logging.info(
                "Expected atoms execution monitor triggered replanning "
                f"because of these atoms: {unsat_atoms}")
            return True
        return False

    def step(self, state: State) -> bool:
        # This monitor only makes sense to use with approaches that expose a
        # high-level plan (e.g., oracle bilevel planning).
        assert "oracle" in CFG.approach or "active_sampler" in CFG.approach \
            or "maple_q" in CFG.approach or "grammar_search_invention" in \
            CFG.approach or "llm" in CFG.approach or "vlm" in CFG.approach

        if self._skip_next_entry_check:
            self._skip_next_entry_check = False
            return False
        parsed = self._get_expected_atoms_entry()
        if parsed is None:
            return False
        expected_atoms, _ = parsed
        self._curr_plan_timestep += 1
        should_replan = self._evaluate_expected_atoms(state, expected_atoms)
        self._prepare_next_entry()
        return should_replan


class InformationOnlyExecutionMonitor(ExpectedAtomsExecutionMonitor):
    """Only checks expected atoms after information-gathering actions."""

    @classmethod
    def get_name(cls) -> str:
        return "information_only"

    def step(self, state: State) -> bool:
        if self._skip_next_entry_check:
            self._skip_next_entry_check = False
            return False
        parsed = self._get_expected_atoms_entry()
        if parsed is None:
            return False
        expected_atoms, is_information_gathering = parsed
        self._curr_plan_timestep += 1
        if not is_information_gathering:
            self._prepare_next_entry()
            return False
        should_replan = self._evaluate_expected_atoms(state, expected_atoms)
        self._prepare_next_entry()
        return should_replan
