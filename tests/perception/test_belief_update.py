"""Tests for the merge_vlm_beliefs helper."""

from typing import Dict, Optional

from predicators.perception.belief_update import merge_vlm_beliefs
from predicators.structs import Object, Type, VLMGroundAtom, VLMPredicate


def _make_known_unknown_atoms(base_name: str) -> Dict[str, VLMGroundAtom]:
    """Helper for constructing Known_/Unknown_ atoms for a single object."""
    obj_type = Type("test_type", [])
    test_obj = Object("obj", obj_type)
    known_pred = VLMPredicate(f"Known_{base_name}", [obj_type], prompt="known")
    unknown_pred = VLMPredicate(f"Unknown_{base_name}", [obj_type],
                                prompt="unknown")
    known_atom = VLMGroundAtom(known_pred, [test_obj])
    unknown_atom = VLMGroundAtom(unknown_pred, [test_obj])
    return {
        "known": known_atom,
        "unknown": unknown_atom,
    }


def test_merge_vlm_beliefs_basic_update() -> None:
    """Basic sanity check: definite labels from the latest observation win."""
    atoms = _make_known_unknown_atoms("CupEmpty")
    prior: Dict[VLMGroundAtom, bool] = {}
    new_values = {
        atoms["known"]: True,
        atoms["unknown"]: False,
    }
    merged = merge_vlm_beliefs(prior, new_values)
    assert merged == {
        atoms["known"]: True,
        atoms["unknown"]: False,
    }


def test_merge_vlm_beliefs_handles_inconsistent_inputs() -> None:
    """If a new observation says Unknown or not Known, prefer the Unknown."""
    atoms = _make_known_unknown_atoms("CupEmpty")
    prior: Dict[VLMGroundAtom, bool] = {}
    # Observation is inconsistent: Known False / Unknown True should collapse
    # to Unknown True.
    new_values = {
        atoms["known"]: False,
        atoms["unknown"]: True,
    }
    merged = merge_vlm_beliefs(prior, new_values)
    assert merged == {
        atoms["known"]: False,
        atoms["unknown"]: True,
    }


def test_merge_vlm_beliefs_preserves_previous_knowledge() -> None:
    """Once Known was True, keep it True unless explicitly overwritten."""
    atoms = _make_known_unknown_atoms("CupEmpty")
    prior = {
        atoms["known"]: True,
        atoms["unknown"]: False,
    }
    # New observation inconclusive (e.g., the object is occluded). Values are
    # None, so prior certainty should remain.
    new_values: Dict[VLMGroundAtom, Optional[bool]] = {
        atoms["known"]: None,
        atoms["unknown"]: None,
    }
    merged = merge_vlm_beliefs(prior, new_values)
    assert merged == prior
