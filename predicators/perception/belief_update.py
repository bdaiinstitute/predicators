"""Shared helpers for maintaining VLM-based belief state.

The mock and real Spot perceivers both rely on the same Known_*/Unknown_* logic
to keep track of partially observable predicates. Centralizing the merge logic
here ensures the two code paths remain in sync and gives us a single place to
unit test the behavior.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Union, List, Tuple

from predicators.structs import VLMGroundAtom


def merge_vlm_beliefs(
        prior: Dict[VLMGroundAtom, bool],
        new_values: Union[Dict[VLMGroundAtom, bool],
                          Dict[VLMGroundAtom, Optional[bool]]],
        logger: Optional[logging.Logger] = None) -> Dict[VLMGroundAtom, bool]:
    """Match the mock perceiver's three-step belief update exactly.

    This helper is a straight port of the logic in
    MockSpotPerceiver._obs_to_state() so that both code paths behave
    identically. Keep the comments and naming in sync with the mock version.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # First get the latest VLM atom values from current observation
    updated_vlm_atom_values = prior.copy()  # Start with previous values

    curr_vlm_atom_values: Dict[VLMGroundAtom, Optional[bool]] = dict(
        new_values)

    # Update with new values from current observation's VLM evaluation
    # This preserves knowledge from previous observations
    # NOTE: We assume no information loss in belief update
    # NOTE: Rule: Known predicates cannot become unknown

    # Step 1: Check consistency of newly detected labels
    # Collect Known/Unknown pairs from current VLM evaluation
    curr_known_unknown_pairs: Dict[str, List[Optional[VLMGroundAtom]]] = {}
    for atom in curr_vlm_atom_values:
        pred_name = atom.predicate.name
        if pred_name.startswith("Known_"):
            base_name = pred_name.replace("Known_", "")
            if base_name not in curr_known_unknown_pairs:
                curr_known_unknown_pairs[base_name] = [atom, None]
            else:
                curr_known_unknown_pairs[base_name][0] = atom
        elif pred_name.startswith("Unknown_"):
            base_name = pred_name.replace("Unknown_", "")
            if base_name not in curr_known_unknown_pairs:
                curr_known_unknown_pairs[base_name] = [None, atom]
            else:
                curr_known_unknown_pairs[base_name][1] = atom

    # Check consistency of current VLM evaluation
    # Being pessimistic: if unknown is true OR known is false, treat as unknown
    for base_name, (known_atom,
                    unknown_atom) in curr_known_unknown_pairs.items():
        if known_atom is not None and unknown_atom is not None:
            known_val = curr_vlm_atom_values.get(known_atom)
            unknown_val = curr_vlm_atom_values.get(unknown_atom)
            if known_val is not None and unknown_val is not None:
                # Both True or False is inconsistent
                if (known_val and unknown_val) or (not known_val
                                                   and not unknown_val):
                    logger.warning(
                        "Inconsistent Known/Unknown values in current VLM evaluation for %s: "
                        "Both Known and Unknown are True or False", base_name)
                # Being pessimistic: if unknown is true OR known is false, set as unknown
                if unknown_val or not known_val:
                    curr_vlm_atom_values[known_atom] = False
                    curr_vlm_atom_values[unknown_atom] = True

    # Step 2: Basic update - update any atom that has a non-None value
    # NOTE: don't use obs.vlm_atom_dict, it's deprecated!
    for atom, value in curr_vlm_atom_values.items():
        if value is not None:
            updated_vlm_atom_values[atom] = value

    # Step 3: Override with previous knowledge for Known/Unknown pairs
    # Collect all Known/Unknown pairs from previous step's values
    known_unknown_pairs: Dict[str, List[Optional[VLMGroundAtom]]] = {}
    for atom in prior:
        pred_name = atom.predicate.name
        if pred_name.startswith("Known_"):
            base_name = pred_name.replace("Known_", "")
            if base_name not in known_unknown_pairs:
                known_unknown_pairs[base_name] = [atom, None]
            else:
                known_unknown_pairs[base_name][0] = atom
        elif pred_name.startswith("Unknown_"):
            base_name = pred_name.replace("Unknown_", "")
            if base_name not in known_unknown_pairs:
                known_unknown_pairs[base_name] = [None, atom]
            else:
                known_unknown_pairs[base_name][1] = atom

    # Update Known/Unknown pairs based on previous knowledge
    for base_name, (known_atom, unknown_atom) in known_unknown_pairs.items():
        if known_atom is not None and unknown_atom is not None:
            # If it was known in previous step, keep it known
            if prior.get(known_atom, False):  # Use previous step's values
                updated_vlm_atom_values[known_atom] = True
                updated_vlm_atom_values[unknown_atom] = False
            # Otherwise, we keep the value from current observation

    return updated_vlm_atom_values
