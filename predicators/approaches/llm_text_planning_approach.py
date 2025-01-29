"""A text-based LLM planning approach that uses natural language state descriptions."""

from typing import List, Optional, Set, Tuple

from predicators import utils
from predicators.approaches.llm_open_loop_approach import LLMOpenLoopApproach
from predicators.settings import CFG
from predicators.structs import GroundAtom, State, Task


class LLMTextPlanningApproach(LLMOpenLoopApproach):
    """An approach that uses text descriptions of states for LLM planning."""

    @classmethod
    def get_name(cls) -> str:
        return "llm_text"

    def _create_prompt(self, state: State, atoms: Set[GroundAtom],
                      goal: Set[GroundAtom]) -> str:
        """Create a prompt using the text description of the state."""
        # Get text description, fall back to structured state if none available
        state_desc = state.text_description
        if state_desc is None:
            state_desc = self._structured_state_to_text(state, atoms)

        # Convert goal atoms to text
        goal_desc = self._goal_atoms_to_text(goal)

        # Create prompt
        prompt = f"""
Current state:
{state_desc}

Goal:
{goal_desc}

Solution:"""

        return prompt

    def _structured_state_to_text(self, state: State,
                                atoms: Set[GroundAtom]) -> str:
        """Convert a structured state to text if no VLM description available."""
        # Convert state dict to text
        state_lines = []
        for obj in state:
            features = [f"{fname}: {state.get(obj, fname)}"
                       for fname in obj.type.feature_names]
            state_lines.append(f"{obj.name} ({obj.type.name}): {', '.join(features)}")

        # Add predicates
        pred_lines = [str(atom) for atom in sorted(atoms)]

        # Combine
        return "\n".join(state_lines + [""] + pred_lines)

    def _goal_atoms_to_text(self, goal: Set[GroundAtom]) -> str:
        """Convert goal atoms to text description."""
        return "\n".join(str(atom) for atom in sorted(goal)) 