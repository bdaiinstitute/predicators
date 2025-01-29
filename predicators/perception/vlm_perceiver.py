"""A VLM-based perceiver that generates text descriptions of states."""

from typing import Dict, Optional, Set

import numpy as np
import os

from predicators import utils
from predicators.perception.base_perceiver import BasePerceiver
from predicators.pretrained_model_interface import create_vlm_by_name
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, GroundAtom, Object, \
    Observation, State, Task, Video


class VLMPerceiver(BasePerceiver):
    """A perceiver that uses a Vision Language Model to generate text descriptions of states."""

    def __init__(self) -> None:
        super().__init__()
        # Initialize the OpenAI VLM
        self._vlm = create_vlm_by_name(CFG.vlm_model_name)
        assert "OPENAI_API_KEY" in os.environ, "OpenAI API key not found in environment variables"

    @classmethod
    def get_name(cls) -> str:
        return "vlm"

    def reset(self, env_task: EnvironmentTask) -> Task:
        """Reset the perceiver with a new environment task."""
        state = self._observation_to_state(env_task.init_obs)
        # For now, just pass through the goal atoms
        # In future we could convert these to text as well
        return Task(state, env_task.goal)

    def step(self, observation: Observation) -> State:
        """Process a new observation into a state."""
        return self._observation_to_state(observation)

    def _observation_to_state(self, obs: Observation) -> State:
        """Convert an observation into a state with text description."""
        # Extract images from observation
        images = obs.get("images", {})
        if not images:
            # If no images, create an empty state with no description
            return State({}, text_description=None)

        # Get text description from VLM
        text_description = self._get_text_description(images)

        # Create state with text description
        # For now, we'll keep an empty data dict since we're focusing on text
        return State({}, text_description=text_description)

    def _get_text_description(self, images: Dict) -> str:
        """Use VLM to generate a text description of the images."""
        # Format prompt for VLM
        prompt = CFG.vlm_text_perceiver_prompt
        
        # Query VLM with images
        completions = self._vlm.sample_completions(
            prompt=prompt,
            imgs=list(images.values()),
            temperature=CFG.vlm_temperature,
            seed=0,  # Fixed seed for deterministic behavior
            num_completions=1
        )
        
        # Return the first (and only) completion
        return completions[0]

    def render_mental_images(self, observation: Observation,
                           env_task: EnvironmentTask) -> Video:
        """Not implemented for VLM perceiver."""
        raise NotImplementedError("Mental images not implemented for VLM perceiver") 