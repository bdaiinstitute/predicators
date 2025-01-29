"""A VLM-based perceiver that generates text descriptions of states."""

from typing import Dict, List, Optional

import numpy as np
import os

from predicators import utils
from predicators.perception.base_perceiver import BasePerceiver
from predicators.pretrained_model_interface import create_vlm_by_name
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, \
    Observation, State, Task, Video, Action
import PIL
from predicators.envs.mock_spot_env import _MockSpotObservation

class VLMPerceiver(BasePerceiver):
    """A perceiver that uses a Vision Language Model to generate text descriptions of states."""

    def __init__(self) -> None:
        super().__init__()
        # Initialize the OpenAI VLM
        self._vlm = create_vlm_by_name(CFG.vlm_model_name)
        self._prev_action: Optional[Action] = None  # Track previous action

        assert "OPENAI_API_KEY" in os.environ, "OpenAI API key not found in environment variables"

    @classmethod
    def get_name(cls) -> str:
        return "vlm_perceiver"

    def reset(self, env_task: EnvironmentTask) -> Task:
        """Reset the perceiver with a new environment task."""
        state = self._observation_to_state(env_task.init_obs)
        self._prev_action = None
        # For now, just pass through the goal atoms
        # In future we could convert these to text as well
        return Task(state, env_task.goal)

    def step(self, observation: Observation) -> State:
        """Process a new observation into a state."""
        return self._observation_to_state(observation)

    def update_perceiver_with_action(self, action: Action) -> None:
        """Update the perceiver with the latest action."""
        self._prev_action = action

    def _observation_to_state(self, obs: _MockSpotObservation) -> State:
        """Convert an observation into a state with text description."""
      

        # Get text description from VLM
        vlm_images = []
        if obs.camera_images_history is not None:
            camera_image_history = obs.camera_images_history
            for images in obs.camera_images_history:
                vlm_images += [PIL.Image.fromarray(img_arr.rgb) for img_arr in images.values()]
        else:
            camera_image_history = []
                    
        vlm_images += [PIL.Image.fromarray(img_arr.rgb) for img_arr in obs.images.values()]
        
        text_description = self._get_text_description(vlm_images)
        
        action_history = []
        if hasattr(obs, 'action_history') and obs.action_history is not None:
            action_history = obs.action_history.copy()
            
        if self._prev_action is not None:
            action_history.append(self._prev_action)
        
        if len(action_history) > CFG.vlm_max_history_steps:
            action_history = action_history[-CFG.vlm_max_history_steps:]
            
        # Create state with text description
        # For now, we'll keep an empty data dict since we're focusing on text
        return State({o: np.zeros(o.type.dim) + 0.5 for o in obs.object_dict.values()}, 
                     text_description=text_description,
                     simulator_state=None,
                     camera_images=obs.images,
                     camera_images_history=camera_image_history+[obs.images],
                     action_history=action_history)

    def _get_text_description(self, vlm_images: List) -> str:
        """Use VLM to generate a text description of the images."""
        # Format prompt for VLM
        prompt = CFG.vlm_text_perceiver_prompt

        # Query VLM with images  
        
        print("Num vlm images in history: "+str(len(vlm_images)))      
        completions = self._vlm.sample_completions(
            prompt=prompt,
            imgs=vlm_images,
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