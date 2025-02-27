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
        # Add history storage
        self._camera_images_history = []
        self._action_history = []

        assert "OPENAI_API_KEY" in os.environ, "OpenAI API key not found in environment variables"

    @classmethod
    def get_name(cls) -> str:
        return "vlm_perceiver"

    def reset(self, env_task: EnvironmentTask) -> Task:
        """Reset the perceiver with a new environment task."""
        # Reset history
        self._camera_images_history = []
        self._action_history = []
        self._prev_action = None
        
        # Get initial observation and initialize history with first images
        init_obs = env_task.init_obs
        assert isinstance(init_obs, _MockSpotObservation)
        
        # Initialize image history with first observation if enabled
        if CFG.vlm_enable_image_history and init_obs.images is not None:
            self._camera_images_history = [init_obs.images]
            
        state = self._observation_to_state(init_obs)
        # For now, just pass through the goal atoms
        # In future we could convert these to text as well
        return Task(state, env_task.goal)

    def step(self, observation: Observation) -> State:
        """Process a new observation into a state."""
        assert isinstance(observation, _MockSpotObservation)
        
        # Update action history
        if self._prev_action is not None:
            self._action_history.append(self._prev_action)
            if len(self._action_history) > CFG.vlm_max_history_steps:
                self._action_history = self._action_history[-CFG.vlm_max_history_steps:]
                
        # Update camera images history if enabled
        if CFG.vlm_enable_image_history and observation.images is not None:
            self._camera_images_history.append(observation.images)
            if len(self._camera_images_history) > CFG.vlm_max_history_steps:
                self._camera_images_history = self._camera_images_history[-CFG.vlm_max_history_steps:]
        
        return self._observation_to_state(observation)

    def update_perceiver_with_action(self, action: Action) -> None:
        """Update the perceiver with the latest action."""
        self._prev_action = action

    def _observation_to_state(self, obs: _MockSpotObservation) -> State:
        """Convert an observation into a state with text description."""
      
        # Get text description from VLM
        vlm_images = []
        # Add images from history
        for images in self._camera_images_history:
            vlm_images += [PIL.Image.fromarray(img_arr.rgb) for img_arr in images.values()]
                    
        # Add current images
        vlm_images += [PIL.Image.fromarray(img_arr.rgb) for img_arr in obs.images.values()]
        
        text_description = self._get_text_description(vlm_images)
            
        return State({o: np.zeros(o.type.dim) + 0.5 for o in obs.object_dict.values()}, 
                     text_description=text_description,
                     simulator_state=None,
                     camera_images=obs.images,
                     camera_images_history=self._camera_images_history,
                     action_history=self._action_history)

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