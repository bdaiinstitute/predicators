"""Tests for the VLM perceiver."""

import os
from typing import List, Optional
from unittest.mock import patch, MagicMock

import PIL.Image
import pytest

from predicators import utils
from predicators.perception.vlm_perceiver import VLMPerceiver
from predicators.pretrained_model_interface import OpenAIVLM, VisionLanguageModel
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, State, Task


def test_vlm_perceiver():
    """Tests for VLMPerceiver()."""
    # Create a dummy image for testing
    dummy_img = PIL.Image.new('RGB', (100, 100))
    
    # Create a test observation with the dummy image
    obs = {"images": {"main": dummy_img}}
    
    # Create a dummy environment task with empty goal
    env_task = EnvironmentTask(obs, set())

    # Create a mock OpenAI VLM that returns a fixed response
    mock_vlm = MagicMock(spec=OpenAIVLM)
    mock_vlm.sample_completions.return_value = ["The scene contains a red cup on a white table. The cup is empty and positioned near the center of the table."]
    mock_vlm.get_id.return_value = "mock-gpt4o"
    
    # Mock the VLM creation and environment variable
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'dummy_key'}):
        with patch('predicators.pretrained_model_interface.create_vlm_by_name', return_value=mock_vlm):
            # Create the perceiver
            perceiver = VLMPerceiver()
            assert perceiver.get_name() == "vlm"
            
            # Test reset
            task = perceiver.reset(env_task)
            assert isinstance(task, Task)
            assert isinstance(task.init, State)
            assert task.init.text_description == "The scene contains a red cup on a white table. The cup is empty and positioned near the center of the table."
            assert not task.goal  # empty set
            
            # Test step with image
            state = perceiver.step(obs)
            assert isinstance(state, State)
            assert state.text_description == "The scene contains a red cup on a white table. The cup is empty and positioned near the center of the table."
            
            # Test step with empty observation
            empty_obs = {}
            empty_state = perceiver.step(empty_obs)
            assert isinstance(empty_state, State)
            assert empty_state.text_description is None
            
            # Test render_mental_images raises NotImplementedError
            with pytest.raises(NotImplementedError):
                perceiver.render_mental_images(obs, env_task)

            # Test that the perceiver works with multiple images
            multi_img_obs = {"images": {"main": dummy_img, "gripper": dummy_img}}
            multi_img_state = perceiver.step(multi_img_obs)
            assert isinstance(multi_img_state, State)
            assert multi_img_state.text_description == "The scene contains a red cup on a white table. The cup is empty and positioned near the center of the table."

            # Verify that the VLM was called with the correct arguments
            mock_vlm.sample_completions.assert_called_with(
                prompt=CFG.vlm_text_perceiver_prompt,
                imgs=list(multi_img_obs["images"].values()),
                temperature=CFG.vlm_temperature,
                seed=0,
                num_completions=1
            ) 