"""Interface to pretrained large models.

These might be joint Vision-Language Models (VLM's) or Large Language
Models (LLM's)
"""
from __future__ import annotations

import abc
import base64
import logging
import os
from io import BytesIO
from typing import Collection, Dict, List, Optional, Union
from collections import defaultdict

import google.generativeai as genai
import imagehash
import openai
import PIL.Image
from tenacity import retry, stop_after_attempt, wait_random_exponential

from predicators.settings import CFG
from predicators.utils import timing

# This is a special string that we assume will never appear in a prompt, and
# which we use to separate prompt and completion in the cache. The reason to
# do it this way, rather than saving the prompt and responses separately,
# is that we want it to be easy to browse the cache as text files.
_CACHE_SEP = "\n####$$$###$$$####$$$$###$$$####$$$###$$$###\n"


class PretrainedLargeModel(abc.ABC):
    """A pretrained large vision or language model."""

    @abc.abstractmethod
    def get_id(self) -> str:
        """Get a string identifier for this model.

        This identifier should include sufficient information so that
        querying the same model with the same prompt and same identifier
        should yield the same result (assuming temperature 0).
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _sample_completions(self,
                            prompt: str,
                            imgs: Optional[List[PIL.Image.Image]],
                            temperature: float,
                            seed: int,
                            stop_token: Optional[str] = None,
                            num_completions: int = 1) -> List[str]:
        """This is the main method that subclasses must implement.

        This helper method is called by sample_completions(), which
        caches the prompts and responses to disk.
        """
        raise NotImplementedError("Override me!")

    def sample_completions(self,
                           prompt: str,
                           imgs: Optional[List[PIL.Image.Image]],
                           temperature: float,
                           seed: int,
                           stop_token: Optional[str] = None,
                           num_completions: int = 1) -> List[str]:
        """Sample one or more completions from a prompt.

        Higher temperatures will increase the variance in the responses.
        The seed may not be used and the results may therefore not be
        reproducible for models where we only have access through an API
        that does not expose the ability to set a random seed. Responses
        are saved to disk.
        """
        # Set up the cache file.
        assert _CACHE_SEP not in prompt
        os.makedirs(CFG.pretrained_model_prompt_cache_dir, exist_ok=True)
        model_id = self.get_id()
        prompt_id = hash(prompt)
        config_id = f"{temperature}_{seed}_{num_completions}_" + \
                    f"{stop_token}"
        # If the temperature is 0, the seed does not matter.
        if temperature == 0.0:
            config_id = f"most_likely_{num_completions}_{stop_token}"
        cache_foldername = f"{model_id}_{config_id}_{prompt_id}"
        if imgs is not None:
            # We also need to hash all the images in the prompt.
            img_hash_list: List[str] = []
            for img in imgs:
                img_hash_list.append(str(imagehash.phash(img)))
            # NOTE: it's possible that this string (the concatenated hashes of
            # each image) is very long. This would make the final cache
            # foldername long. In many operating systems, the maximum folder
            # name length is 255 characters. To shorten this foldername more, we
            # can hash this string into a shorter string. For example, look at
            # https://stackoverflow.com/questions/57263436/hash-like-string-shortener-with-decoder  # pylint:disable=line-too-long
            imgs_id = hash("".join(img_hash_list))
            cache_foldername += f"{imgs_id}"
        cache_folderpath = os.path.join(CFG.pretrained_model_prompt_cache_dir,
                                        cache_foldername)
        os.makedirs(cache_folderpath, exist_ok=True)
        cache_filename = "prompt.txt"
        cache_filepath = os.path.join(CFG.pretrained_model_prompt_cache_dir,
                                      cache_foldername, cache_filename)
        if not os.path.exists(cache_filepath):
            if CFG.llm_use_cache_only:
                raise ValueError("No cached response found for prompt.")
            logging.debug(f"Querying model {model_id} with new prompt.")
            # Query the model.
            with timing("[Model API Query]"):
                completions = self._sample_completions(prompt, imgs, temperature,
                                                       seed, stop_token,
                                                       num_completions)
            # Cache the completion.
            cache_str = prompt + _CACHE_SEP + _CACHE_SEP.join(completions)
            with open(cache_filepath, 'w', encoding='utf-8') as f:
                f.write(cache_str)
            if imgs is not None:
                # Also save the images for easy debugging.
                imgs_folderpath = os.path.join(cache_folderpath, "imgs")
                os.makedirs(imgs_folderpath, exist_ok=True)
                for i, img in enumerate(imgs):
                    filename_suffix = str(i) + ".jpg"
                    # Convert to RGB mode before saving to avoid potential mode issues
                    img_rgb = img.convert('RGB')
                    img_rgb.save(os.path.join(imgs_folderpath, filename_suffix), 'JPEG')
                logging.debug(f"Saved model response to {cache_filepath}.")
        # Load the saved completion.
        with open(cache_filepath, 'r', encoding='utf-8') as f:
            cache_str = f.read()
        logging.debug(f"Loaded model response from {cache_filepath}.")
        assert cache_str.count(_CACHE_SEP) == num_completions
        cached_prompt, completion_strs = cache_str.split(_CACHE_SEP, 1)
        assert cached_prompt == prompt
        completions = completion_strs.split(_CACHE_SEP)
        return completions


class VisionLanguageModel(PretrainedLargeModel):
    """A class for all VLM's."""

    def sample_completions(
            self,
            prompt: str,
            imgs: Optional[List[PIL.Image.Image]],
            temperature: float,
            seed: int,
            stop_token: Optional[str] = None,
            num_completions: int = 1) -> List[str]:  # pragma: no cover
        assert imgs is not None
        return super().sample_completions(prompt, imgs, temperature, seed,
                                          stop_token, num_completions)


class LargeLanguageModel(PretrainedLargeModel):
    """A class for all LLM's."""

    def sample_completions(
            self,
            prompt: str,
            imgs: Optional[List[PIL.Image.Image]],
            temperature: float,
            seed: int,
            stop_token: Optional[str] = None,
            num_completions: int = 1) -> List[str]:  # pragma: no cover
        assert imgs is None
        return super().sample_completions(prompt, imgs, temperature, seed,
                                          stop_token, num_completions)


class OpenAIModel():
    """Common interface with methods for all OpenAI-based models."""

    # Class variables to track costs and usage across all instances
    _total_costs = defaultdict(float)
    _total_tokens = defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0})
    
    # [2025/01] Latest pricing per 1M tokens from https://platform.openai.com/docs/pricing
    _COSTS = {
        # GPT-4 Optimized (gpt-4o)
        "gpt-4o": (2.50/1000000, 1.25/1000000),  # $2.50/1M input, $1.25/1M output
        "gpt-4o-2024-08-06": (2.50/1000000, 1.25/1000000),
        
        # GPT-4 Optimized Mini
        "gpt-4o-mini": (0.15/1000000, 0.075/1000000),  # $0.15/1M input, $0.075/1M output
        "gpt-4o-mini-2024-07-18": (0.15/1000000, 0.075/1000000),
        
        # O1 Models
        "o1": (15.00/1000000, 7.50/1000000),  # $15.00/1M input, $7.50/1M output
        "o1-2024-12-17": (15.00/1000000, 7.50/1000000),
        
        # O1 Mini
        "o1-mini": (3.00/1000000, 1.50/1000000),  # $3.00/1M input, $1.50/1M output
        "o1-mini-2024-09-12": (3.00/1000000, 1.50/1000000),
    }

    def set_openai_key(self, key: Optional[str] = None) -> None:
        """Set the OpenAI API key."""
        if key is None:
            assert "OPENAI_API_KEY" in os.environ
            key = os.environ["OPENAI_API_KEY"]

    @classmethod
    def log_total_costs(cls) -> None:
        """Log the total costs and token usage for all OpenAI API calls."""
        if cls._total_costs:
            logging.info("\n=== OpenAI API Usage (2025/01 Pricing) ===")
            total = 0.0
            for model, cost in sorted(cls._total_costs.items()):
                tokens = cls._total_tokens[model]
                logging.info(f"{model}:")
                logging.info(f"  Cost: ${cost:.4f}")
                logging.info(f"  Tokens - Input: {tokens['prompt']}, Output: {tokens['completion']}, Total: {tokens['total']}")
                total += cost
            logging.info(f"Total Cost: ${total:.4f}")

    @retry(wait=wait_random_exponential(min=1, max=60),
           stop=stop_after_attempt(10))
    def call_openai_api(self,
                        messages: list,
                        model: str = "gpt-4",
                        seed: Optional[int] = None,
                        max_tokens: int = 32,
                        temperature: float = 0.2,
                        verbose: bool = False) -> str:  # pragma: no cover
        """Make an API call to OpenAI."""
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            seed=seed,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        if verbose:
            logging.debug(f"OpenAI API response: {completion}")


        # Track costs and tokens
        if model in self._COSTS and hasattr(completion, 'usage') and completion.usage is not None:
            input_cost, output_cost = self._COSTS[model]
            prompt_tokens = getattr(completion.usage, 'prompt_tokens', 0)
            completion_tokens = getattr(completion.usage, 'completion_tokens', 0)
            total_tokens = getattr(completion.usage, 'total_tokens', 0)
            
            total_cost = (prompt_tokens * input_cost) + (completion_tokens * output_cost)
            self._total_costs[model] += total_cost
            
            # Update token counts
            self._total_tokens[model]["prompt"] += prompt_tokens
            self._total_tokens[model]["completion"] += completion_tokens
            self._total_tokens[model]["total"] += total_tokens
            
            # Simple running cost display
            total_cost = sum(self._total_costs.values())
            logging.info(f"[OpenAI Models Running Cost in Entire Run: ${total_cost:.4f}]")

        assert len(completion.choices) == 1
        assert completion.choices[0].message.content is not None
        return completion.choices[0].message.content


class GoogleGeminiModel():
    """Common interface and methods for all Gemini-based models.

    Assumes that an environment variable GOOGLE_API_KEY is set with the
    necessary API key to query the particular model name.
    """

    def __init__(self, model_name: str) -> None:
        """See https://ai.google.dev/models/gemini for the list of available
        model names."""
        self._model_name = model_name
        assert "GOOGLE_API_KEY" in os.environ
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self._model = genai.GenerativeModel(self._model_name)  # pylint:disable=no-member


class OpenAILLM(LargeLanguageModel, OpenAIModel):
    """Interface to openAI LLMs.

    Assumes that an environment variable OPENAI_API_KEY is set to a
    private API key for beta.openai.com.
    """

    def __init__(self, model_name: str) -> None:
        """See https://platform.openai.com/docs/models for the list of
        available model names."""
        self._model_name = model_name
        # Note that max_tokens is the maximum response length (not prompt).
        # From OpenAI docs: "The token count of your prompt plus max_tokens
        # cannot exceed the model's context length."
        self._max_tokens = CFG.llm_openai_max_response_tokens
        self.set_openai_key()

    def get_id(self) -> str:
        return f"openai-{self._model_name}"

    def _sample_completions(
            self,
            prompt: str,
            imgs: Optional[List[PIL.Image.Image]],
            temperature: float,
            seed: int,
            stop_token: Optional[str] = None,
            num_completions: int = 1) -> List[str]:  # pragma: no cover
        del imgs, seed, stop_token  # unused
        messages = [{"role": "user", "content": prompt, "type": "text"}]
        responses = [
            self.call_openai_api(messages,
                                 model=self._model_name,
                                 temperature=temperature,
                                 max_tokens=self._max_tokens)
            for _ in range(num_completions)
        ]
        return responses


class GoogleGeminiLLM(LargeLanguageModel, GoogleGeminiModel):
    """Interface to the Google Gemini VLM (1.5).

    Assumes that an environment variable GOOGLE_API_KEY is set with the
    necessary API key to query the particular model name.
    """

    @retry(wait=wait_random_exponential(min=1, max=60),
           stop=stop_after_attempt(10))
    def _sample_completions(
            self,
            prompt: str,
            imgs: Optional[List[PIL.Image.Image]],
            temperature: float,
            seed: int,
            stop_token: Optional[str] = None,
            num_completions: int = 1) -> List[str]:  # pragma: no cover
        del seed, stop_token  # unused
        assert imgs is None
        generation_config = genai.types.GenerationConfig(  # pylint:disable=no-member
            candidate_count=num_completions,
            temperature=temperature)
        response = self._model.generate_content(
            [prompt], generation_config=generation_config)  # type: ignore
        response.resolve()  # type: ignore
        return [response.text]

    def get_id(self) -> str:
        return f"Google-{self._model_name}"


class GoogleGeminiVLM(VisionLanguageModel, GoogleGeminiModel):
    """Interface to the Google Gemini VLM (1.5).

    Assumes that an environment variable GOOGLE_API_KEY is set with the
    necessary API key to query the particular model name.
    """

    @retry(wait=wait_random_exponential(min=1, max=60),
           stop=stop_after_attempt(20))
    def _sample_completions(
            self,
            prompt: str,
            imgs: Optional[List[PIL.Image.Image]],
            temperature: float,
            seed: int,
            stop_token: Optional[str] = None,
            num_completions: int = 1) -> List[str]:  # pragma: no cover
        del seed, stop_token  # unused
        assert imgs is not None
        generation_config = genai.types.GenerationConfig(  # pylint:disable=no-member
            candidate_count=num_completions,
            temperature=temperature)
        response = self._model.generate_content(
            [prompt] + imgs,  # type: ignore
            generation_config=generation_config)  # type: ignore
        response.resolve()  # type: ignore
        return [response.text]

    def get_id(self) -> str:
        return f"Google-{self._model_name}"


class OpenAIVLM(VisionLanguageModel, OpenAIModel):
    """Interface for OpenAI's VLMs, including GPT-4 Turbo (and preview
    versions)."""

    def __init__(self, model_name: str):
        """Initialize with a specific model name."""
        self.model_name = model_name
        # Note that max_tokens is the maximum response length (not prompt).
        # From OpenAI docs: "The token count of your prompt plus max_tokens
        # cannot exceed the model's context length."
        self._max_tokens = CFG.llm_openai_max_response_tokens
        self.set_openai_key()

    def prepare_vision_messages(
        self,
        images: List[PIL.Image.Image],
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        image_size: Optional[int] = 512,
        detail: str = "auto"
    ) -> List[Dict[str, Union[str, List[Dict[str, str]], List[Dict[
            str, Collection[str]]]]]]:
        """Prepare text and image messages for the OpenAI API."""
        content: List[Dict[str, Union[str, Collection[str]]]] = []
        if prefix:
            content.append({"text": prefix, "type": "text"})
        assert images
        assert detail in ["auto", "low", "high"]
        for img in images:
            img_resized = img
            if image_size:
                factor = image_size / max(img.size)
                img_resized = img.resize(
                    (int(img.size[0] * factor), int(img.size[1] * factor)))
            # Convert the image to PNG format and encode it in base64
            buffer = BytesIO()
            img_resized.save(buffer, format="PNG")
            buf = buffer.getvalue()
            frame = base64.b64encode(buf).decode("utf-8")
            content_str = {
                "image_url": {
                    "url": f"data:image/png;base64,{frame}",
                    "detail": detail
                },
                "type": "image_url"
            }
            content.append(content_str)
        if suffix:
            content.append({"text": suffix, "type": "text"})
        return [{"role": "user", "content": content}]

    def get_id(self) -> str:
        """Get an identifier for the model."""
        return f"OpenAI-{self.model_name}"

    def _sample_completions(
        self,
        prompt: str,
        imgs: Optional[List[PIL.Image.Image]],
        temperature: float,
        seed: int,
        stop_token: Optional[str] = None,
        num_completions: int = 1,
    ) -> List[str]:  # pragma: no cover
        """Query the model and get responses."""
        del seed, stop_token  # unused.
        if imgs is None:
            raise ValueError("images cannot be None")
        messages = self.prepare_vision_messages(prefix=prompt,
                                                images=imgs,
                                                detail="auto")
        responses = [
            self.call_openai_api(messages,
                                 model=self.model_name,
                                 max_tokens=self._max_tokens,
                                 temperature=temperature)
            for _ in range(num_completions)
        ]
        return responses


def create_llm_by_name(
        model_name: str) -> LargeLanguageModel:  # pragma: no cover
    """Create particular llm using a provided name."""
    if "gemini" in model_name:
        return GoogleGeminiLLM(model_name)
    return OpenAILLM(model_name)


def create_vlm_by_name(
        model_name: str) -> VisionLanguageModel:  # pragma: no cover
    """Create particular vlm using a provided name."""
    if "gemini" in model_name:
        return GoogleGeminiVLM(model_name)
    return OpenAIVLM(model_name)
