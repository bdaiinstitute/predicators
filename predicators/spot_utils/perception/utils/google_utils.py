import os
from typing import Any, List, Optional, Union

import google.generativeai as genai
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_random_exponential
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.types import GenerationConfig


# GEMINI_MODEL_NAME = "gemini-2.5.pro-exp-03-25"
# GEMINI_MODEL_NAME = "gemini-2.5.pro"
GEMINI_MODEL_NAME = "gemini-2.0-flash"


class GeminiClient:
    """Client for interacting with Gemini models."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini client.

        Args:
            api_key: Optional API key. If not provided, will use GEMINI_API_KEY env var.
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable must be set or api_key must be provided"
            )
        genai.configure(api_key=api_key)

    def generate(
        self,
        model: str = GEMINI_MODEL_NAME,
        prompt: str = "",
        image: Optional[Union[str, Image.Image]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any
    ) -> str:
        """Generate a response from Gemini.

        Args:
            model: Name of the Gemini model to use
            prompt: Text prompt
            image: Optional image (path or PIL Image)
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments for the model

        Returns:
            Generated text response
        """
        gen_model = genai.GenerativeModel(model)
        
        if image is None:
            response = gen_model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    **kwargs
                ),
            )
        else:
            if isinstance(image, str):
                image = Image.open(image)
            response = gen_model.generate_content(
                [prompt, image],
                generation_config=GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    **kwargs
                ),
            )
        
        return response.text

    def generate_stream(
        self,
        model: str = GEMINI_MODEL_NAME,
        prompt: str = "",
        image: Optional[Union[str, Image.Image]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any
    ) -> List[str]:
        """Generate a streaming response from Gemini.

        Args:
            model: Name of the Gemini model to use
            prompt: Text prompt
            image: Optional image (path or PIL Image)
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments for the model

        Returns:
            List of text chunks
        """
        gen_model = genai.GenerativeModel(model)
        
        if image is None:
            response = gen_model.generate_content(
                prompt,
                stream=True,
                generation_config=GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    **kwargs
                ),
            )
        else:
            if isinstance(image, str):
                image = Image.open(image)
            response = gen_model.generate_content(
                [prompt, image],
                stream=True,
                generation_config=GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    **kwargs
                ),
            )
        
        return [chunk.text for chunk in response]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_google_api(
    message: Union[str, List[Union[Any, Image.Image]]],
    model: str = "gemini-pro",  # gemini-pro, gemini-pro-vision
) -> str:
    try:
        gen_model = genai.GenerativeModel(model)
        response = gen_model.generate_content(message)
        response.resolve()
        return response.text
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        raise e


if __name__ == "__main__":
    # Example usage
    client = GeminiClient()
    input = "What color are apples?"
    print("input: {}".format(input))
    output = call_google_api(input)
    print("output: {}".format(output))
