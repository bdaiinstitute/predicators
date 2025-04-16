"""Standalone version of Gemini pointing capability with optional SAM2 segmentation."""

import os
import base64
import io
from typing import Dict, List, Optional, Tuple, Union, cast
from pathlib import Path

import ray
import torch
from PIL import Image, ImageDraw
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.traceback import install

from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

from predicators.spot_utils.perception.server.sam2_actor import SAM2Actor
from predicators.spot_utils.perception.utils.gemini_parsing import (
    parse_gemini_point_response,
    parse_gemini_detection_response,
    denormalize_point,
    denormalize_box,
)
from predicators.spot_utils.perception.utils.google_utils import GeminiClient, GEMINI_MODEL_NAME
from predicators.utils import Timer

# Initialize console globally
console = Console()
install(show_locals=True)


def encode_image(image: Union[str, Image.Image]) -> str:
    """Encode image to base64 string."""
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    elif isinstance(image, Image.Image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    else:
        raise ValueError("Image must be either a file path or PIL Image")


def decode_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes))


def draw_points(
    image: Image.Image,
    points_data: Union[Dict[str, List[Dict[str, Union[List[float], str]]]], List[Dict[str, Union[List[float], str]]]],
    point_radius: int = 10,
    text_offset: int = 15,
) -> Image.Image:
    """Draw points and labels on the image with enhanced visibility."""
    draw = ImageDraw.Draw(image)
    
    # Handle both dictionary and list inputs
    if isinstance(points_data, dict):
        points_list = points_data["points"]
    else:
        points_list = points_data
    
    # Get image dimensions for denormalization
    width, height = image.size
    
    for point_data in points_list:
        point = point_data["point"]
        label = str(point_data["label"])
        
        # Denormalize coordinates from 0-1000 range to image coordinates
        y = int(float(point[0]) * height / 1000)
        x = int(float(point[1]) * width / 1000)
        
        # Draw point with larger radius and white outline
        draw.ellipse(
            [
                x - point_radius,
                y - point_radius,
                x + point_radius,
                y + point_radius,
            ],
            fill="red",
            outline="white",
            width=2,
        )
        
        # Draw label with stroke for better visibility
        draw.text(
            (x + text_offset, y - text_offset),
            label,
            fill="red",
            stroke_width=2,
            stroke_fill="white",
        )
    
    return image


def draw_boxes(
    image: Image.Image,
    detection_data: Dict[str, List[Dict[str, Union[List[float], str]]]],
    box_width: int = 2,
    text_offset: int = 10,
) -> Image.Image:
    """Draw bounding boxes and labels on the image."""
    draw = ImageDraw.Draw(image)
    
    for box_data in detection_data["detections"]:
        box = box_data["box_2d"]
        label = str(box_data["label"])
        
        # Convert normalized box to image coordinates
        x1, y1, x2, y2 = box
        x1 = int(float(x1) * image.width / 1000)
        y1 = int(float(y1) * image.height / 1000)
        x2 = int(float(x2) * image.width / 1000)
        y2 = int(float(y2) * image.height / 1000)
        
        # Draw box
        draw.rectangle(
            (x1, y1, x2, y2),  # Convert to tuple
            outline="red",
            width=box_width,
        )
        
        # Draw label
        draw.text(
            (x1 + text_offset, y1 - text_offset),
            label,
            fill="red",
            stroke_width=2,
            stroke_fill="white",
        )
    
    return image


# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI app."""
    global service
    # Initialize service on startup
    service = PointingGeminiSAM2Service()
    yield
    # Cleanup on shutdown
    if service is not None:
        service.cleanup()

app.router.lifespan_context = lifespan

class PointingGeminiSAM2Service:
    """Pointing Gemini SAM2 service."""

    def __init__(
        self,
        use_gpu: bool = True,
        verbose: bool = True,
        save_visualizations: bool = False,
        visualization_dir: str = os.path.join("src", "models", "tmp"),
    ):
        """Initialize the service."""
        self.verbose = verbose
        self.save_visualizations = save_visualizations
        self.visualization_dir = visualization_dir

        if self.verbose:
            console.rule("[bold blue]Device Configuration")
            console.print(f"Using GPU: {use_gpu}", style="cyan")
            console.print(f"CUDA available: {torch.cuda.is_available()}", style="cyan")
            if torch.cuda.is_available():
                console.print(f"CUDA device: {torch.cuda.get_device_name(0)}", style="green")

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            if not torch.cuda.is_available():
                raise RuntimeError("No CUDA GPU available. This service requires a GPU.")
            ray.init(num_gpus=1)

        # Initialize SAM2 actor
        self.sam_actor = SAM2Actor.options(num_gpus=1).remote()  # type: ignore
        
        # Initialize Gemini client
        self.engine = GeminiClient()
        if self.engine is None:
            raise ValueError(f"Failed to initialize engine for model {GEMINI_MODEL_NAME}")
        
        self.model_name = GEMINI_MODEL_NAME

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'sam_actor'):
            ray.kill(self.sam_actor)

    def _format_prompt(self, object_names: Union[str, List[str]], detection: bool = False) -> str:
        """Format the prompt for Gemini with the exact required format."""
        if isinstance(object_names, str):
            object_names = [object_names]
            
        objects_str = ", ".join(object_names)
        
        if detection:
            return (
                f"Detect {objects_str}, with no more than 20 items. "
                "Output a json list where each entry contains the 2D bounding box in \"box_2d\" and a text label in \"label\"."
            )
        else:
            return (
                f"Point to the following items in the image: {objects_str}. "
                "The answer should follow the json format: "
                '[{"point": [y, x], "label": "label1"}, ...]. '
                "The points are in [y, x] format normalized to 0-1000."
            )

    async def predict(
        self,
        images: List[str],
        prompts: List[str],
        points: bool = True,
        segmentation: bool = False,
        detection: bool = False,
    ) -> List[Dict]:
        """Main prediction method for multiple images and prompts."""
        results = []
        # Process each image
        for img_idx, image_b64 in enumerate(images):
            image = decode_image(image_b64)
            width, height = image.size

            # Process each prompt
            for prompt_idx, prompt in enumerate(prompts):
                # Format prompt for Gemini
                formatted_prompt = self._format_prompt(prompt, detection)

                # Get predictions from Gemini
                response = self.engine.generate(
                    model=self.model_name,
                    prompt=formatted_prompt,
                    image=image,
                    temperature=0.0,
                    max_tokens=4096,
                )
                
                # Parse response
                if detection:
                    result = parse_gemini_detection_response(response, (width, height))
                else:
                    result = parse_gemini_point_response(response, (width, height))

                # Get masks from SAM2 if requested
                masks_data = []
                if segmentation:
                    if detection:
                        for box_data in result["detections"]:
                            box = cast(List[float], box_data["box_2d"])
                            # Convert normalized box to image coordinates
                            x1, y1, x2, y2 = denormalize_box(box, (width, height))
                            mask = ray.get(
                                self.sam_actor.predict.remote(  # type: ignore
                                    image, input=[(x1, y1, x2, y2)], type="box"
                                )
                            )
                            masks_data.append(mask)
                    else:
                        for point_data in result["points"]:
                            point = cast(List[float], point_data["point"])
                            # Convert normalized point to image coordinates
                            y, x = denormalize_point(point, (width, height))
                            mask = ray.get(
                                self.sam_actor.predict.remote(  # type: ignore
                                    image, input=[(x, y)], type="point"
                                )
                            )
                            masks_data.append(mask)

                # Format result to match Molmo's format
                formatted_result = {
                    "image_index": img_idx,
                    "prompt_index": prompt_idx,
                    "prompt": prompt,
                    "points": result.get("points", []),
                    "boxes": [
                        [float(x) for x in box["box_2d"]]
                        for box in result.get("detections", [])
                    ] if result.get("detections") else None,
                    "masks": masks_data if segmentation else None,
                    "image_width": width,
                    "image_height": height,
                    "image": image_b64
                }
                results.append(formatted_result)

        return results

@app.post("/pointing_gemini_sam2_service")
async def predict(
    request: Request,
    verbose: bool = Query(True, description="Enable verbose timing and device information"),
):
    """Handle prediction requests."""
    console.rule("[bold blue]New Request")
    console.print(f"[{Timer.get_current_time()}] Received new request", style="cyan")

    with Timer(enable_print=True, name="FastAPI total handler") as t_total:
        # Time JSON parsing
        with Timer(enable_print=True, name="Request JSON parsing") as t:
            try:
                data = await request.json()
                images = data.get("images", [])
                prompts = data.get("prompts", [])
                points = data.get("points", True)
                segmentation = data.get("segmentation", False)
                detection = data.get("detection", False)

                if not images or not prompts:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Missing required fields: images and/or prompts"},
                    )

                console.print(f"Number of images: {len(images)}", style="cyan")
                console.print(f"Number of prompts: {len(prompts)}", style="cyan")
                console.print(f"Total combinations: {len(images) * len(prompts)}", style="cyan")
                console.print(f"Total input size: {sum(len(img) for img in images) / 1024 / 1024:.2f} MB", style="cyan")
            except Exception as e:
                console.print(f"[red]Error parsing request: {str(e)}[/red]")
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid request format: {str(e)}"}
                )

        console.print(f"json parsing time: {t.elapsed_time:.3f}s", style="cyan")
        
        # Time actual prediction
        with Timer(enable_print=True, name="Service prediction") as t:
            try:
                if service is None:
                    return JSONResponse(
                        status_code=500,
                        content={"error": "Service not initialized"}
                    )

                service.verbose = verbose
                result = await service.predict(images, prompts, points, segmentation, detection)
                console.print(f"Prediction completed successfully", style="green")
            except Exception as e:
                console.print(f"[red]Error in prediction: {str(e)}[/red]")
                import traceback
                console.print(traceback.format_exc(), style="red")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Prediction failed: {str(e)}"}
                )

        console.print(f"service prediction time: {t.elapsed_time:.3f}s", style="cyan")
        
        # Time response serialization
        with Timer(enable_print=True, name="Response serialization") as t:
            try:
                console.print(f"Response size: {len(str(result)) / 1024 / 1024:.2f} MB", style="cyan")
                return JSONResponse(content=result)
            except Exception as e:
                console.print(f"[red]Error serializing response: {str(e)}[/red]")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Failed to serialize response: {str(e)}"}
                )

        console.print(f"response serialization time: {t.elapsed_time:.3f}s", style="cyan")
    
    console.print(f"[{Timer.get_current_time()}] Request completed", style="cyan")
    console.print("=" * 50 + "\n", style="cyan")

def start_server(
    host: str = "0.0.0.0",
    port: int = 7100,
    use_gpu: bool = True,
    save_visualizations: bool = False,
):
    """Start the FastAPI server."""
    # Create visualization directory if needed
    if save_visualizations:
        os.makedirs("src/models/tmp", exist_ok=True)
        print("Saving visualizations to: src/models/tmp")

    # Start server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )

if __name__ == "__main__":
    import typer

    def cli(
        host: str = typer.Option("0.0.0.0", help="Host to bind to"),
        port: int = typer.Option(7100, help="Port to bind to"),
        use_gpu: bool = typer.Option(True, help="Whether to use GPU"),
        save_visualizations: bool = typer.Option(False, help="Whether to save visualizations"),
    ):
        """Start the PointingGeminiSAM2 service"""
        start_server(host, port, use_gpu, save_visualizations)

    typer.run(cli) 