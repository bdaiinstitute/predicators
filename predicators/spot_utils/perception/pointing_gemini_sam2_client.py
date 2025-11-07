"""Client for Gemini pointing capability with optional SAM2 segmentation."""

import asyncio
import base64
import io
import os
import time
from typing import Dict, List, Union, Sequence

import httpx
from PIL import Image, ImageDraw
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer

# Initialize console globally
console = Console()


class Timer:
    """Simple timer context manager."""
    def __init__(self, enable_print: bool = True):
        self.enable_print = enable_print
        self.start_time = 0.0
        self.elapsed_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_time = time.time() - self.start_time
        if self.enable_print:
            print(f"Elapsed time: {self.elapsed_time:.3f}s")


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


def draw_points(
    image: Image.Image,
    points_data: List[Dict[str, Union[List[float], str]]],
    point_radius: int = 10,
    text_offset: int = 15,
) -> Image.Image:
    """Draw points and labels on the image with enhanced visibility."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    for point_data in points_data:
        point = point_data["point"]
        label = str(point_data["label"])
        
        # Denormalize coordinates from 0-1000 range to image coordinates
        y = int(float(point[0]) * height / 1000)
        x = int(float(point[1]) * width / 1000)
        
        # Draw point with larger radius and white outline
        draw.ellipse(
            [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
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


class PointingGeminiSAM2Client:
    def __init__(self, host: str = "localhost", port: int = 7100):
        """Initialize client with host and port."""
        self.endpoint = f"http://{host}:{port}/pointing_gemini_sam2_service"

    async def predict_async(
        self,
        images: Union[str, Image.Image, Sequence[Union[str, Image.Image]]],
        prompts: Union[str, Sequence[str]],
        points: bool = True,
        segmentation: bool = False,
        detection: bool = False,
    ) -> Dict:
        """Async prediction with support for multiple images and prompts."""
        timings = {}
        console.rule("[bold blue]Starting PointingGeminiSAM2 Client Request")
        console.print(f"Endpoint: {self.endpoint}", style="cyan")

        # Convert single inputs to lists
        if isinstance(images, (str, Image.Image)):
            images = [images]
        if isinstance(prompts, str):
            prompts = [prompts]

        # Process images
        processed_images = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Processing images...", total=len(images))
            
            with Timer(enable_print=False) as t:
                for i, img in enumerate(images):
                    if isinstance(img, str):
                        progress.update(task, description=f"Loading image {i+1}: {os.path.basename(img)}")
                        img_pil = Image.open(img)
                    else:
                        progress.update(task, description=f"Processing image {i+1}")
                        img_pil = img
                    
                    img_b64 = encode_image(img_pil)
                    processed_images.append(img_b64)
                    progress.advance(task)
            timings["preprocessing"] = t.elapsed_time

        # Make server request
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Waiting for server response...")
            
            with Timer(enable_print=False) as t:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.endpoint,
                        json={
                            "images": processed_images,
                            "prompts": prompts,
                            "points": points,
                            "segmentation": segmentation,
                            "detection": detection,
                        },
                        timeout=120.0
                    )

                    progress.update(task, completed=True)

                    if response.status_code != 200:
                        raise RuntimeError(
                            f"Pointing service returned status {response.status_code}:\n"
                            f"{response.text}"
                        )

                    result = response.json()
                    entries = []
                    count = "unknown"
                    example = None
                    if isinstance(result, dict):
                        entries = result.get("results") or []
                        count = len(entries)
                        example = entries[0].get("prompt") if entries else None
                        if not entries:
                            logging.warning(
                                "[PointingClient] No detections returned (HTTP 200)"
                                "; check server logs for details.")
                    extra = f", sample='{example}'" if example else ""
                    console.print(
                        f"[green]Server request completed successfully[/green]"
                        f" (results={count}{extra})")
            timings["server_request"] = t.elapsed_time

        return {"results": result, "timings": timings}

    def predict(
        self,
        images: Union[str, Image.Image, Sequence[Union[str, Image.Image]]],
        prompts: Union[str, Sequence[str]],
        points: bool = True,
        segmentation: bool = False,
        detection: bool = False,
    ) -> Dict:
        """Synchronous wrapper for predict_async."""
        with Timer(enable_print=False) as t:
            result = asyncio.run(self.predict_async(
                images, prompts, points, segmentation, detection
            ))
        result["timings"]["total"] = t.elapsed_time
        return result


app = typer.Typer()


@app.command()
def predict(
    image: List[str] = typer.Option(
        None,
        "--image", "-i",
        help="Image path(s). Can be specified multiple times for multiple images.",
        callback=lambda x: x or [],
    ),
    prompt: List[str] = typer.Option(
        None,
        "--prompt", "-p",
        help="Text prompt(s). Can be specified multiple times for multiple prompts.",
        callback=lambda x: x or [],
    ),
    host: str = typer.Option("localhost", help="Host of the PointingGeminiSAM2 service"),
    port: int = typer.Option(7100, help="Port of the PointingGeminiSAM2 service"),
    points: bool = typer.Option(True, help="Whether to return point coordinates"),
    segmentation: bool = typer.Option(False, help="Whether to perform segmentation"),
    detection: bool = typer.Option(False, help="Whether to perform detection instead of pointing"),
):
    """Run predictions using the PointingGeminiSAM2 service."""
    try:
        # Clean up inputs
        prompts = [p.strip().strip("\"'").replace('\\"', '"').replace("\\'", "'").replace("\\", "") for p in prompt]
        image_paths = [p.strip().strip("\"'").replace('\\"', '"').replace("\\'", "'").replace("\\", "") for p in image]

        if not image_paths or not prompts:
            raise typer.BadParameter("Must provide at least one image (-i) and one prompt (-p)")

        client = PointingGeminiSAM2Client(host=host, port=port)
        result = client.predict(image_paths, prompts, points=points, segmentation=segmentation, detection=detection)

        # Print results
        console.rule("[bold blue]Results")
        for res in result["results"]:
            img_idx = res["image_index"]
            prompt_idx = res["prompt_index"]
            prompt = res["prompt"]
            
            console.print(f"\n[yellow]Image {img_idx + 1}, Prompt {prompt_idx + 1}: '{prompt}'[/yellow]")
            if res.get("points"):
                console.print(f"Points coordinates: {res['points']}", style="cyan")
            if res.get("detections"):
                console.print("\nBounding boxes:", style="cyan")
                for i, box in enumerate(res["detections"], 1):
                    console.print(f"Box {i}: {box['box_2d']} ({box['label']})", style="cyan")

        # Print timing
        console.rule("[bold blue]Timing")
        console.print(f"Preprocessing: {result['timings']['preprocessing']:.3f}s", style="cyan")
        console.print(f"Server request: {result['timings']['server_request']:.3f}s", style="cyan")
        console.print(f"Total time: {result['timings']['total']:.3f}s", style="green")

    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        console.print("\n[yellow]Make sure to:[/yellow]")
        console.print("1. Start the server first: [cyan]python -m src.models.segmentation.pointing_gemini_sam2[/cyan]")
        console.print("2. Wait a few seconds for the server to initialize")
        console.print("3. Try the request again")


if __name__ == "__main__":
    app()
