"""Client for Gemini pointing capability with optional SAM2 segmentation."""

import asyncio
import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence, Union

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
        norm_y = _clamp(float(point[0]), 0.0, 1000.0)
        norm_x = _clamp(float(point[1]), 0.0, 1000.0)
        y = int(round(norm_y * height / 1000))
        x = int(round(norm_x * width / 1000))
        y = int(_clamp(y, 0, height - 1))
        x = int(_clamp(x, 0, width - 1))
        
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


def draw_detections(
    image: Image.Image,
    detections: List[Dict[str, Union[List[float], str]]],
    box_width: int = 2,
    text_offset: int = 10,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for det in detections:
        box = det.get("box_2d")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        x1 = int(round(_clamp(float(x1), 0, 1000) * width / 1000))
        x2 = int(round(_clamp(float(x2), 0, 1000) * width / 1000))
        y1 = int(round(_clamp(float(y1), 0, 1000) * height / 1000))
        y2 = int(round(_clamp(float(y2), 0, 1000) * height / 1000))
        x1 = int(_clamp(x1, 0, width - 1))
        x2 = int(_clamp(x2, 0, width - 1))
        y1 = int(_clamp(y1, 0, height - 1))
        y2 = int(_clamp(y2, 0, height - 1))
        draw.rectangle((x1, y1, x2, y2), outline="cyan", width=box_width)
        label = str(det.get("label", ""))
        if label:
            draw.text((x1 + text_offset, y1 - text_offset), label, fill="cyan")
    return image


_MASK_COLORS = [
    (255, 0, 0),
    (0, 200, 255),
    (0, 255, 127),
    (255, 165, 0),
    (186, 85, 211),
]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def overlay_masks(
    image: Image.Image,
    masks: List[Dict[str, Union[str, int]]],
    alpha: int = 110,
) -> Image.Image:
    """Blend segmentation masks onto the provided image."""
    if not masks:
        return image
    base = image.convert("RGBA")
    for idx, mask_entry in enumerate(masks):
        mask_b64 = mask_entry.get("mask_png")
        if not mask_b64:
            continue
        try:
            mask = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")
        except Exception as exc:  # pragma: no cover - debug helper
            console.print(f"[red]Failed to decode mask for visualization: {exc}[/red]")
            continue
        if mask.size != base.size:
            mask = mask.resize(base.size, Image.NEAREST)
        color = _MASK_COLORS[idx % len(_MASK_COLORS)]
        overlay = Image.new("RGBA", base.size, color + (alpha,))
        base = Image.composite(overlay, base, mask)
    return base.convert("RGB")


def _save_client_visualizations(entries: Union[List[Dict], Dict], directory: str) -> None:
    if isinstance(entries, dict):
        entries = [entries]
    if not entries:
        console.print("[yellow]No results to visualize.[/yellow]")
        return
    Path(directory).mkdir(parents=True, exist_ok=True)
    for entry in entries:
        image_b64 = entry.get("image")
        if not image_b64:
            continue
        try:
            image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        except Exception as exc:
            console.print(f"[red]Failed to decode image for visualization: {exc}[/red]")
            continue
        if entry.get("points"):
            image = draw_points(image, entry["points"])
        detections = entry.get("detections")
        if not detections and entry.get("boxes"):
            detections = [{
                "box_2d": box,
                "label": entry.get("prompt", "")
            } for box in entry["boxes"]]
        if detections:
            image = draw_detections(image, detections)
        masks = entry.get("masks") or []
        if masks:
            image = overlay_masks(image, masks)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        prompt = entry.get("prompt", "prompt").replace(" ", "_")
        image_idx = entry.get("image_index", 0)
        filename = Path(directory) / f"client_pointing_{timestamp}_{prompt}_{image_idx}.png"
        image.save(filename)
        console.print(f"Saved pointing visualization to {filename}", style="yellow")


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
                    entries: List[Dict] = []
                    if isinstance(result, dict):
                        entries = result.get("results") or []
                    elif isinstance(result, list):
                        entries = result
                    count = len(entries)
                    example = None
                    if entries and isinstance(entries[0], dict):
                        example = entries[0].get("prompt")
                    if not entries:
                        logging.warning(
                            "[PointingClient] No detections returned (HTTP 200); "
                            "check server logs for details.")
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
        save_visualizations: bool = False,
        viz_dir: str = "spot_pointing_outputs_cli",
    ) -> Dict:
        """Synchronous wrapper for predict_async."""
        with Timer(enable_print=False) as t:
            result = asyncio.run(
                self.predict_async(images, prompts, points, segmentation, detection))
        result["timings"]["total"] = t.elapsed_time
        if save_visualizations:
            try:
                _save_client_visualizations(result.get("results", []), viz_dir)
            except Exception as exc:  # pragma: no cover - debug helper
                console.print(f"[red]Failed to save pointing visualizations: {exc}[/red]")
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
    save_viz: bool = typer.Option(False, "--save-viz", help="Save annotated results to disk"),
    viz_dir: str = typer.Option("spot_pointing_outputs_cli", "--viz-dir", help="Directory for saved visualizations"),
):
    """Run predictions using the PointingGeminiSAM2 service."""
    try:
        # Clean up inputs
        prompts = [p.strip().strip("\"'").replace('\\"', '"').replace("\\'", "'").replace("\\", "") for p in prompt]
        image_paths = [p.strip().strip("\"'").replace('\\"', '"').replace("\\'", "'").replace("\\", "") for p in image]

        if not image_paths or not prompts:
            raise typer.BadParameter("Must provide at least one image (-i) and one prompt (-p)")

        client = PointingGeminiSAM2Client(host=host, port=port)
        result = client.predict(image_paths,
                                prompts,
                                points=points,
                                segmentation=segmentation,
                                detection=detection,
                                save_visualizations=save_viz,
                                viz_dir=viz_dir)

        # Print results
        console.rule("[bold blue]Results")
        for res in result["results"]:
            img_idx = res["image_index"]
            prompt_idx = res["prompt_index"]
            prompt = res["prompt"]
            
            console.print(f"\n[yellow]Image {img_idx + 1}, Prompt {prompt_idx + 1}: '{prompt}'[/yellow]")
            if res.get("points"):
                console.print(f"Points coordinates: {res['points']}", style="cyan")
            boxes = res.get("detections")
            if not boxes and res.get("boxes"):
                boxes = [{
                    "box_2d": box,
                    "label": res.get("prompt", "")
                } for box in res["boxes"]]
            if boxes:
                console.print("\nBounding boxes:", style="cyan")
                for i, box in enumerate(boxes, 1):
                    console.print(f"Box {i}: {box['box_2d']} ({box['label']})", style="cyan")
            if res.get("masks"):
                console.print(f"Segmentation masks: {len(res['masks'])}", style="magenta")

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
