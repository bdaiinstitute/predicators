import asyncio
import base64
import io
import os
from typing import Dict, Union, List

import httpx
import numpy as np
from PIL import Image, ImageDraw
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from predicators.utils import Timer

# Initialize console globally
console = Console()

def draw_points(image, points):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    for point in points:
        draw.ellipse(
            [point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5],
            fill="blue",
            outline="white",
        )
    return image

def encode_image(image: Union[str, Image.Image]) -> str:
    """Encode image to base64 string"""
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
    """Decode base64 string to PIL Image"""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes))


def decode_rle_mask(rle_mask: dict) -> np.ndarray:
    """Decode RLE mask to numpy array"""
    from pycocotools import mask as mask_util

    rle = {
        "size": rle_mask["size"],
        "counts": rle_mask["counts"].encode()
        if isinstance(rle_mask["counts"], str)
        else rle_mask["counts"],
    }
    return mask_util.decode(rle)


def draw_mask(
    image: Image.Image, mask: np.ndarray, alpha: float = 0.5, color: tuple = (255, 0, 0)
) -> Image.Image:
    """Draw a binary mask on an image with transparency"""
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Convert binary mask to image coordinates
    mask_image = Image.fromarray((mask * 255).astype("uint8"), "L")

    # Create colored overlay
    colored_overlay = Image.new("RGBA", image.size, (*color, int(255 * alpha)))
    overlay.paste(colored_overlay, mask=mask_image)

    return Image.alpha_composite(image, overlay).convert("RGB")


def draw_box(image: Image.Image, box: list) -> Image.Image:
    """Draw bounding box on image with confidence score

    Args:
        image: PIL Image to draw on
        box: List of [x1, y1, x2, y2, confidence, class_id]
    """
    try:
        draw = ImageDraw.Draw(image)
        if len(box) >= 6:  # [x1, y1, x2, y2, conf, class_id]
            x1, y1, x2, y2, conf, _ = box
            # Draw rectangle
            draw.rectangle(
                [float(x1), float(y1), float(x2), float(y2)], outline="red", width=2
            )
            # Draw confidence score
            conf_text = f"{conf:.2f}"
            draw.text(
                (float(x1), float(y1) - 20),
                conf_text,
                fill="red",
                stroke_width=1,
                stroke_fill="white",
            )
        return image
    except Exception as e:
        print(f"Error drawing box {box}: {str(e)}")
        return image


def render_result(image: Image.Image, result: Dict) -> Dict[str, Image.Image]:
    """Render points, boxes, and masks on the image"""
    rendered = {}

    # Draw points
    if result.get("points"):
        try:
            points_img = image.copy()
            points_img = draw_points(points_img, result["points"])
            rendered["points_img"] = points_img
        except Exception as e:
            print(f"Error rendering points: {str(e)}")

    # Draw boxes
    if result.get("boxes"):
        try:
            boxes_img = image.copy()
            for box in result["boxes"]:
                boxes_img = draw_box(boxes_img, box)
            rendered["boxes_img"] = boxes_img
        except Exception as e:
            print(f"Error rendering boxes: {str(e)}")

    # Draw masks
    if result.get("masks"):
        try:
            masks_img = image.copy()
            for i, rle_mask in enumerate(result["masks"]):
                # Decode RLE mask
                mask = decode_rle_mask(rle_mask)
                # Use different colors for multiple masks
                color = (255, 0, 0) if i == 0 else (0, 255, 0)
                masks_img = draw_mask(masks_img, mask, alpha=0.5, color=color)
            rendered["masks_img"] = masks_img
        except Exception as e:
            print(f"Error rendering masks: {str(e)}")

    return rendered


class MolmoSAM2Client:
    def __init__(self, host: str = "localhost", port: int = 8100):
        """Initialize client with host and port"""
        self.endpoint = f"http://{host}:{port}/molmo_sam2_service"

    async def predict_async(
        self, 
        images: Union[str, Image.Image, List[Union[str, Image.Image]]], 
        prompts: Union[str, List[str]], 
        render: bool = False
    ) -> Dict:
        """Async prediction with support for multiple images and prompts
        
        Args:
            images: Single image (path or PIL) or list of images
            prompts: Single prompt or list of prompts
            render: Whether to render results
        """
        timings = {}
        console.rule("[bold blue]Starting MolmoSAM2 Client Request")
        console.print(f"Endpoint: {self.endpoint}", style="cyan")

        # Convert single inputs to lists
        if isinstance(images, (str, Image.Image)):
            images = [images]
        if isinstance(prompts, str):
            prompts = [prompts]

        # Process images with progress spinner
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

        # Print batch information
        console.print(f"\n[yellow]Batch Information:[/yellow]")
        console.print(f"Number of images: {len(processed_images)}")
        console.print(f"Number of prompts: {len(prompts)}")
        console.print(f"Total combinations: {len(processed_images) * len(prompts)}")
        console.print("\n[yellow]Prompts:[/yellow]")
        for i, prompt in enumerate(prompts, 1):
            console.print(f"{i}. '{prompt}'")

        # Make server request
        console.print("\n[yellow]Making Server Request...[/yellow]")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Waiting for server response...")
            
            with Timer(enable_print=False) as t:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.endpoint,
                        json={
                            "images": processed_images,
                            "prompts": prompts,
                            "render": render
                        },
                        timeout=120.0
                    )
                    
                    progress.update(task, completed=True)

                    if response.status_code != 200:
                        console.print(f"[red]Request failed with status {response.status_code}[/red]")
                        raise Exception(f"Request failed: {response.text}")

                    result = response.json()
                    console.print("[green]Server request completed successfully[/green]")
            timings["server_request"] = t.elapsed_time

        console.rule("[bold blue]Request Completed")
        return {"results": result, "timings": timings}

    def predict(
        self, 
        images: Union[str, Image.Image, List[Union[str, Image.Image]]], 
        prompts: Union[str, List[str]], 
        render: bool = False
    ) -> Dict:
        with Timer(enable_print=False) as t:
            result = asyncio.run(self.predict_async(images, prompts, render))
        result["timings"]["total"] = t.elapsed_time
        return result


# NOTE: uncomment this to run as a standalone script
# import typer

# app = typer.Typer()


# @app.command()
# def predict(
#     image: List[str] = typer.Option(
#         None,  # Default value
#         "--image", "-i",
#         help="Image path(s). Can be specified multiple times for multiple images.",
#         callback=lambda x: x or [],  # Convert None to empty list
#     ),
#     prompt: List[str] = typer.Option(
#         None,  # Default value
#         "--prompt", "-p",
#         help="Text prompt(s). Can be specified multiple times for multiple prompts.",
#         callback=lambda x: x or [],  # Convert None to empty list
#     ),
#     host: str = typer.Option("localhost", help="Host of the MolmoSAM2 service"),
#     port: int = typer.Option(8100, help="Port of the MolmoSAM2 service"),
#     output_prefix: str = typer.Option("segmentation_result", help="Prefix for output files"),
#     server_render: bool = typer.Option(
#         False,  # Default to False for client-side rendering
#         "--server-render/--no-server-render",
#         help="Whether to render results on server side",
#     ),
# ):
#     """Run predictions using the MolmoSAM2 service with multiple images and prompts
    
#     Examples:
#         # Single image and prompt
#         molmo_sam2_http_client -i image.jpg -p "Point at the door"
        
#         # Multiple images, single prompt
#         molmo_sam2_http_client -i image1.jpg -i image2.jpg -p "Point at the door"
        
#         # Single image, multiple prompts
#         molmo_sam2_http_client -i image.jpg -p "Point at the door" -p "Point at the window"
        
#         # Multiple images and prompts
#         molmo_sam2_http_client -i image1.jpg -i image2.jpg -p "Point at the door" -p "Point at the window"
#     """
#     try:
#         # Clean up any quotes or escapes from prompts
#         prompts = [
#             p.strip()
#             .strip('"\'')
#             .replace('\\"', '"')
#             .replace("\\'", "'")
#             .replace("\\", "")
#             for p in prompt
#         ]
        
#         # Clean up any quotes or escapes from image paths
#         image_paths = [
#             p.strip()
#             .strip('"\'')
#             .replace('\\"', '"')
#             .replace("\\'", "'")
#             .replace("\\", "")
#             for p in image
#         ]

#         if not image_paths or not prompts:
#             raise typer.BadParameter(
#                 "Must provide at least one image (-i) and one prompt (-p)"
#             )

#         # Debug print the processed inputs
#         print("\nProcessed inputs:")
#         print("Images:", image_paths)
#         print("Prompts:", prompts)

#         print(f"Processing {len(image_paths)} images with {len(prompts)} prompts:")
#         print("\nImages:")
#         for i, path in enumerate(image_paths, 1):
#             print(f"{i}. {path}")
#         print("\nPrompts:")
#         for i, prompt in enumerate(prompts, 1):
#             print(f"{i}. '{prompt}'")

#         client = MolmoSAM2Client(host=host, port=port)

#         # Time the entire process
#         with Timer(enable_print=False) as total_timer:
#             # Get raw results from server
#             result = client.predict(image_paths, prompts, render=server_render)

#             # Client-side rendering and saving
#             with Progress(
#                 SpinnerColumn(),
#                 TextColumn("[progress.description]{task.description}"),
#                 console=console
#             ) as progress:
#                 save_task = progress.add_task(
#                     "Processing results...", 
#                     total=len(result["results"])  # One task per result
#                 )

#                 tmp_dir = os.path.join("src", "models", "tmp")
#                 os.makedirs(tmp_dir, exist_ok=True)

#                 # Process each result
#                 for res in result["results"]:
#                     img_idx = res["image_index"]
#                     prompt_idx = res["prompt_index"]
                    
#                     # Load original image for rendering
#                     image = Image.open(image_paths[img_idx])
                    
#                     # Render results
#                     rendered = render_result(image, res)
                    
#                     # Save rendered images
#                     for key, img in rendered.items():
#                         if isinstance(img, Image.Image):
#                             path = os.path.join(
#                                 tmp_dir, 
#                                 f"{key}_{output_prefix}_img{img_idx}_prompt{prompt_idx}.png"
#                             )
#                             img.save(path)
#                             console.print(f"[green]Saved[/green] {key} for image {img_idx + 1}, prompt {prompt_idx + 1}")
                    
#                     progress.advance(save_task)

#             # Print timing information
#             console.rule("[bold blue]Performance Summary")
#             console.print("\n[yellow]Timing Breakdown:[/yellow]")
#             console.print(f"Preprocessing: {result['timings']['preprocessing']:.3f}s", style="cyan")
#             console.print(f"Server request: {result['timings']['server_request']:.3f}s", style="cyan")
#             console.print(f"Total time: {result['timings']['total']:.3f}s", style="green")

#             # Print detection results
#             console.rule("[bold blue]Detection Results")
#             for res in result["results"]:
#                 img_idx = res["image_index"]
#                 prompt_idx = res["prompt_index"]
#                 prompt = res["prompt"]
                
#                 console.print(f"\n[yellow]Image {img_idx + 1}, Prompt {prompt_idx + 1}: '{prompt}'[/yellow]")
#                 console.print(f"Points coordinates: {res['points']}", style="cyan")
#                 if res.get("boxes"):
#                     console.print("\nBounding boxes (x1, y1, x2, y2, confidence, class_id):", style="cyan")
#                     for i, box in enumerate(res["boxes"], 1):
#                         console.print(f"Box {i}: {[round(x, 2) for x in box]}", style="cyan")

#     except Exception as e:
#         console.print(f"\n[red]Error: {str(e)}[/red]")
#         console.print("\n[yellow]Make sure to:[/yellow]")
#         console.print("1. Start the server first: [cyan]python -m src.models.segmentation.molmo_sam2_http_server[/cyan]")
#         console.print("2. Wait a few seconds for the server to initialize")
#         console.print("3. Try the request again")


# if __name__ == "__main__":
#     app()
