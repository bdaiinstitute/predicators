"""High-level helpers for Gemini-based object pointing/detection."""

from __future__ import annotations

import argparse
import base64
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
from PIL import Image

from predicators.spot_utils.perception.pointing_gemini_sam2_client import \
    PointingGeminiSAM2Client
from predicators.spot_utils.perception.utils.gemini_parsing import \
    denormalize_box, denormalize_point


ImageLike = Union[str, Image.Image, np.ndarray]


@dataclass
class GeminiPointPrediction:
    """Pixel selected by Gemini's pointing response."""
    label: str
    pixel: Tuple[int, int]


@dataclass
class GeminiBoxPrediction:
    """Bounding-box detection returned by Gemini."""
    label: str
    box: Tuple[float, float, float, float]

    @property
    def centroid(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)))


@dataclass
class GeminiMaskPrediction:
    """Segmentation mask (SAM2) returned by Gemini service."""
    label: str
    image: Image.Image
    source: str = "point"
    anchor_index: int = 0
    variant_index: int = 0


@dataclass
class GeminiPointingResult:
    """Aggregated pointing/detection results for an image/prompt pair."""
    image_index: int
    prompt: str
    points: List[GeminiPointPrediction]
    boxes: List[GeminiBoxPrediction]
    masks: List[GeminiMaskPrediction] = field(default_factory=list)
    image_path: Optional[str] = None

    def primary_pixel(self) -> Optional[Tuple[int, int]]:
        """Return the best-guess pixel (points first, detection centroid second)."""
        if self.points:
            return self.points[0].pixel
        if self.boxes:
            return self.boxes[0].centroid
        return None


def _to_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise ValueError("numpy image must have dtype uint8")
        if image.ndim == 2:
            return Image.fromarray(image)
        if image.ndim == 3:
            return Image.fromarray(image)
        raise ValueError(f"Unsupported ndarray shape {image.shape}")
    if isinstance(image, str):
        return Image.open(image)
    raise TypeError(f"Unsupported image type: {type(image)}")


T = TypeVar("T")


def _coerce_sequence(value: Union[T, Sequence[T]]) -> List[T]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _denormalize_points(points: List[dict], width: int,
                        height: int) -> List[GeminiPointPrediction]:
    results: List[GeminiPointPrediction] = []
    for entry in points:
        y_norm, x_norm = entry["point"]
        y, x = denormalize_point([y_norm, x_norm], (width, height))
        pixel = (int(round(x)), int(round(y)))
        results.append(GeminiPointPrediction(entry.get("label", ""), pixel))
    return results


def _denormalize_boxes(boxes: List[List[float]], width: int, height: int,
                       label: str) -> List[GeminiBoxPrediction]:
    results: List[GeminiBoxPrediction] = []
    for box in boxes:
        x1, y1, x2, y2 = denormalize_box(box, (width, height))
        results.append(GeminiBoxPrediction(label, (x1, y1, x2, y2)))
    return results


def _decode_masks(mask_entries: List[dict], width: int,
                  height: int, fallback_label: str) -> List[GeminiMaskPrediction]:
    decoded: List[GeminiMaskPrediction] = []
    for entry in mask_entries:
        mask_b64 = entry.get("mask_png")
        if not mask_b64:
            continue
        try:
            mask_img = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")
        except Exception as exc:  # pragma: no cover - debug helper
            logging.debug("Failed to decode Gemini mask: %s", exc)
            continue
        if mask_img.size != (width, height):
            mask_img = mask_img.resize((width, height), Image.NEAREST)
        decoded.append(
            GeminiMaskPrediction(
                label=entry.get("prompt", fallback_label),
                image=mask_img,
                source=str(entry.get("source", "point")),
                anchor_index=int(entry.get("anchor_index", 0)),
                variant_index=int(entry.get("variant_index", 0)),
            ))
    return decoded


def point_objects_with_gemini(
        images: Union[ImageLike, Sequence[ImageLike]],
        prompts: Union[str, Sequence[str]],
        *,
        host: str = "localhost",
        port: int = 7100,
        detection: bool = False,
        segmentation: bool = False) -> List[GeminiPointingResult]:
    """Call the Gemini pointing service and parse the response."""
    image_list = [_to_pil(img) for img in _coerce_sequence(images)]
    prompt_list = [str(p) for p in _coerce_sequence(prompts)]
    client = PointingGeminiSAM2Client(host=host, port=port)
    response = client.predict(image_list,
                              prompt_list,
                              points=not detection,
                              segmentation=segmentation,
                              detection=detection)

    raw_results = response.get("results", [])
    parsed: List[GeminiPointingResult] = []
    for entry in raw_results:
        width = entry.get("image_width")
        height = entry.get("image_height")
        if width is None or height is None:
            logging.warning("[GeminiPointing] Missing image size in response; "
                            "skipping entry.")
            continue
        points_data = entry.get("points") or []
        boxes_data = entry.get("boxes") or []
        mask_entries = entry.get("masks") or []
        label = entry.get("prompt", "")
        points = _denormalize_points(points_data, width, height)
        boxes = _denormalize_boxes(boxes_data, width, height, label)
        masks = _decode_masks(mask_entries, width, height, label)
        parsed.append(
            GeminiPointingResult(
                image_index=entry.get("image_index", 0),
                prompt=entry.get("prompt", ""),
                points=points,
                boxes=boxes,
                masks=masks,
                image_path=None,
            ))
    return parsed


def point_single_image(
        image: ImageLike,
        prompt: str,
        *,
        host: str = "localhost",
        port: int = 7100,
        detection: bool = False,
        segmentation: bool = False) -> Optional[GeminiPointingResult]:
    """Convenience wrapper for a single image/prompt pair."""
    results = point_objects_with_gemini(image,
                                        prompt,
                                        host=host,
                                        port=port,
                                        detection=detection,
                                        segmentation=segmentation)
    return results[0] if results else None


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Gemini object pointing CLI")
    parser.add_argument("image",
                        type=Path,
                        help="Path to the image to analyze.")
    parser.add_argument("prompt", type=str, help="Prompt describing the object.")
    parser.add_argument("--host",
                        default="localhost",
                        help="Gemini pointing server host.")
    parser.add_argument("--port",
                        default=7100,
                        type=int,
                        help="Gemini pointing server port.")
    parser.add_argument("--detection",
                        action="store_true",
                        help="Request detection boxes instead of direct points.")
    parser.add_argument("--segmentation",
                        action="store_true",
                        help="Request SAM2 masks for debugging.")
    args = parser.parse_args()

    result = point_single_image(args.image,
                                args.prompt,
                                host=args.host,
                                port=args.port,
                                detection=args.detection,
                                segmentation=args.segmentation)
    if result is None:
        print("No pointing result returned.")
        return
    print(f"Prompt: {result.prompt}")
    if result.points:
        print("Points:")
        for point in result.points:
            print(f"  {point.label}: {point.pixel}")
    if result.boxes:
        print("Detections:")
        for box in result.boxes:
            print(f"  {box.label}: {box.box} (centroid={box.centroid})")


if __name__ == "__main__":  # pragma: no cover - CLI utility
    _cli()
