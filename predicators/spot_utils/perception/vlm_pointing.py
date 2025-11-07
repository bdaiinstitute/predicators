"""Adapter that proxies hand-view pointing requests to Gemini."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from bosdyn.client import math_helpers
from PIL import Image, ImageDraw

from predicators.settings import CFG
from predicators.spot_utils.perception.object_pointing import \
    GeminiPointingResult, point_single_image
from predicators.spot_utils.perception.perception_structs import \
    ObjectDetectionID, RGBDImageWithContext
from predicators.structs import Object

_POINTING_SERVICE_AVAILABLE = True
_POINTING_WARNING_EMITTED = False


@dataclass
class VLMPointingResult:
    """Record of a pixel selected for a target object."""
    object_name: str
    pixel: Tuple[int, int]
    orientation: Optional[math_helpers.Quat]
    source: str = "gemini"


def compute_pointing_result(  # pylint: disable=unused-argument
        target_object: Object,
        images: Dict[str, RGBDImageWithContext],
        detection_artifacts: Optional[Dict[str, Any]],
        detection_id_to_obj: Dict[ObjectDetectionID, Object],
        rng: Optional[Any],
        camera_name: str = "hand_color_image") -> Optional[VLMPointingResult]:
    """Select a pixel for the target object via Gemini pointing service."""
    del detection_artifacts, detection_id_to_obj, rng  # unused in Gemini path
    global _POINTING_SERVICE_AVAILABLE, _POINTING_WARNING_EMITTED  # pylint:disable=global-statement

    if not CFG.spot_use_vlm_pointing:
        return None
    if not _POINTING_SERVICE_AVAILABLE:
        if not _POINTING_WARNING_EMITTED:
            logging.warning("Gemini pointing disabled after connection failure; "
                            "skipping until service becomes reachable again.")
            _POINTING_WARNING_EMITTED = True
        return None
    if camera_name not in images:
        logging.debug("[GeminiPointing] Camera %s missing; cannot compute pixel.",
                      camera_name)
        return None
    rgbd = images[camera_name]
    try:
        pil_image = Image.fromarray(rgbd.rgb)
    except Exception as exc:  # pragma: no cover - defensive
        logging.debug("Failed to convert image for Gemini pointing: %s", exc)
        return None

    detection_mode = CFG.spot_pointing_mode.lower().startswith("det")
    prompt = f"Point to the {target_object.name}"
    try:
        result = point_single_image(pil_image,
                                    prompt,
                                    host=CFG.spot_pointing_host,
                                    port=CFG.spot_pointing_port,
                                    detection=detection_mode)
    except Exception as exc:  # pragma: no cover - network failure
        logging.warning("Gemini pointing failed: %s", exc)
        _POINTING_SERVICE_AVAILABLE = False
        _POINTING_WARNING_EMITTED = False
        return None
    if result is None:
        return None
    _POINTING_SERVICE_AVAILABLE = True
    _POINTING_WARNING_EMITTED = False
    if CFG.spot_pointing_debug_visuals:
        _save_pointing_debug_visual(target_object.name, rgbd, result)
    pixel = result.primary_pixel()
    if pixel is None:
        return None
    return VLMPointingResult(target_object.name,
                             pixel,
                             orientation=None,
                             source="gemini")


def _save_pointing_debug_visual(target_name: str,
                                rgbd: RGBDImageWithContext,
                                result: GeminiPointingResult) -> None:
    """Best-effort visualization of Gemini pointing results."""
    try:
        outdir = Path(CFG.spot_pointing_debug_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(rgbd.rgb.copy())
        draw = ImageDraw.Draw(image)
        for point in result.points:
            x, y = point.pixel
            r = 8
            draw.ellipse((x - r, y - r, x + r, y + r),
                         outline="red",
                         width=2)
            draw.text((x + 5, y - 5), point.label or "", fill="red")
        for box in result.boxes:
            x1, y1, x2, y2 = box.box
            draw.rectangle((x1, y1, x2, y2), outline="cyan", width=2)
            draw.text((x1 + 5, y1 - 5), box.label or "", fill="cyan")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = outdir / \
            f"{timestamp}_{target_name}_{rgbd.camera_name}_pointing.png"
        image.save(filename)
        logging.info("[GeminiPointing] Saved debug visualization to %s",
                     filename)
    except Exception as exc:  # pragma: no cover - best-effort logging
        logging.debug("Failed to save pointing debug visual: %s", exc)
