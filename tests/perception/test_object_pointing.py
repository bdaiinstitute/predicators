"""Tests for the Gemini object pointing helpers."""

import base64
import io
from pathlib import Path
from unittest import mock

from PIL import Image

from predicators.spot_utils.perception import object_pointing


def _fake_response(points=None, boxes=None, masks=None):
    return {
        "results": [{
            "image_index": 0,
            "prompt_index": 0,
            "prompt": "cup",
            "points": points,
            "boxes": boxes,
            "masks": masks,
            "image_width": 640,
            "image_height": 480,
            "image": "",
        }],
        "timings": {},
    }


def test_pointing_parses_points():
    points = [{"point": [500.0, 250.0], "label": "cup"}]
    fake_payload = _fake_response(points=points, boxes=None)
    image_path = str(Path("tmp.png"))

    with mock.patch(
            "predicators.spot_utils.perception.object_pointing."
            "PointingGeminiSAM2Client") as mock_client:
        instance = mock_client.return_value
        instance.predict.return_value = fake_payload
        result = object_pointing.point_single_image(image_path,
                                                    "cup",
                                                    host="localhost",
                                                    port=0)

    assert result is not None
    assert result.points[0].pixel == (160, 240)  # denormalized
    assert not result.boxes


def test_pointing_parses_boxes_and_centroid():
    # Boxes are normalized [xmin, ymin, xmax, ymax].
    boxes = [[150.0, 200.0, 450.0, 600.0]]
    fake_payload = _fake_response(points=[], boxes=boxes)
    image_path = str(Path("tmp.png"))

    with mock.patch(
            "predicators.spot_utils.perception.object_pointing."
            "PointingGeminiSAM2Client") as mock_client:
        instance = mock_client.return_value
        instance.predict.return_value = fake_payload
        result = object_pointing.point_single_image(image_path,
                                                    "cup",
                                                    host="localhost",
                                                    port=0,
                                                    detection=True)

    assert result is not None
    assert not result.points
    assert result.boxes[0].centroid == (192, 192)


def test_pointing_parses_masks():
    mask = Image.new("L", (10, 10), color=255)
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    mask_payload = base64.b64encode(buffer.getvalue()).decode()
    masks = [{
        "mask_png": mask_payload,
        "prompt": "cup",
        "source": "point",
        "anchor_index": 0,
        "variant_index": 0,
    }]
    fake_payload = _fake_response(points=[], boxes=[], masks=masks)
    image_path = str(Path("tmp.png"))

    with mock.patch(
            "predicators.spot_utils.perception.object_pointing."
            "PointingGeminiSAM2Client") as mock_client:
        instance = mock_client.return_value
        instance.predict.return_value = fake_payload
        result = object_pointing.point_single_image(image_path,
                                                    "cup",
                                                    host="localhost",
                                                    port=0,
                                                    segmentation=True)

    assert result is not None
    assert len(result.masks) == 1
    assert result.masks[0].label == "cup"
