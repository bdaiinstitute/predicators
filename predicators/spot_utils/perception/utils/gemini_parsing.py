"""Utility functions for parsing Gemini model responses."""

import json
import logging
from typing import Dict, List, Optional, Tuple, Union


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def parse_gemini_point_response(
    response: str,
    image_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
) -> Dict[str, List[Dict[str, Union[List[float], str]]]]:
    """Parse Gemini point response to extract point coordinates and labels.
    
    Args:
        response: Raw response from Gemini
        image_size: Optional size of the input image (width, height) for normalization
        normalize: Whether to normalize points to 0-1000 range
        
    Returns:
        Dictionary with points and labels
    """
    try:
        # Extract JSON from response
        start = response.find("[")
        end = response.rfind("]") + 1
        json_str = response[start:end]
        points_data = json.loads(json_str)
        
        # Validate and normalize points
        normalized_points = []
        for item in points_data:
            point = item["point"]
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"Invalid point format: {point}")
                
            if normalize and image_size:
                # Convert to normalized coordinates [0-1000]
                y, x = point
                y = y * 1000 / image_size[1]
                x = x * 1000 / image_size[0]
                point = [y, x]
            point = [
                _clamp(float(point[0]), 0.0, 1000.0),
                _clamp(float(point[1]), 0.0, 1000.0),
            ]
            
            normalized_points.append({
                "point": point,
                "label": str(item["label"])
            })
        
        return {"points": normalized_points}
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logging.error(f"Failed to parse Gemini point response: {e}")
        logging.error(f"Response: {response}")
        raise ValueError("Failed to parse Gemini point response") from e


def parse_gemini_detection_response(
    response: str,
    image_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
) -> Dict[str, List[Dict[str, Union[List[float], str]]]]:
    """Parse Gemini detection response to extract bounding boxes and labels.
    
    Args:
        response: Raw response from Gemini
        image_size: Optional size of the input image (width, height) for normalization
        normalize: Whether to normalize boxes to 0-1000 range
        
    Returns:
        Dictionary with bounding boxes and labels
    """
    try:
        # Extract JSON from response
        start = response.find("[")
        end = response.rfind("]") + 1
        json_str = response[start:end]
        detection_data = json.loads(json_str)
        
        # Validate and normalize boxes
        normalized_boxes = []
        for item in detection_data:
            box = item["box_2d"]
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f"Invalid box format: {box}")
                
            if normalize and image_size:
                # Convert to normalized coordinates [0-1000]
                x1, y1, x2, y2 = box
                x1 = x1 * 1000 / image_size[0]
                y1 = y1 * 1000 / image_size[1]
                x2 = x2 * 1000 / image_size[0]
                y2 = y2 * 1000 / image_size[1]
                box = [x1, y1, x2, y2]
            
            normalized_boxes.append({
                "box_2d": box,
                "label": str(item["label"])
            })
        
        return {"detections": normalized_boxes}
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logging.error(f"Failed to parse Gemini detection response: {e}")
        logging.error(f"Response: {response}")
        raise ValueError("Failed to parse Gemini detection response") from e


def denormalize_point(
    point: List[float],
    image_size: Tuple[int, int],
) -> List[float]:
    """Convert normalized point [0-1000] back to image coordinates.
    
    Args:
        point: Normalized point coordinates [y, x]
        image_size: Size of the image (width, height)
        
    Returns:
        Point coordinates in image space
    """
    y, x = point
    y = _clamp(float(y), 0.0, 1000.0) * image_size[1] / 1000
    x = _clamp(float(x), 0.0, 1000.0) * image_size[0] / 1000
    return [y, x]


def denormalize_box(
    box: List[float],
    image_size: Tuple[int, int],
) -> List[float]:
    """Convert normalized box [0-1000] back to image coordinates.
    
    Args:
        box: Normalized box coordinates [x1, y1, x2, y2]
        image_size: Size of the image (width, height)
        
    Returns:
        Box coordinates in image space
    """
    x1, y1, x2, y2 = box
    x1 = _clamp(float(x1), 0.0, 1000.0) * image_size[0] / 1000
    y1 = _clamp(float(y1), 0.0, 1000.0) * image_size[1] / 1000
    x2 = _clamp(float(x2), 0.0, 1000.0) * image_size[0] / 1000
    y2 = _clamp(float(y2), 0.0, 1000.0) * image_size[1] / 1000
    return [x1, y1, x2, y2]
