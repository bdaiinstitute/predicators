"""Object-specific grasp selectors."""

from typing import Any, Callable, Dict, Tuple

from predicators.spot_utils.perception.cv2_utils import \
    find_color_based_centroid
from predicators.spot_utils.perception.perception_structs import \
    AprilTagObjectDetectionID, ObjectDetectionID, RGBDImageWithContext


def _get_platform_grasp_pixel(rgbds: Dict[str, RGBDImageWithContext],
                              artifacts: Dict[str, Any],
                              camera_name: str) -> Tuple[int, int]:
    # This assumes that we have just navigated to the april tag and are now
    # looking down at the platform. We crop the top half of the image and
    # then use CV2 to find the blue handle inside of it.
    del artifacts  # not used
    rgb = rgbds[camera_name].rgb
    half_height = rgb.shape[0] // 2

    # Crop the bottom half of the image.
    img = rgb[half_height:]

    # Use CV2 to find a pixel.
    lo, hi = ((0, 130, 130), (130, 255, 255))

    cropped_centroid = find_color_based_centroid(img, lo, hi)
    if cropped_centroid is None:
        raise RuntimeError("Could not find grasp for platform from image.")

    # Undo cropping.
    cropped_x, cropped_y = cropped_centroid
    x = cropped_x
    y = cropped_y + half_height

    return (x, y)


def _get_movoroom_chair_grasp_pixel(rgbds: Dict[str, RGBDImageWithContext],
                                    artifacts: Dict[str, Any],
                                    camera_name: str) -> Tuple[int, int]:
    # This is a hack because the apriltag was too low on the chair, but 
    # placing it higher would have cause the gripper to damage it quickly
    
    try:
        april_detection = artifacts["april"][camera_name][AprilTagObjectDetectionID(11)]
    except KeyError:
        raise ValueError(f"{AprilTagObjectDetectionID(11)} not detected in {camera_name}")
    pr, pc = april_detection.center
    corners = april_detection.corners

    r = pr
    height = (corners[3][1] + corners[2][1]) / 2 - (corners[1][1] + corners[0][1]) / 2
    c = pc - 3.5 * height

    import cv2
    rgb = rgbds[camera_name].rgb
    cv2.circle(rgb, (int(r), int(c)), 5, (0, 255, 0), -1)
    cv2.imshow('image', rgb)
    cv2.waitKey(0)

    return int(r), int(c)


# Maps an object ID to a function from rgbds, artifacts and camera to pixel.
OBJECT_SPECIFIC_GRASP_SELECTORS: Dict[ObjectDetectionID, Callable[
    [Dict[str,
          RGBDImageWithContext], Dict[str, Any], str], Tuple[int, int]]] = {
              # Platform-specific grasp selection.
              AprilTagObjectDetectionID(411):
              _get_platform_grasp_pixel,
              AprilTagObjectDetectionID(11):
              _get_movoroom_chair_grasp_pixel,
          }
