"""Object-specific grasp selectors."""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from predicators.spot_utils.perception.cv2_utils import \
    find_color_based_centroid
from predicators.spot_utils.perception.perception_structs import \
    AprilTagObjectDetectionID, LanguageObjectDetectionID, ObjectDetectionID, \
    RGBDImageWithContext

ball_prompt = "/".join([
    "small white ball", "ping-pong ball", "snowball", "cotton ball",
    "white button"
])
ball_obj = LanguageObjectDetectionID(ball_prompt)
cup_obj = LanguageObjectDetectionID("yellow hoop toy/yellow donut")


def _get_platform_grasp_pixel(rgbds: Dict[str, RGBDImageWithContext],
                              artifacts: Dict[str, Any], camera_name: str,
                              extra_info: Optional[Any]) -> Tuple[int, int]:
    # This assumes that we have just navigated to the april tag and are now
    # looking down at the platform. We crop the top half of the image and
    # then use CV2 to find the blue handle inside of it.
    del extra_info, artifacts  # not used
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


def _get_ball_grasp_pixel(rgbds: Dict[str, RGBDImageWithContext],
                          artifacts: Dict[str, Any], camera_name: str,
                          extra_info: Optional[Any]) -> Tuple[int, int]:
    del rgbds, extra_info
    detections = artifacts["language"]["object_id_to_img_detections"]
    try:
        seg_bb = detections[ball_obj][camera_name]
    except KeyError:
        raise ValueError(f"{ball_obj} not detected in {camera_name}")
    # Select the last (bottom-most) pixel from the mask. We do this because the
    # back finger of the robot gripper might displace the ball during grasping
    # if we try to grasp at the center.
    mask = seg_bb.mask
    pixels_in_mask = np.where(mask)
    return (pixels_in_mask[1][-1], pixels_in_mask[0][-1])


def _get_cup_grasp_pixel(rgbds: Dict[str, RGBDImageWithContext],
                         artifacts: Dict[str, Any], camera_name: str,
                         extra_info: Optional[Any]) -> Tuple[int, int]:
    # del rgbds
    detections = artifacts["language"]["object_id_to_img_detections"]
    try:
        seg_bb = detections[cup_obj][camera_name]
    except KeyError:
        raise ValueError(f"{cup_obj} not detected in {camera_name}")
    mask = seg_bb.mask

    from scipy.ndimage import convolve
    small_kernel = np.ones((3, 3))
    large_kernel = np.ones((10, 10))
    convolved_mask = convolve(mask.astype(np.uint8), small_kernel, mode="constant")
    smoothed_mask = (convolved_mask > 0)
    convolved_smoothed_mask = convolve(smoothed_mask.astype(np.uint8), large_kernel, mode="constant")
    surrounded_mask = (convolved_smoothed_mask == convolved_smoothed_mask.max())
    
    # import imageio.v2 as iio
    # iio.imsave("original.png", 255 * mask.astype(np.uint8))
    # iio.imsave("smoothed.png", 255 * smoothed_mask.astype(np.uint8))
    # iio.imsave("surrounded.png", 255 * surrounded_mask.astype(np.uint8))

    pixels_in_mask = np.where(surrounded_mask)
    
    mask_size = len(pixels_in_mask[0])
    percentile_idx = int(mask_size / 20)
    idx = np.argsort(pixels_in_mask[0])[percentile_idx]
    pixel = (pixels_in_mask[1][idx],
            pixels_in_mask[0][idx])
    
    import cv2
    rgbd = rgbds[camera_name]
    bgr = cv2.cvtColor(rgbd.rgb, cv2.COLOR_RGB2BGR)
    cv2.circle(bgr, pixel, 5, (0, 255, 0), -1)
    cv2.imshow("Selected grasp", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pixel


# Maps an object ID to a function from rgbds, artifacts and camera to pixel.
OBJECT_SPECIFIC_GRASP_SELECTORS: Dict[ObjectDetectionID, Callable[
    [Dict[str, RGBDImageWithContext], Dict[str, Any], str, Optional[Any]],
    Tuple[int, int]]] = {
        # Platform-specific grasp selection.
        AprilTagObjectDetectionID(411): _get_platform_grasp_pixel,
        # Ball-specific grasp selection.
        ball_obj: _get_ball_grasp_pixel,
        # Cup-specific grasp selection.
        cup_obj: _get_cup_grasp_pixel
    }
