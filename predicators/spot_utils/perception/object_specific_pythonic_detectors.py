"""Object-specific arbitrary pythonic functions for detection."""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from bosdyn.client import math_helpers

from predicators.settings import CFG
from predicators.spot_utils.perception.cv2_utils import \
    find_color_based_centroid
from predicators.spot_utils.perception.object_detection import \
    detect_objects, visualize_all_artifacts
from predicators.spot_utils.perception.perception_structs import \
    LanguageObjectDetectionID, PythonicObjectDetectionID, \
    RGBDImageWithContext, SegmentedBoundingBox
from predicators.spot_utils.utils import get_graph_nav_dir


def detect_bowl(
        rgbds: Dict[str,
                    RGBDImageWithContext]) -> Optional[math_helpers.SE3Pose]:
    # ONLY use the hand camera (which we assume is looking down)
    # because otherwise it's impossible to see the top/bottom.
    hand_camera = "hand_color_image"
    assert hand_camera in rgbds
    rgbds = {hand_camera: rgbds[hand_camera]}
    # Start by using vision-language.
    language_id = LanguageObjectDetectionID("large cup")
    detections, artifacts = detect_objects([language_id], rgbds)
    if not detections:
        return None
    # Crop using the bounding box. If there were multiple detections,
    # choose the highest scoring one.
    obj_id_to_img_detections = artifacts["language"][
        "object_id_to_img_detections"]
    img_detections = obj_id_to_img_detections[language_id]
    assert len(img_detections) > 0
    best_seg_bb: Optional[SegmentedBoundingBox] = None
    best_seg_bb_score = -np.inf
    best_camera: Optional[str] = None
    for camera, seg_bb in img_detections.items():
        if seg_bb.score > best_seg_bb_score:
            best_seg_bb_score = seg_bb.score
            best_seg_bb = seg_bb
            best_camera = camera
    assert best_camera is not None
    assert best_seg_bb is not None
    x1, y1, x2, y2 = best_seg_bb.bounding_box
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    best_rgb = rgbds[best_camera].rgb
    height, width = best_rgb.shape[:2]
    r_min = min(max(int(y_min), 0), height)
    r_max = min(max(int(y_max), 0), height)
    c_min = min(max(int(x_min), 0), width)
    c_max = min(max(int(x_max), 0), width)
    cropped_img = best_rgb[r_min:r_max, c_min:c_max]
    # Look for the blue tape inside the bounding box.
    lo, hi = ((0, 145, 145), (90, 255, 255))
    centroid = find_color_based_centroid(cropped_img, lo, hi)
    blue_tape_found = (centroid is not None)
    # If the blue tape was found, assume that the bowl is oriented
    # upside-down; otherwise, it's right-side up.
    if blue_tape_found:
        roll = np.pi
        print("Detected blue tape; bowl is upside-down!")
    else:
        roll = 0.0
        print("Did NOT detect blue tape; bowl is right side-up!")
    rot = math_helpers.Quat.from_roll(roll)
    # Use the x, y, z from vision-language.
    vision_language_pose = detections[language_id]
    pose = math_helpers.SE3Pose(x=vision_language_pose.x,
                                y=vision_language_pose.y,
                                z=vision_language_pose.z,
                                rot=rot)
    return pose


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # NOTE: make sure the spot hand camera sees the 408 april tag, a brush,
    # and a drill. It is recommended to run this test a few times in a row
    # while moving the robot around, but keeping the objects in place.

    # pylint: disable=ungrouped-imports
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.util import authenticate

    from predicators import utils
    from predicators.spot_utils.perception.spot_cameras import capture_images
    from predicators.spot_utils.spot_localization import SpotLocalizer
    from predicators.spot_utils.utils import verify_estop

    TEST_CAMERAS = [
        "hand_color_image", "frontleft_fisheye_image", "left_fisheye_image",
        "right_fisheye_image"
    ]
    TEST_APRIL_TAG_ID = 408
    TEST_LANGUAGE_DESCRIPTIONS = ["brush", "drill"]

    def _run_pythonic_bowl_test() -> None:
        # Test for using an arbitrary python function to detect objects,
        # which in this case uses a combination of vision-language and
        # colored-based detection to find a bowl that has blue tape on the
        # bottom. The tape is used to crudely orient the bowl. Like the
        # previous test, this one assumes that the bowl is within view.
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)

        # Get constants.
        hostname = CFG.spot_robot_ip
        path = get_graph_nav_dir()

        # First, capture images.
        sdk = create_standard_sdk('SpotCameraTestClient')
        robot = sdk.create_robot(hostname)
        authenticate(robot)
        verify_estop(robot)
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        lease_keepalive = LeaseKeepAlive(lease_client,
                                         must_acquire=True,
                                         return_at_exit=True)

        assert path.exists()
        localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)
        rgbds = capture_images(robot, localizer)

        bowl_id = PythonicObjectDetectionID("bowl", detect_bowl)
        detections, artifacts = detect_objects([bowl_id], rgbds)
        for obj_id, detection in detections.items():
            print(f"Detected {obj_id} at {detection}")

        # Visualize the artifacts.
        detections_outfile = Path(".") / "object_detection_artifacts.png"
        no_detections_outfile = Path(".") / "no_detection_artifacts.png"
        visualize_all_artifacts(artifacts, detections_outfile,
                                no_detections_outfile)

    _run_pythonic_bowl_test()
