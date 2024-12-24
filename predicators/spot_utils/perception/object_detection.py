""""Interface for detecting objects with fiducials or pretrained models.

The fiducials are april tags. The size of the april tag is important and can be
configured via CFG.spot_fiducial_size.

The pretrained models are currently DETIC and SAM (used together). DETIC takes
a language description of an object (e.g., "brush") and an RGB image and finds
a bounding box. SAM segments objects within the bounding box (class-agnostic).
The associated depth image is then used to estimate the depth of the object
based on the median depth of segmented points. See the README in this directory
for instructions on setting up DETIC and SAM on a server.

Object detection returns SE3Poses in the world frame but only x, y, z positions
are currently detected. Rotations should be ignored.
"""

import io
import logging
from functools import partial
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Set, Tuple
from enum import Enum
import requests
from predicators.spot_utils.perception.molmo_sam2_client import MolmoSAM2Client, decode_rle_mask

try:
    import apriltag
    _APRILTAG_IMPORTED = True
except ModuleNotFoundError:
    _APRILTAG_IMPORTED = False
import cv2
import dill as pkl
import numpy as np
import PIL.Image
import requests
from bosdyn.client import math_helpers
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.spatial import Delaunay

from predicators.settings import CFG
from predicators.spot_utils.perception.object_specific_grasp_selection import \
    OBJECT_SPECIFIC_GRASP_SELECTORS
from predicators.spot_utils.perception.perception_structs import \
    AprilTagObjectDetectionID, KnownStaticObjectDetectionID, \
    LanguageObjectDetectionID, ObjectDetectionID, PythonicObjectDetectionID, \
    RGBDImageWithContext, SegmentedBoundingBox
from predicators.spot_utils.utils import get_april_tag_transform, \
    get_graph_nav_dir
from predicators.utils import rotate_point_in_image

# Hack to avoid double image capturing when we want to (1) get object states
# and then (2) use the image again for pixel-based grasping.
_LAST_DETECTED_OBJECTS: Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose],
                              Dict[str, Any]] = ({}, {})


def get_last_detected_objects(
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any]]:
    """Return the last output from detect_objects(), ignoring inputs."""
    return _LAST_DETECTED_OBJECTS


class ModelType(Enum):
    DETIC_SAM = "detic_sam"
    MOLMO_SAM2 = "molmo_sam2"


# Add configuration - can be moved to CFG if needed
# VISION_MODEL = ModelType.DETIC_SAM
VISION_MODEL = ModelType.MOLMO_SAM2


def _preprocess_images(rgbds: Dict[str, RGBDImageWithContext]) -> Dict:
    """Common preprocessing for both models."""
    buf_dict = {}
    for camera_name, rgbd in rgbds.items():
        pil_rotated_img = PIL.Image.fromarray(rgbd.rotated_rgb)
        buf_dict[camera_name] = _image_to_bytes(pil_rotated_img)
    return buf_dict

def _request_detic_sam(
    buf_dict: Dict,
    classes: List[str],
    max_retries: int = 5,
    detection_threshold: float = CFG.spot_vision_detection_threshold
) -> Optional[Dict]:
    """Make request to DETIC-SAM server."""
    for _ in range(max_retries):
        try:
            r = requests.post("http://localhost:5550/batch_predict",
                            files=buf_dict,
                            data={"classes": ",".join(classes), 
                                 "threshold": detection_threshold})
            break
        except requests.exceptions.ConnectionError:
            continue
    else:
        logging.warning("DETIC-SAM FAILED, POSSIBLE SERVER/WIFI ISSUE")
        return None

    if r.status_code != 200:
        logging.warning(f"DETIC-SAM FAILED! STATUS CODE: {r.status_code}")
        return None

    try:
        # Load the results immediately and return as dict
        with io.BytesIO(r.content) as f:
            server_results = np.load(f, allow_pickle=True)
            # Convert to dict to avoid file closure issues
            return {k: server_results[k] for k in server_results.files}
    except pkl.UnpicklingError:
        logging.warning("DETIC-SAM FAILED DURING UNPICKLING!")
        return None

def _process_detic_sam_results(
    server_results: Optional[Dict],
    object_ids: Collection[LanguageObjectDetectionID],
    rgbds: Dict[str, RGBDImageWithContext],
    detection_threshold: float = CFG.spot_vision_detection_threshold
) -> Dict[ObjectDetectionID, Dict[str, SegmentedBoundingBox]]:
    """Process DETIC-SAM results."""
    object_id_to_img_detections = {obj_id: {} for obj_id in object_ids}
    
    if server_results is None:
        return object_id_to_img_detections

    for camera_name, rgbd in rgbds.items():
        try:
            rot_boxes = server_results[f"{camera_name}_boxes"]
            ret_classes = server_results[f"{camera_name}_classes"]
            rot_masks = server_results[f"{camera_name}_masks"]
            scores = server_results[f"{camera_name}_scores"]
        except KeyError:
            logging.warning(f"Missing data for camera {camera_name}")
            continue

        # Invert rotation
        h, w = rgbd.rgb.shape[:2]
        image_rot = rgbd.image_rot
        boxes = [_rotate_bounding_box(bb, -image_rot, h, w) for bb in rot_boxes]
        masks = [ndimage.rotate(m.squeeze(), -image_rot, reshape=False) 
                for m in rot_masks]

        for obj_id in object_ids:
            if ret_classes.size == 0:
                continue
            obj_id_mask = (ret_classes == obj_id.language_id)
            if not np.any(obj_id_mask):
                continue
            max_score = np.max(scores[obj_id_mask])
            best_idx = np.where(scores == max_score)[0].item()
            if scores[best_idx] < detection_threshold:
                continue
            seg_bb = SegmentedBoundingBox(boxes[best_idx], masks[best_idx],
                                        scores[best_idx])
            object_id_to_img_detections[obj_id][rgbd.camera_name] = seg_bb

    return object_id_to_img_detections

def _request_molmo_sam2(
    images: List[PIL.Image.Image],
    prompts: List[str],
    max_retries: int = 5
) -> Dict:
    """Make request to Molmo-SAM2 server."""
    client = MolmoSAM2Client()
    for _ in range(max_retries):
        try:
            result = client.predict(images, prompts, render=False)
            return result
        except Exception as e:
            logging.warning(f"MOLMO-SAM2 request failed: {str(e)}")
            continue
    return None

def _process_molmo_sam2_results(
    result: Dict,
    object_ids: Collection[LanguageObjectDetectionID],
    rgbds: Dict[str, RGBDImageWithContext],
    camera_names: List[str]
) -> Dict[ObjectDetectionID, Dict[str, SegmentedBoundingBox]]:
    """Process Molmo-SAM2 results."""
    object_id_to_img_detections = {obj_id: {} for obj_id in object_ids}
    
    if result is None:
        logging.warning("Molmo-SAM2 returned None result")
        return object_id_to_img_detections
    if "results" not in result:
        logging.warning(f"No 'results' in Molmo-SAM2 response. Keys: {result.keys()}")
        return object_id_to_img_detections

    logging.info(f"Processing {len(result['results'])} Molmo-SAM2 results")
    
    for res_idx, res in enumerate(result["results"]):
        img_idx = res["image_index"]
        prompt_idx = res["prompt_index"]
        camera_name = camera_names[img_idx]
        obj_id = list(object_ids)[prompt_idx]

        logging.info(f"\nResult {res_idx + 1}:")
        logging.info(f"Camera: {camera_name}, Prompt: {obj_id.language_id}")

        boxes = res.get("boxes", [])
        masks = res.get("masks", [])

        logging.info(f"Found {len(boxes)} boxes and {len(masks)} masks")
        if boxes:
            logging.info(f"Box scores: {[box[4] for box in boxes]}")

        if not boxes or not masks:
            logging.warning(f"No detection for {obj_id.language_id} in {camera_name}")
            continue

        # Get best detection based on confidence score (box[4])
        scores = [box[4] for box in boxes]
        max_score_idx = np.argmax(scores)
        box = boxes[max_score_idx]
        score = box[4]

        logging.info(f"Best detection score: {score:.3f}")
        logging.info(f"Original box: {[round(x, 3) for x in box]}")

        if score < CFG.spot_vision_detection_threshold:
            logging.info(f"Score {score:.3f} below threshold {CFG.spot_vision_detection_threshold}")
            continue

        # Invert rotation
        rgbd = rgbds[camera_name]
        h, w = rgbd.rgb.shape[:2]
        image_rot = rgbd.image_rot
        box[:4] = _rotate_bounding_box(box[:4], -image_rot, h, w)
        logging.info(f"Rotated box: {[round(x, 3) for x in box[:4]]}")

        mask = masks[max_score_idx]
        if isinstance(mask, dict) and "counts" in mask:
            logging.info("Processing RLE mask")
            mask = decode_rle_mask(mask)
        elif isinstance(mask, list):
            logging.info("Processing list mask")
            mask = np.array(mask)
        else:
            logging.info(f"Mask type: {type(mask)}")
        
        if mask is not None:
            logging.info(f"Mask shape before rotation: {mask.shape}")
            mask = ndimage.rotate(mask.squeeze(), -image_rot, reshape=False)
            logging.info(f"Mask shape after rotation: {mask.shape}")
            logging.info(f"Mask values: min={mask.min():.3f}, max={mask.max():.3f}, mean={mask.mean():.3f}")

        seg_bb = SegmentedBoundingBox(
            bounding_box=box[:4],
            mask=mask,
            score=score
        )
        object_id_to_img_detections[obj_id][camera_name] = seg_bb
        logging.info(f"Successfully added detection for {obj_id.language_id} in {camera_name}")

    logging.info(f"\nTotal detections: {sum(len(v) for v in object_id_to_img_detections.values())}")
    return object_id_to_img_detections

def detect_objects_from_language(
    object_ids: Collection[LanguageObjectDetectionID],
    rgbds: Dict[str, RGBDImageWithContext],
    allowed_regions: Optional[Collection[Delaunay]] = None,
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict]:
    """Detect an object pose using a vision-language model."""
    
    # Common preprocessing
    if VISION_MODEL == ModelType.MOLMO_SAM2:
        images = []
        camera_names = []
        for camera_name, rgbd in rgbds.items():
            pil_rotated_img = PIL.Image.fromarray(rgbd.rotated_rgb)
            images.append(pil_rotated_img)
            camera_names.append(camera_name)
        prompts = sorted(o.language_id for o in object_ids)
        result = _request_molmo_sam2(images, prompts)
        object_id_to_img_detections = _process_molmo_sam2_results(
            result, object_ids, rgbds, camera_names)
    else:
        # DETIC-SAM processing
        buf_dict = _preprocess_images(rgbds)
        classes = sorted(o.language_id for o in object_ids)
        server_results = _request_detic_sam(buf_dict, classes)
        object_id_to_img_detections = _process_detic_sam_results(
            server_results, object_ids, rgbds)

    # Convert detections to poses
    detections: Dict[ObjectDetectionID, math_helpers.SE3Pose] = {}
    for obj_id, img_detections in object_id_to_img_detections.items():
        # Consider detections from best (highest) to worst score.
        best_camera = None
        best_score = -float("inf")
        for camera, seg_bb in img_detections.items():
            if seg_bb.score > best_score:
                best_score = seg_bb.score
                best_camera = camera

        if best_camera is None:
            continue

        # Get pose from best detection
        rgbd = rgbds[best_camera]
        seg_bb = img_detections[best_camera]
        pose = _get_pose_from_segmented_bounding_box(seg_bb, rgbd)
        if pose is None:
            continue

        # If the detected pose is outside the allowed bounds, skip
        if allowed_regions is not None:
            pose_xy = np.array([pose.x, pose.y])
            in_allowed_region = False
            for region in allowed_regions:
                if region.find_simplex(pose_xy).item() >= 0:
                    in_allowed_region = True
                    break
            if not in_allowed_region:
                logging.info(f"WARNING: throwing away detection for {obj_id} " + \
                           f"because it's out of bounds. (pose = {pose_xy})")
                continue

        detections[obj_id] = pose

    artifacts = {
        "rgbds": rgbds,
        "object_id_to_img_detections": object_id_to_img_detections
    }

    return detections, artifacts


def detect_objects(
    object_ids: Collection[ObjectDetectionID],
    rgbds: Dict[str, RGBDImageWithContext],  # camera name to RGBD
    allowed_regions: Optional[Collection[Delaunay]] = None,
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any]]:
    """Detect object poses (in the world frame!) from RGBD.

    Each object ID is assumed to exist at most once in each image, but can
    exist in multiple images.

    The second return value is a collection of artifacts that can be useful
    for debugging / analysis.
    """
    global _LAST_DETECTED_OBJECTS  # pylint: disable=global-statement

    # Collect and dispatch.
    april_tag_object_ids: Set[AprilTagObjectDetectionID] = set()
    language_object_ids: Set[LanguageObjectDetectionID] = set()
    pythonic_object_ids: Set[PythonicObjectDetectionID] = set()
    known_static_object_ids: Set[KnownStaticObjectDetectionID] = set()
    for object_id in object_ids:
        if isinstance(object_id, AprilTagObjectDetectionID):
            april_tag_object_ids.add(object_id)
        elif isinstance(object_id, KnownStaticObjectDetectionID):
            known_static_object_ids.add(object_id)
        elif isinstance(object_id, PythonicObjectDetectionID):
            pythonic_object_ids.add(object_id)
        else:
            assert isinstance(object_id, LanguageObjectDetectionID)
            language_object_ids.add(object_id)
    detections: Dict[ObjectDetectionID, math_helpers.SE3Pose] = {}
    artifacts: Dict[str, Any] = {"april": {}, "language": {}}

    # Read off known objects directly.
    for known_obj_id in known_static_object_ids:
        detections[known_obj_id] = known_obj_id.pose

    # There is no batching over images for april tag detection.
    for rgbd in rgbds.values():
        img_detections, img_artifacts = detect_objects_from_april_tags(
            april_tag_object_ids, rgbd)
        # Possibly overrides previous detections.
        detections.update(img_detections)
        artifacts["april"][rgbd.camera_name] = img_artifacts

    # There IS batching over images here for efficiency.
    language_detections, language_artifacts = detect_objects_from_language(
        language_object_ids, rgbds, allowed_regions)
    detections.update(language_detections)
    artifacts["language"] = language_artifacts

    # Handle pythonic object detection.
    for object_id in pythonic_object_ids:
        detection = object_id.fn(rgbds)
        if detection is not None:
            detections[object_id] = detection
            break

    _LAST_DETECTED_OBJECTS = (detections, artifacts)

    return detections, artifacts


def detect_objects_from_april_tags(
    object_ids: Collection[AprilTagObjectDetectionID],
    rgbd: RGBDImageWithContext,
    fiducial_size: float = CFG.spot_fiducial_size,
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict]:
    """Detect an object pose from an april tag.

    The rotation is currently not detected (set to default).

    The second return value is a dictionary of "artifacts", which include
    the raw april tag detection results. These are primarily useful for
    debugging / analysis.
    """
    if not object_ids:
        return {}, {}

    if not _APRILTAG_IMPORTED:
        raise ModuleNotFoundError("Need to install 'apriltag' package")

    tag_num_to_object_id = {t.april_tag_number: t for t in object_ids}

    # Convert the RGB image to grayscale.
    image_grey = cv2.cvtColor(rgbd.rgb, cv2.COLOR_RGB2GRAY)

    # Create apriltag detector and get all apriltag locations.
    options = apriltag.DetectorOptions(families="tag36h11")
    options.refine_pose = 1
    detector = apriltag.Detector(options)
    apriltag_detections = detector.detect(image_grey)

    detections: Dict[ObjectDetectionID, math_helpers.SE3Pose] = {}
    artifacts: Dict = {}

    # For every detection, find pose in world frame.
    for apriltag_detection in apriltag_detections:
        # Only include requested tags.
        if apriltag_detection.tag_id not in tag_num_to_object_id:
            continue
        obj_id = tag_num_to_object_id[apriltag_detection.tag_id]

        # Save the detection for external analysis.
        artifacts[obj_id] = apriltag_detection

        # Get the pose from the apriltag library.
        intrinsics = rgbd.camera_model.intrinsics
        pose = detector.detection_pose(
            apriltag_detection,
            (intrinsics.focal_length.x, intrinsics.focal_length.y,
             intrinsics.principal_point.x, intrinsics.principal_point.y),
            fiducial_size)[0]
        tx, ty, tz, tw = pose[:, -1]
        assert np.isclose(tw, 1.0)

        # Detection is in meters, we want mm.
        camera_tform_tag = math_helpers.SE3Pose(
            x=float(tx) / 1000.0,
            y=float(ty) / 1000.0,
            z=float(tz) / 1000.0,
            rot=math_helpers.Quat(),
        )

        # Look up transform.
        world_object_tform_tag = get_april_tag_transform(
            obj_id.april_tag_number)

        # Apply transforms.
        world_frame_pose = rgbd.world_tform_camera * camera_tform_tag
        world_frame_pose = world_object_tform_tag * world_frame_pose

        # Save in detections.
        detections[obj_id] = world_frame_pose

    return detections, artifacts


def _image_to_bytes(img: PIL.Image.Image) -> io.BytesIO:
    """Helper function to convert from a PIL image into a bytes object."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _rotate_bounding_box(bb: Tuple[float, float, float,
                                   float], rot_degrees: float, height: int,
                         width: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bb
    ry1, rx1 = rotate_point_in_image(y1, x1, rot_degrees, height, width)
    ry2, rx2 = rotate_point_in_image(y2, x2, rot_degrees, height, width)
    return (rx1, ry1, rx2, ry2)


def _get_pose_from_segmented_bounding_box(
        seg_bb: SegmentedBoundingBox,
        rgbd: RGBDImageWithContext,
        min_depth_value: float = 2) -> Optional[math_helpers.SE3Pose]:
    """Returns None if the depth of the object cannot be estimated.

    The known case where this happens is when the robot's hand occludes
    the depth camera (which is physically above the RGB camera).
    """
    # Get the center of the bounding box.
    x1, y1, x2, y2 = seg_bb.bounding_box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    # Get the median depth value of segmented points.
    # Filter 0 points out of the depth map.
    seg_mask = seg_bb.mask & (rgbd.depth > min_depth_value)
    segmented_depth = rgbd.depth[seg_mask]
    # See docstring.
    if len(segmented_depth) == 0:
        # logging.warning doesn't work here because of poor spot logging.
        print("WARNING: depth reading failed. Is hand occluding?")
        return None
    depth_value = np.median(segmented_depth)

    # Convert to camera frame position.
    fx = rgbd.camera_model.intrinsics.focal_length.x
    fy = rgbd.camera_model.intrinsics.focal_length.y
    cx = rgbd.camera_model.intrinsics.principal_point.x
    cy = rgbd.camera_model.intrinsics.principal_point.y
    depth_scale = rgbd.depth_scale
    camera_z = depth_value / depth_scale
    camera_x = np.multiply(camera_z, (x_center - cx)) / fx
    camera_y = np.multiply(camera_z, (y_center - cy)) / fy
    camera_frame_pose = math_helpers.SE3Pose(x=camera_x,
                                             y=camera_y,
                                             z=camera_z,
                                             rot=math_helpers.Quat())

    # Convert camera to world.
    world_frame_pose = rgbd.world_tform_camera * camera_frame_pose

    # The angles are not meaningful, so override them.
    final_pose = math_helpers.SE3Pose(x=world_frame_pose.x,
                                      y=world_frame_pose.y,
                                      z=world_frame_pose.z,
                                      rot=math_helpers.Quat())

    return final_pose


def get_grasp_pixel(
    rgbds: Dict[str, RGBDImageWithContext], artifacts: Dict[str, Any],
    object_id: ObjectDetectionID, camera_name: str, rng: np.random.Generator
) -> Tuple[Tuple[int, int], Optional[math_helpers.Quat]]:
    """Select a pixel for grasping in the given camera image.

    NOTE: for april tag detections, the pixel returned will correspond to the
    center of the april tag, which may not always be ideal for grasping.
    Consider using OBJECT_SPECIFIC_GRASP_SELECTORS in this case.
    """

    if object_id in OBJECT_SPECIFIC_GRASP_SELECTORS:
        selector = OBJECT_SPECIFIC_GRASP_SELECTORS[object_id]
        return selector(rgbds, artifacts, camera_name, rng)

    pixel = get_random_mask_pixel_from_artifacts(artifacts, object_id,
                                                 camera_name, rng)
    return (pixel[0], pixel[1]), None


def get_random_mask_pixel_from_artifacts(
        artifacts: Dict[str, Any], object_id: ObjectDetectionID,
        camera_name: str, rng: np.random.Generator) -> Tuple[int, int]:
    """Extract the pixel in the image corresponding to the center of the object
    with object ID.

    The typical use case is to get the pixel to pass into the grasp
    controller. This is a fairly hacky way to go about this, but since
    the grasp controller parameterization is a special case (most users
    of object detection shouldn't need to care about the pixel), we do
    this.
    """
    if isinstance(object_id, AprilTagObjectDetectionID):
        try:
            april_detection = artifacts["april"][camera_name][object_id]
        except KeyError:
            raise ValueError(f"{object_id} not detected in {camera_name}")
        pr, pc = april_detection.center
        return int(pr), int(pc)

    assert isinstance(object_id, LanguageObjectDetectionID)
    detections = artifacts["language"]["object_id_to_img_detections"]
    try:
        seg_bb = detections[object_id][camera_name]
    except KeyError:
        raise ValueError(f"{object_id} not detected in {camera_name}")

    # Select a random valid pixel from the mask.
    mask = seg_bb.mask
    pixels_in_mask = np.where(mask)
    mask_idx = rng.choice(len(pixels_in_mask))
    pixel_tuple = (pixels_in_mask[1][mask_idx], pixels_in_mask[0][mask_idx])
    # Uncomment to plot the grasp pixel being selected!
    """
    rgb_img = artifacts["language"]["rgbds"][camera_name].rgb
    _, axes = plt.subplots()
    axes.imshow(rgb_img)
    axes.add_patch(
        plt.Rectangle((pixel_tuple[0], pixel_tuple[1]), 5, 5, color='red'))
    plt.tight_layout()
    outdir = Path(CFG.spot_perception_outdir)
    plt.savefig(outdir / "grasp_pixel.png", dpi=300)
    plt.close()
    """
    return pixel_tuple


def visualize_all_artifacts(artifacts: Dict[str,
                                            Any], detections_outfile: Path,
                            no_detections_outfile: Path) -> None:
    """Analyze the artifacts."""
    # At the moment, only language detection artifacts are visualized.
    rgbds = artifacts["language"]["rgbds"]
    detections = artifacts["language"]["object_id_to_img_detections"]
    flat_detections: List[Tuple[RGBDImageWithContext,
                                LanguageObjectDetectionID,
                                SegmentedBoundingBox]] = []
    for obj_id, img_detections in detections.items():
        for camera, seg_bb in img_detections.items():
            rgbd = rgbds[camera]
            flat_detections.append((rgbd, obj_id, seg_bb))

    # Visualize in subplots where columns are: rotated RGB, original RGB,
    # original depth, bounding box, mask. Each row is one detection, so if
    # there are multiple detections in a single image, then there will be
    # duplicate first cols.
    fig_scale = 2
    if flat_detections:
        _, axes = plt.subplots(len(flat_detections),
                               5,
                               squeeze=False,
                               figsize=(5 * fig_scale,
                                        len(flat_detections) * fig_scale))
        plt.suptitle("Detections")
        for i, (rgbd, obj_id, seg_bb) in enumerate(flat_detections):
            ax_row = axes[i]
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
            ax_row[0].imshow(rgbd.rotated_rgb)
            ax_row[1].imshow(rgbd.rgb)
            ax_row[2].imshow(rgbd.depth, cmap='Greys_r', vmin=0, vmax=10000)

            # Bounding box.
            ax_row[3].imshow(rgbd.rgb)
            box = seg_bb.bounding_box
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax_row[3].add_patch(
                plt.Rectangle((x0, y0),
                              w,
                              h,
                              edgecolor='green',
                              facecolor=(0, 0, 0, 0),
                              lw=1))

            ax_row[4].imshow(seg_bb.mask, cmap="binary_r", vmin=0, vmax=1)

            # Labels.
            abbreviated_name = obj_id.language_id
            max_abbrev_len = 24
            if len(abbreviated_name) > max_abbrev_len:
                abbreviated_name = abbreviated_name[:max_abbrev_len] + "..."
            row_label = "\n".join([
                abbreviated_name, f"[{rgbd.camera_name}]",
                f"[score: {seg_bb.score:.2f}]"
            ])
            ax_row[0].set_ylabel(row_label, fontsize=6)
            if i == len(flat_detections) - 1:
                ax_row[0].set_xlabel("Rotated RGB")
                ax_row[1].set_xlabel("Original RGB")
                ax_row[2].set_xlabel("Original Depth")
                ax_row[3].set_xlabel("Bounding Box")
                ax_row[4].set_xlabel("Mask")

        plt.tight_layout()
        plt.savefig(detections_outfile, dpi=300)
        print(f"Wrote out to {detections_outfile}.")
        plt.close()

    # Visualize all of the images that have no detections.
    all_cameras = set(rgbds)
    cameras_with_detections = {r.camera_name for r, _, _ in flat_detections}
    cameras_without_detections = sorted(all_cameras - cameras_with_detections)

    if cameras_without_detections:
        _, axes = plt.subplots(len(cameras_without_detections),
                               3,
                               squeeze=False,
                               figsize=(3 * fig_scale,
                                        len(cameras_without_detections) *
                                        fig_scale))
        plt.suptitle("Cameras without Detections")
        for i, camera in enumerate(cameras_without_detections):
            rgbd = rgbds[camera]
            ax_row = axes[i]
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
            ax_row[0].imshow(rgbd.rotated_rgb)
            ax_row[1].imshow(rgbd.rgb)
            ax_row[2].imshow(rgbd.depth, cmap='Greys_r', vmin=0, vmax=10000)

            # Labels.
            ax_row[0].set_ylabel(f"[{rgbd.camera_name}]", fontsize=6)
            if i == len(flat_detections) - 1:
                ax_row[0].set_xlabel("Rotated RGB")
                ax_row[1].set_xlabel("Original RGB")
                ax_row[2].set_xlabel("Original Depth")

        plt.tight_layout()
        plt.savefig(no_detections_outfile, dpi=300)
        print(f"Wrote out to {no_detections_outfile}.")
        plt.close()


def display_camera_detections(artifacts: Dict[str, Any],
                              axes: plt.Axes) -> None:
    """Plot per-camera detections on the given axes.

    The axes are given as input because we might want to update the same
    axes repeatedly, e.g., during object search.
    """

    # At the moment, only language detection artifacts are visualized.
    rgbds = artifacts["language"]["rgbds"]
    detections = artifacts["language"]["object_id_to_img_detections"]
    # Organize detections by camera.
    camera_to_rgbd = {rgbd.camera_name: rgbd for rgbd in rgbds.values()}
    camera_to_detections: Dict[str, List[Tuple[LanguageObjectDetectionID,
                                               SegmentedBoundingBox]]] = {
                                                   c: []
                                                   for c in camera_to_rgbd
                                               }
    for obj_id, img_detections in detections.items():
        for camera, seg_bb in img_detections.items():
            camera_to_detections[camera].append((obj_id, seg_bb))

    # Plot per-camera.
    box_colors = ["green", "red", "blue", "purple", "gold", "brown", "black"]
    camera_order = sorted(camera_to_rgbd)
    for ax, camera in zip(axes.flat, camera_order):
        ax.clear()
        ax.set_title(camera)
        ax.set_xticks([])
        ax.set_yticks([])

        # Display the RGB image.
        rgbd = camera_to_rgbd[camera]
        ax.imshow(rgbd.rotated_rgb)

        for i, (obj_id, seg_bb) in enumerate(camera_to_detections[camera]):

            color = box_colors[i % len(box_colors)]

            # Display the bounding box.
            box = seg_bb.bounding_box
            # Rotate.
            image_rot = rgbd.image_rot
            h, w = rgbd.rgb.shape[:2]
            box = _rotate_bounding_box(box, image_rot, h, w)
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax.add_patch(
                plt.Rectangle((x0, y0),
                              w,
                              h,
                              edgecolor=color,
                              facecolor=(0, 0, 0, 0),
                              lw=1))
            # Label with the detection and score.
            ax.text(
                -250,  # off to the left side
                50 + 60 * i,
                f'{obj_id.language_id}: {seg_bb.score:.2f}',
                color='white',
                fontsize=12,
                fontweight='bold',
                bbox=dict(facecolor=color, edgecolor=color, alpha=0.5))


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
    from predicators.spot_utils.perception.cv2_utils import \
        find_color_based_centroid
    from predicators.spot_utils.perception.spot_cameras import capture_images
    from predicators.spot_utils.spot_localization import SpotLocalizer
    from predicators.spot_utils.utils import verify_estop

    TEST_CAMERAS = [
        "hand_color_image",
        "frontleft_fisheye_image",
        "left_fisheye_image",
        "right_fisheye_image",
        "frontright_fisheye_image",
    ]
    TEST_APRIL_TAG_ID = 408
    TEST_LANGUAGE_DESCRIPTIONS = [
        # "small basketball toy/stuffed toy basketball/small orange ball",
        # "small football toy/stuffed toy football/small brown ball",
        # "blue block",
        # "blue cup",
        "red cup",
        "green cup",
        "yellow tape",
        # "point at the blue block",
        # "point at the blue cup",
    ]

    def _run_manual_test() -> None:
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
        rgbds = capture_images(robot, localizer, TEST_CAMERAS)

        # Detect the april tag and brush.
        april_tag_id: ObjectDetectionID = AprilTagObjectDetectionID(
            TEST_APRIL_TAG_ID)
        language_ids: List[ObjectDetectionID] = [
            LanguageObjectDetectionID(d) for d in TEST_LANGUAGE_DESCRIPTIONS
        ]
        known_static_id: ObjectDetectionID = KnownStaticObjectDetectionID(
            "imaginary_box",
            math_helpers.SE3Pose(-5, 0, 0, rot=math_helpers.Quat()))
        object_ids: List[ObjectDetectionID] = [april_tag_id, known_static_id
                                               ] + language_ids
        detections, artifacts = detect_objects(object_ids, rgbds)
        for obj_id, detection in detections.items():
            print(f"Detected {obj_id} at {detection}")

        # Visualize the artifacts.
        detections_outfile = Path(".") / "object_detection_artifacts.png"
        no_detections_outfile = Path(".") / "no_detection_artifacts.png"
        visualize_all_artifacts(artifacts, detections_outfile,
                                no_detections_outfile)

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

        def _detect_bowl(
            rgbds: Dict[str, RGBDImageWithContext]
        ) -> Optional[math_helpers.SE3Pose]:
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
            lo, hi = ((0, 130, 130), (130, 255, 255))
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

        bowl_id = PythonicObjectDetectionID("bowl", _detect_bowl)
        detections, artifacts = detect_objects([bowl_id], rgbds)
        for obj_id, detection in detections.items():
            print(f"Detected {obj_id} at {detection}")

        # Visualize the artifacts.
        detections_outfile = Path(".") / "object_detection_artifacts.png"
        no_detections_outfile = Path(".") / "no_detection_artifacts.png"
        visualize_all_artifacts(artifacts, detections_outfile,
                                no_detections_outfile)

    _run_manual_test()
    # _run_pythonic_bowl_test()
