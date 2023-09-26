"""Integration tests for spot utilities.

Run with --spot_robot_ip and any other flags.
"""
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import dill as pkl

import numpy as np
from bosdyn.client import create_standard_sdk, math_helpers
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.sdk import Robot
from bosdyn.client.util import authenticate

from predicators import utils
from predicators.settings import CFG
from predicators.spot_utils.perception.object_detection import \
    AprilTagObjectDetectionID, LanguageObjectDetectionID, detect_objects, \
    get_object_center_pixel_from_artifacts, _visualize_all_artifacts
from predicators.spot_utils.perception.perception_structs import \
    AprilTagObjectDetectionID, LanguageObjectDetectionID, ObjectDetectionID, \
    RGBDImageWithContext, SegmentedBoundingBox
from predicators.spot_utils.perception.spot_cameras import capture_images
from predicators.spot_utils.skills.spot_find_objects import find_objects
from predicators.spot_utils.skills.spot_grasp import grasp_at_pixel
from predicators.spot_utils.skills.spot_hand_move import \
    move_hand_to_relative_pose, open_gripper
from predicators.spot_utils.skills.spot_navigation import go_home, \
    navigate_to_relative_pose
from predicators.spot_utils.skills.spot_place import place_at_relative_position
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.utils import DEFAULT_HAND_LOOK_DOWN_POSE, \
    get_relative_se2_from_se3, verify_estop
from predicators.structs import Array
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


def move_pick_place(
    robot: Robot,
    localizer: SpotLocalizer,
    manipuland_id: ObjectDetectionID,
    init_surface_id: Optional[ObjectDetectionID],
    target_surface_id: ObjectDetectionID,
    pre_pick_surface_nav_distance: float = 1.25,
    pre_pick_floor_nav_distance: float = 1.75,
    pre_place_nav_distance: float = 0.45,
    pre_pick_nav_angle: float = -np.pi / 2,
    pre_place_nav_angle: float = -np.pi / 2,
    place_offset_z: float = 0.25,
) -> None:
    """Find the given object and surfaces, pick the object from the first
    surface, and place it on the second surface.

    The surface nav parameters determine where the robot should navigate
    with respect to the surfaces when picking and placing. The
    intelligence for choosing these offsets is external to the skills
    (e.g., they might be sampled).
    """
    go_home(robot, localizer)
    localizer.localize()

    # Find objects.
    object_ids = [manipuland_id]
    if init_surface_id is not None:
        object_ids.append(init_surface_id)
    object_ids.append(target_surface_id)
    detections, _ = find_objects(robot, localizer, object_ids)

    # Get current robot pose.
    robot_pose = localizer.get_last_robot_pose()
    if init_surface_id is not None:
        # Navigate to the first surface.
        rel_pose = get_relative_se2_from_se3(robot_pose,
                                             detections[init_surface_id],
                                             pre_pick_surface_nav_distance,
                                             pre_pick_nav_angle)
        navigate_to_relative_pose(robot, rel_pose)
        localizer.localize()
    else:
        # In this case, we assume the object is on the floor.
        rel_pose = get_relative_se2_from_se3(robot_pose,
                                             detections[manipuland_id],
                                             pre_pick_floor_nav_distance,
                                             pre_pick_nav_angle)
        navigate_to_relative_pose(robot, rel_pose)
        localizer.localize()

    # Look down at the surface.
    move_hand_to_relative_pose(robot, DEFAULT_HAND_LOOK_DOWN_POSE)
    open_gripper(robot)

    # Capture an image from the hand camera.
    hand_camera = "hand_color_image"
    rgbds = capture_images(robot, localizer, [hand_camera])

    # Run detection to get a pixel for grasping.
    obj_detection_dict, artifacts = detect_objects([manipuland_id], rgbds)
    # _visualize_all_artifacts(artifacts)

    # NOTE: we currently get the oracle pixel, but we will replace this call
    # soon with Andi's NN.
    pixel = get_object_center_pixel_from_artifacts(artifacts, manipuland_id,
                                                   hand_camera)

    # Pick at the pixel with a top-down grasp.
    grasp_at_pixel(robot, rgbds[hand_camera], pixel)
    localizer.localize()

    # Stow the arm.
    stow_arm(robot)

    # Navigate to the other surface.
    robot_pose = localizer.get_last_robot_pose()
    rel_pose = get_relative_se2_from_se3(robot_pose,
                                         detections[target_surface_id],
                                         pre_place_nav_distance,
                                         pre_place_nav_angle)
    navigate_to_relative_pose(robot, rel_pose)
    localizer.localize()

    # Place on the surface.
    robot_pose = localizer.get_last_robot_pose()
    surface_rel_pose = robot_pose.inverse() * detections[target_surface_id]
    place_rel_pos = math_helpers.Vec3(x=surface_rel_pose.x,
                                      y=surface_rel_pose.y,
                                      z=surface_rel_pose.z + place_offset_z)
    place_at_relative_position(robot, place_rel_pos)

    # Finish by stowing arm again.
    stow_arm(robot)


def _get_training_datapoint(artifacts: Dict[str, Any]) -> Tuple[Array, Array, List[Tuple[str, Array, Tuple[int, int]]]]:
    """Output both the original rgb image and an image with all the bounding
    boxes outlined in green."""
    # At the moment, only language detection artifacts are visualized.
    fig = plt.figure()
    ax = fig.gca()
    rgbds = artifacts["language"]["rgbds"]
    detections = artifacts["language"]["object_id_to_img_detections"]
    flat_detections: List[Tuple[RGBDImageWithContext,
                                LanguageObjectDetectionID,
                                SegmentedBoundingBox]] = []
    for obj_id, img_detections in detections.items():
        for camera, seg_bb in img_detections.items():
            rgbd = rgbds[camera]
            flat_detections.append((rgbd, obj_id, seg_bb))

    if flat_detections:
        # First, put all the bounding boxes together.
        bboxes = []
        for i, (rgbd, obj_id, seg_bb) in enumerate(flat_detections):
            bboxes.append(seg_bb.bounding_box)

        # Bounding box.
        rgb_np_arr = rgbd.rgb
        ax.imshow(rgbd.rgb)
        for box in bboxes:
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax.add_patch(
                plt.Rectangle((x0, y0),
                            w,
                            h,
                            edgecolor='green',
                            facecolor=(0, 0, 0, 0),
                            lw=1))
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        bbox_np_arr = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        lang_mask_center_list = []
        for i, (rgbd, obj_id, seg_bb) in enumerate(flat_detections):
            lang = obj_id.language_id
            mask = seg_bb.mask
            center = get_object_center_pixel_from_artifacts(artifacts, obj_id, "hand_color_image")
            lang_mask_center_list.append((lang, mask, center))

    return (rgb_np_arr, bbox_np_arr, lang_mask_center_list)



def generate_single_datapoint(robot, localizer, object_detection_ids) -> None:
    """Generates a single datapoint and saves a pickle of it"""
    # Capture an image from the hand camera.
    hand_camera = "hand_color_image"
    rgbds = capture_images(robot, localizer, [hand_camera])

    # Run detection to get a pixel for grasping.
    detections_outfile = Path(".") / "object_detection_artifacts.png"
    no_detections_outfile = Path(".") / "no_detection_artifacts.png"
    _, artifacts = detect_objects(object_detection_ids, rgbds)
    _visualize_all_artifacts(artifacts, detections_outfile, no_detections_outfile)
    training_tuple = _get_training_datapoint(artifacts)
    data_file_path = Path(".") / "sample_data.pkl"
    with open(data_file_path, "wb") as f:
        pkl.dump(training_tuple, f)
    

if __name__ == "__main__":
    # Create detection id's for the different objects we're interested
    # in manipulating.
    apple = LanguageObjectDetectionID("apple")
    orange = LanguageObjectDetectionID("orange")
    water_bottle = LanguageObjectDetectionID("water bottle")

    # Parse flags.
    args = utils.parse_args(env_required=False,
                            seed_required=False,
                            approach_required=False)
    utils.update_config(args)
    # Set up the robot and localizer.
    hostname = CFG.spot_robot_ip
    upload_dir = Path(__file__).parent / "graph_nav_maps"
    path = upload_dir / CFG.spot_graph_nav_map
    sdk = create_standard_sdk("TestClient")
    robot = sdk.create_robot(hostname)
    authenticate(robot)
    verify_estop(robot)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    lease_keepalive = LeaseKeepAlive(lease_client,
                                     must_acquire=True,
                                     return_at_exit=True)
    assert path.exists()
    # localizer = None
    localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)

    # generate_single_datapoint(robot, localizer, [apple, orange, water_bottle])

    # Call a simple pick-place-move sequence for the orange.
    init_surface = AprilTagObjectDetectionID(
        408, math_helpers.SE3Pose(-0.25, 0.0, 0.0, math_helpers.Quat()))
    target_surface = AprilTagObjectDetectionID(
        409, math_helpers.SE3Pose(0.25, 0.0, 0.0, math_helpers.Quat()))
    move_pick_place(robot, localizer, orange, init_surface, target_surface)

    # data_file_path = Path(".") / "sample_data.pkl"
    # with open(data_file_path, "rb") as f:
    #     data = pkl.load(f)
    #     import ipdb; ipdb.set_trace()
