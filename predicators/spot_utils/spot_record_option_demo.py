"""Script to make it easy to record demonstrations from the Spot."""

from predicators.perception.spot_perceiver import save_annotated_imgs_for_vlm_demo, annotate_imgs_with_detections
from predicators import utils
from predicators.settings import CFG
from predicators.spot_utils.utils import verify_estop
from predicators.spot_utils.perception.spot_cameras import capture_images_without_context
from pathlib import Path

def main():
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # pylint: disable=ungrouped-imports
    import numpy as np
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient
    from bosdyn.client.util import authenticate

    # Put inside a function to avoid variable scoping issues.
    args = utils.parse_args(env_required=False,
                            seed_required=False,
                            approach_required=False)
    utils.update_config(args)

    # Get constants.
    hostname = CFG.spot_robot_ip

    # Instantiate a robot.
    sdk = create_standard_sdk('MoveHandSkillTestClient')
    robot = sdk.create_robot(hostname)
    authenticate(robot)
    verify_estop(robot)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()

    assert len(CFG.spot_vlm_teleop_demo_folderpath) > 0, "Please set the spot_vlm_teleop_demo_folderpath!"

    # Pull all the images from the spot cameras and annotate them
    # with the camera names.
    # NOTE: we currently don't run any object detection, though we could.
    rgbd_images = capture_images_without_context(robot)
    annotated_imgs = annotate_imgs_with_detections(rgbd_images, {})
    # Save the images properly.
    save_annotated_imgs_for_vlm_demo(annotated_imgs, Path(CFG.spot_vlm_teleop_demo_folderpath))


if __name__ == '__main__':
    main()
