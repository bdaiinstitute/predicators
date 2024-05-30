# MOVE TO LOCATION
# X,Y in map, Theta is direction of robot (in radians)
move_hand_to_relative_pose(robot, se3_pose(x, y, z, quaternion)) # x is forward from robot, y is right from robot, and z is up from robot
get_grasp_pixel(robot, ) -> returns pixel to grasp 
grasp_at_pixel(robot, rgbds[hand_camera], pixel) -> grasps at pixel

# MOVE HAND TO RELATIVE POSE
target_pos = math_helpers.Vec3(0.8, 0.0, 0.25)
downward_angle = np.pi / 2.5
target_pos = math_helpers.Vec3(0.8, 0.0, 0.25)
rot = math_helpers.Quat.from_pitch(downward_angle)
body_tform_goal = math_helpers.SE3Pose(x=target_pos.x, y=target_pos.y, z=target_pos.z, rot=rot)
move_hand_to_relative_pose(robot, body_tform_goal)
# move_hand_to_relative_pose(robot, math_helpers.SE3Pose(0.753, 0.018, 0.89, math_helpers.Quat(0.6870, 0.0664, 0.7200, -0.0722)))
# time.sleep(0.5)

# GRASP

# GET PIXEL FROM USER
pixel = get_pixel_from_user(rgbds[hand_camera].rgb)

# GET PIXEL FROM OBJECT DETECTION
rgbds = capture_images(robot, localizer, TEST_CAMERAS)
language_ids: List[ObjectDetectionID] = [
    LanguageObjectDetectionID(d) for d in TEST_LANGUAGE_DESCRIPTIONS
]
hand_camera = "hand_color_image"
detections, artifacts = detect_objects(language_ids, rgbds)

detections_outfile = Path(".") / "object_detection_artifacts.png"
no_detections_outfile = Path(".") / "no_detection_artifacts.png"
visualize_all_artifacts(artifacts, detections_outfile,
                        no_detections_outfile)

pixel, _ = get_grasp_pixel(rgbds, artifacts, language_ids[-1],
                                       hand_camera, rng)

grasp_at_pixel(robot, rgbds[hand_camera], pixel)
stow_arm(robot)
time.sleep(0.5)