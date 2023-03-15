"""Utility functions to interface with the Boston Dynamics Spot robot."""

import sys
import time
from typing import Any, Sequence

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
import cv2
import numpy as np
from bosdyn.api import arm_command_pb2, basic_command_pb2, estop_pb2, \
    geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, \
    ODOM_FRAME_NAME, VISION_FRAME_NAME, get_a_tform_b, get_vision_tform_body, \
    math_helpers
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandBuilder, \
    RobotCommandClient, block_until_arm_arrives, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.sdk import Robot

from predicators.settings import CFG
from predicators.spot_utils.helpers.graph_nav_command_line import \
    GraphNavInterface
from predicators.structs import Object

import torch

VELOCITY_CMD_DURATION = 1 # sec
VELOCITY_BASE_SPEED = 0.5 # m/s

g_image_click = None
g_image_display = None
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

graph_nav_loc_to_id = {
    "microwave_0": "sudden-buck-mZtfRKlks3p+yJjac4NenQ==",
    "kitchen_counter_0": "spoken-sawfly-sv+CrbBRCfenw7QX5dvrPQ==",
    "compost_bin_0": "punic-cicada-65HnkpdVrQtpteI4nG.qUw==",
    "dishwasher_0": "urbane-otter-bP2FTCyWk.bKjBub.WwShA==",
    "sink_0": "titled-rabbit-2wr0E8RQx5YH7lNsO9zD+Q==",
    "fridge_0": "blabby-howler-sukc2NseJ9SYW0zkSR4NcQ==",
    "kitchen_counter_1": "hunted-equid-AjeBjxUoRRzymDHNNakkgA==",
    "room_0_outside_0": "germy-fly-WtbLPt6S3+9ns4oOjsAUkQ==",
    "room_0_inside_0": "hectic-collie-Fg0izmsfWi8kgURPvGk0qw==",
    "room_0_outside_1": "rize-buck-rV5ZXQ12GxQuDIUGKVu6Mg==",
    "room_1_outside_0": "votive-jaguar-putQqtZVgU9Hvxx5eIm2Fg==",
    "room_1_inside_0": "boss-pug-wi6QuZZbAtr5V7krU.ewkQ===",
    "room_1_outside_1": "fleet-bunny-2bZe9+H+l1qbECZ9ED6IYw==",
    "storage_room_0_outside_0": "missed-falcon-DMuRgeXB0EXMZS4MYSd5kQ==",
    "storage_room_0_inside_0": "abuzz-newt-Rx1E8Dzhs7jLRhmono.YXA==",
    "storage_room_0_outside_1": "curly-colt-28WukUdR02W3QbBCSzqUsw==",
    "room_2_outside_0": "averse-dayfly-sl2pPLUDDSiRvCjdt6jj9g==",
    "room_2_inside_0": "unary-amoeba-WGnqua.JuUXk1xFO3JalXQ==",
    "room_2__outside_1": "ace-worm-opjd54QQSj5acIX19ve8jg=="
}

DRINK_COUNTER_MAP = 'kitchen_counter_1'
SNACK_COUNTER_MAP = 'kitchen_counter_0'
STORAGE_COUNTER_MAP = 'storage_room_0_inside_0'
WORK_TABLE_MAP = 'room_0_inside_0' #input("\n(Init State) Where is room table located?\n\n>> ")

assert WORK_TABLE_MAP in graph_nav_loc_to_id.keys()

# OBJECT_LOCATION = "kitchen_counter_1" #input("\n(Init State) Where is object located?\n\n>> ")
# ROOM_TABLE_LOCATION = "room_0_inside_0" #input("\n(Init State) Where is room table located?\n\n>> ")

# assert OBJECT_LOCATION in graph_nav_loc_to_id.keys()
# assert ROOM_TABLE_LOCATION in graph_nav_loc_to_id.keys()

# pylint: disable=no-member
class SpotControllers():
    """Implementation of interface with low-level controllers for the Spot
    robot."""

    def __init__(self) -> None:
        self._hostname = CFG.spot_robot_ip
        self._verbose = False
        self._force_45_angle_grasp = False
        self._force_horizontal_grasp = True
        self._force_squeeze_grasp = False
        self._force_top_down_grasp = False
        self._image_source = "hand_color_image"

        self.hand_x, self.hand_y, self.hand_z = (0.80, 0, 0.45)

        # See hello_spot.py for an explanation of these lines.
        bosdyn.client.util.setup_logging(self._verbose)

        self.sdk = bosdyn.client.create_standard_sdk('ArmObjectGraspClient')
        self.robot: Robot = self.sdk.create_robot(self._hostname)
        bosdyn.client.util.authenticate(self.robot)
        self.robot.time_sync.wait_for_sync()

        assert self.robot.has_arm(
        ), "Robot requires an arm to run this example."

        # Verify the robot is not estopped and that an external application has
        # registered and holds an estop endpoint.
        self.verify_estop(self.robot)

        self.lease_client = self.robot.ensure_client(
            bosdyn.client.lease.LeaseClient.default_service_name)
        self.robot_state_client: RobotStateClient = self.robot.ensure_client(
            RobotStateClient.default_service_name)
        self.robot_command_client: RobotCommandClient = \
            self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.image_client = self.robot.ensure_client(
            ImageClient.default_service_name)
        self.manipulation_api_client = self.robot.ensure_client(
            ManipulationApiClient.default_service_name)
        self.lease_client.take()
        self.lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(
            self.lease_client, must_acquire=True, return_at_exit=True)

        # Create Graph Nav Command Line
        self.upload_filepath = "predicators/spot_utils/kitchen/" + \
            "downloaded_graph/"
        self.graph_nav_command_line = GraphNavInterface(
            self.robot, self.upload_filepath, self.lease_client,
            self.lease_keepalive)

        # Initializing Spot
        self.robot.logger.info(
            "Powering on robot... This may take a several seconds.")
        self.robot.power_on(timeout_sec=20)
        assert self.robot.is_powered_on(), "Robot power on failed."

        self.robot.logger.info("Commanding robot to stand...")
        blocking_stand(self.robot_command_client, timeout_sec=10)
        self.robot.logger.info("Robot standing.")

    def navigateToController(self, objs: Sequence[Object]) -> None:
        """Controller that navigates to specific pre-specified locations."""
        print("NavigateTo", objs)

        # waypoint_id = ""
        # objects by types
        if objs[1].type.name == 'soda_can':
            waypoint_id = graph_nav_loc_to_id[DRINK_COUNTER_MAP]
        elif objs[1].type.name == 'snack':
            waypoint_id = graph_nav_loc_to_id[SNACK_COUNTER_MAP]
        elif objs[1].type.name == 'toy':
            waypoint_id = graph_nav_loc_to_id[STORAGE_COUNTER_MAP]
         
        
        # surface
        elif objs[1].name == 'drink_counter':
            waypoint_id = graph_nav_loc_to_id[DRINK_COUNTER_MAP]
        elif objs[1].name == 'snack_counter':
            waypoint_id = graph_nav_loc_to_id[SNACK_COUNTER_MAP]
        elif objs[1].name == 'storage_counter':
            waypoint_id = graph_nav_loc_to_id[STORAGE_COUNTER_MAP]
        elif objs[1].name == 'work_table':
            waypoint_id = graph_nav_loc_to_id[WORK_TABLE_MAP]
        else:
            raise NotImplementedError()
        
        self.navigate_to(waypoint_id)

    def graspController(self, objs: Sequence[Object]) -> None:
        """Wrapper method for grasp controller."""
        print("Grasp", objs)
        self.arm_object_grasp(objs[1])

    def placeOntopController(self, objs: Sequence[Object]) -> None:
        """Wrapper method for placeOnTop controller."""
        print("PlaceOntop", objs)
        self.hand_movement()

    def verify_estop(self, robot: Any) -> None:
        """Verify the robot is not estopped."""

        client = robot.ensure_client(EstopClient.default_service_name)
        if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
            error_message = "Robot is estopped. Please use an external" + \
                " E-Stop client, such as the estop SDK example, to" + \
                " configure E-Stop."
            robot.logger.error(error_message)
            raise Exception(error_message)

    # NOTE: We want to deprecate this over the long-term!
    def cv_mouse_callback(self, event, x, y, flags, param):  # type: ignore
        """Callback for the click-to-grasp functionality with the Spot API's
        grasping interface."""
        del flags, param
        # pylint: disable=global-variable-not-assigned
        global g_image_click, g_image_display
        clone = g_image_display.copy()
        if event == cv2.EVENT_LBUTTONUP:
            g_image_click = (x, y)
        else:
            # Draw some lines on the image.
            #print('mouse', x, y)
            color = (30, 30, 30)
            thickness = 2
            image_title = 'Click to grasp'
            height = clone.shape[0]
            width = clone.shape[1]
            cv2.line(clone, (0, y), (width, y), color, thickness)
            cv2.line(clone, (x, 0), (x, height), color, thickness)
            cv2.imshow(image_title, clone)

    def add_grasp_constraint(
        self, grasp: manipulation_api_pb2.PickObjectInImage,
        robot_state_client: RobotStateClient
    ) -> manipulation_api_pb2.PickObjectInImage:
        """Method to constrain desirable grasps."""
        # There are 3 types of constraints:
        #   1. Vector alignment
        #   2. Full rotation
        #   3. Squeeze grasp
        #
        # You can specify more than one if you want and they will be
        # OR'ed together.

        # For these options, we'll use a vector alignment constraint.
        use_vector_constraint = self._force_top_down_grasp or \
            self._force_horizontal_grasp

        # Specify the frame we're using.
        grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

        if use_vector_constraint:
            if self._force_top_down_grasp:
                # Add a constraint that requests that the x-axis of the
                # gripper is pointing in the negative-z direction in the
                # vision frame.

                # The axis on the gripper is the x-axis.
                axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

                # The axis in the vision frame is the negative z-axis
                axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

            if self._force_horizontal_grasp:
                # Add a constraint that requests that the y-axis of the
                # gripper is pointing in the positive-z direction in the
                # vision frame.  That means that the gripper is
                # constrained to be rolled 90 degrees and pointed at the
                # horizon.

                # The axis on the gripper is the y-axis.
                axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

                # The axis in the vision frame is the positive z-axis
                axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

            # Add the vector constraint to our proto.
            constraint = grasp.grasp_params.allowable_orientation.add()
            constraint.vector_alignment_with_tolerance.\
                axis_on_gripper_ewrt_gripper.\
                    CopyFrom(axis_on_gripper_ewrt_gripper)
            constraint.vector_alignment_with_tolerance.\
                axis_to_align_with_ewrt_frame.\
                    CopyFrom(axis_to_align_with_ewrt_vo)

            # We'll take anything within about 10 degrees for top-down or
            # horizontal grasps.
            constraint.vector_alignment_with_tolerance.\
                threshold_radians = 0.17

        elif self._force_45_angle_grasp:
            # Demonstration of a RotationWithTolerance constraint.
            # This constraint allows you to specify a full orientation you
            # want the hand to be in, along with a threshold.
            # You might want this feature when grasping an object with known
            # geometry and you want to make sure you grasp a specific part
            # of it. Here, since we don't have anything in particular we
            # want to grasp,  we'll specify an orientation that will have the
            # hand aligned with robot and rotated down 45 degrees as an
            # example.

            # First, get the robot's position in the world.
            robot_state = robot_state_client.get_robot_state()
            vision_T_body = get_vision_tform_body(
                robot_state.kinematic_state.transforms_snapshot)

            # Rotation from the body to our desired grasp.
            body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45 degrees
            vision_Q_grasp = vision_T_body.rotation * body_Q_grasp

            # Turn into a proto
            constraint = grasp.grasp_params.allowable_orientation.add()
            constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(
                vision_Q_grasp.to_proto())

            # We'll accept anything within +/- 10 degrees
            constraint.rotation_with_tolerance.threshold_radians = 0.17

        elif self._force_squeeze_grasp:
            # Tell the robot to just squeeze on the ground at the given point.
            constraint = grasp.grasp_params.allowable_orientation.add()
            constraint.squeeze_grasp.SetInParent()

        return grasp

    def arm_object_grasp(self, obj) -> None:
        """A simple example of using the Boston Dynamics API to command Spot's
        arm."""
        assert self.robot.is_powered_on(), "Robot power on failed."
        assert basic_command_pb2.StandCommand.Feedback.STATUS_IS_STANDING

        vel_command = RobotCommandBuilder.synchro_velocity_command(v_x=-VELOCITY_BASE_SPEED, v_y=0.0, v_rot=0.0)
        self.robot_command_client.robot_command(command=vel_command, end_time_secs=time.time() + VELOCITY_CMD_DURATION)

        self.robot.logger.info("Move back command issued.")
        time.sleep(2)

        # Take a picture with a camera
        self.robot.logger.info(f'Getting an image from: {self._image_source}')
        image_responses = self.image_client.get_image_from_sources(
            [self._image_source])

        if len(image_responses) != 1:
            print(f'Got invalid number of images: {str(len(image_responses))}')
            print(image_responses)
            assert False

        image = image_responses[0]
        if image.shot.image.pixel_format == image_pb2.Image.\
            PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16  # type: ignore
        else:
            dtype = np.uint8  # type: ignore
        img = np.fromstring(image.shot.image.data, dtype=dtype)  # type: ignore
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(image.shot.image.rows, image.shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)

        # # Show the image to the user and wait for them to click on a pixel
        # self.robot.logger.info('Click on an object to start grasping...')
        # image_title = 'Click to grasp'
        # cv2.namedWindow(image_title)
        # cv2.setMouseCallback(image_title, self.cv_mouse_callback)

        # # pylint: disable=global-variable-not-assigned
        # global g_image_click, g_image_display
        # g_image_display = img
        # cv2.imshow(image_title, g_image_display)
        # while g_image_click is None:
        #     key = cv2.waitKey(1) & 0xFF
        #     if key == ord('q') or key == ord('Q'):
        #         # Quit
        #         print('"q" pressed, exiting.')
        #         sys.exit()

        #####################
        global g_image_click, g_image_display
        g_image_display = img
        # Inference
        results = model([g_image_display], size=g_image_display.shape[0]) # batch of images

        # Results
        #results.print()
        # results.show()

        #results.xyxy[0]  # im1 predictions (tensor)
        #print(results.xyxy[0])
        for result in results.xyxy[0]:
            x = float(result[0] + result[2])/2
            y = float(result[1] + result[3])/2
            prob = float(result[4])
            name = model.names[int(result[5])]
            print(name, x, y, prob)
            if 'soda_can' in obj.name and name == 'bottle':
                g_image_click = (int(x), int(y))
                break
        
        if g_image_click is None:       
            # # Show the image to the user and wait for them to click on a pixel
            self.robot.logger.info('Click on an object to start grasping...')
            image_title = 'Click to grasp'
            cv2.namedWindow(image_title)
            cv2.setMouseCallback(image_title, self.cv_mouse_callback)

            # pylint: disable=global-variable-not-assigned
            cv2.imshow(image_title, g_image_display)
            while g_image_click is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    # Quit
                    print('"q" pressed, exiting.')
                    sys.exit()
        #####################

        self.robot.\
            logger.info(f"Object at ({g_image_click[0]}, {g_image_click[1]})")

        pick_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])

        # Build the proto
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole)

        # Optionally add a grasp constraint.  This lets you tell the robot you
        # only want top-down grasps or side-on grasps.
        grasp = self.add_grasp_constraint(grasp, self.robot_state_client)

        # Ask the robot to pick up the object
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(
            pick_object_in_image=grasp)

        # Send the request
        cmd_response = self.manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request)

        # Get feedback from the robot
        while True:
            feedback_request = manipulation_api_pb2.\
                ManipulationApiFeedbackRequest(manipulation_cmd_id=\
                    cmd_response.manipulation_cmd_id)

            # Send the request
            response = self.manipulation_api_client.\
                manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print(
                'Current state: ',
                manipulation_api_pb2.ManipulationFeedbackState.Name(
                    response.current_state))

            if response.current_state in [manipulation_api_pb2.\
                MANIP_STATE_GRASP_SUCCEEDED, manipulation_api_pb2.\
                MANIP_STATE_GRASP_FAILED]:
                break

        # Unstow the arm
        unstow = RobotCommandBuilder.arm_ready_command()

        # Issue the command via the RobotCommandClient
        unstow_command_id = self.robot_command_client.robot_command(unstow)

        self.robot.logger.info("Unstow command issued.")
        block_until_arm_arrives(self.robot_command_client, unstow_command_id,
                                3.0)

        time.sleep(1.0)

        ### TODO (wmcclinton) Does not work!!! Stow the arm
        # To allow stowing
        grasp_carry_state_override = manipulation_api_pb2.ApiGraspedCarryStateOverride(override_request=3)
        grasp_override_request = manipulation_api_pb2.ApiGraspOverrideRequest(carry_state_override=grasp_carry_state_override)
        cmd_response = self.manipulation_api_client.grasp_override_command(grasp_override_request)
        #
        
        stow_cmd = RobotCommandBuilder.arm_stow_command()
        stow_command_id = self.robot_command_client.robot_command(stow_cmd)
        self.robot.logger.info("Stow command issued.")
        block_until_arm_arrives(self.robot_command_client, stow_command_id,
                                3.0)

        self.robot.logger.info('Finished grasp.')

        g_image_click = None
        g_image_display = None  

        time.sleep(2.0)

    def block_until_arm_arrives_with_prints(self, robot: Robot,
                                            command_client: RobotCommandClient,
                                            cmd_id: int) -> None:
        """Block until the arm arrives at the goal and print the distance
        remaining.

        Note: a version of this function is available as a helper in
        robot_command without the prints.
        """
        while True:
            feedback_resp = command_client.robot_command_feedback(cmd_id)

            if feedback_resp.feedback.synchronized_feedback.\
                arm_command_feedback.arm_cartesian_feedback.status == \
                arm_command_pb2.ArmCartesianCommand.Feedback.\
                    STATUS_TRAJECTORY_COMPLETE:
                robot.logger.info('Move complete.')
                break
            time.sleep(0.1)

    def hand_movement(self) -> None:
        """Move arm to infront of robot an open gripper."""
        # Move the arm to a spot in front of the robot, and open the gripper.
        assert self.robot.is_powered_on(), "Robot power on failed."
        assert basic_command_pb2.StandCommand.Feedback.STATUS_IS_STANDING

        # Rotation as a quaternion
        qw = np.cos((np.pi / 4) / 2)
        qx = 0
        qy = np.sin((np.pi / 4) / 2)
        qz = 0
        flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

        # Make the arm pose RobotCommand
        # Build a position to move the arm to (in meters, relative to and
        # expressed in the gravity aligned body frame).
        x = self.hand_x
        y = self.hand_y
        z = self.hand_z
        hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

        flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                                rotation=flat_body_Q_hand)

        robot_state = self.robot_state_client.get_robot_state()
        odom_T_flat_body = get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME,
            GRAV_ALIGNED_BODY_FRAME_NAME)

        odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(
            flat_body_T_hand)

        # duration in seconds
        seconds = 2

        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w,
            odom_T_hand.rot.x, odom_T_hand.rot.y, odom_T_hand.rot.z,
            ODOM_FRAME_NAME, seconds)

        # Make the open gripper RobotCommand
        gripper_command = RobotCommandBuilder.\
            claw_gripper_open_fraction_command(0.0)

        # Combine the arm and gripper commands into one RobotCommand
        command = RobotCommandBuilder.build_synchro_command(
            gripper_command, arm_command)

        # Send the request
        cmd_id: int = self.robot_command_client.robot_command(command)
        self.robot.logger.info('Moving arm to position.')

        # Wait until the arm arrives at the goal.
        self.block_until_arm_arrives_with_prints(self.robot,
                                                 self.robot_command_client,
                                                 cmd_id)

        time.sleep(2)

        # Make the open gripper RobotCommand
        gripper_command = RobotCommandBuilder.\
            claw_gripper_open_fraction_command(1.0)

        # Combine the arm and gripper commands into one RobotCommand
        command = RobotCommandBuilder.build_synchro_command(
            gripper_command, arm_command)

        # Send the request
        cmd_id = self.robot_command_client.robot_command(command)
        self.robot.logger.info('Moving arm to position.')

        # Wait until the arm arrives at the goal.
        self.block_until_arm_arrives_with_prints(self.robot,
                                                 self.robot_command_client,
                                                 cmd_id)

        time.sleep(2)

        stow_cmd = RobotCommandBuilder.arm_stow_command()
        stow_command_id = self.robot_command_client.robot_command(stow_cmd)
        self.robot.logger.info("Stow command issued.")
        block_until_arm_arrives(self.robot_command_client, stow_command_id,
                                3.0)

    def navigate_to(self, waypoint_id: str) -> None:
        """Use GraphNavInterface to localize robot and go to a location."""
        # pylint: disable=broad-except
        try:
            # (1) Initialize location
            self.graph_nav_command_line.set_initial_localization_fiducial()
        except Exception as e:
            print(e)
        
        try:
            # (2) Get localization state
            self.graph_nav_command_line.get_localization_state()
        except Exception as e:
            print(e)
        
        try:
            # (4) Navigate to
            self.graph_nav_command_line.navigate_to([waypoint_id])
        except Exception as e:
            print(e)
