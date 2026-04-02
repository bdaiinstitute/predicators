"""Calibrate external iPhone camera w.r.t. Spot body frame.

This script has two main modes:

1) Data collection:
   - Move Spot's hand to different poses observing a Charuco board.
   - Capture synchronized RGBD from Spot's in-hand camera.
   - Capture synchronized RGBD (and intrinsics) from an external iPhone.
   - Save all data + necessary poses to disk.

2) Calibration:
   - Load the saved dataset.
   - Estimate the rigid transform between the iPhone camera frame
     and Spot's body frame (T_body_iphone) via a hand-eye style calibration.
   - Save T_body_iphone to disk for later use in skills such as wipe_online.

The iPhone integration is intentionally modular. You should provide a small
helper (e.g. in a separate module) that returns the latest RGBD frame plus
intrinsics, or write the latest frame to disk from your Kiwi receiver and
have this script read it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization
import rerun as rr
from bosdyn.client import create_standard_sdk
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, get_a_tform_b
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.sdk import Robot
from bosdyn.client.util import authenticate

from predicators.spot_utils.skills.iphone_kiwi_receiver import KiwiReceiver
from predicators.spot_utils.perception.spot_cameras import _image_response_to_image
from predicators.spot_utils.utils import verify_estop


class ThreadedKiwiReceiver:
    """Wrapper around KiwiReceiver that continuously drains frames in a background thread.

    This ensures we always get the latest frame from the iPhone TCP stream,
    rather than buffered/old frames that accumulate in the TCP queue.
    """

    def __init__(self):
        """Initialize and start the background receiver thread."""
        self._receiver = KiwiReceiver()
        self._latest_frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        print("[INFO] Started background iPhone frame receiver thread")

    def _receive_loop(self):
        """Background thread that continuously receives frames."""
        while self._running:
            try:
                frame = self._receiver.recv_frame()
                with self._lock:
                    self._latest_frame = frame
            except Exception as e:
                if self._running:
                    print(f"[WARN] Error receiving iPhone frame: {e}")
                break

    def get_latest_frame(self):
        """Get the most recent frame received from the iPhone.

        Returns:
            The latest frame object from KiwiReceiver, or None if no frames received yet.

        """
        with self._lock:
            return self._latest_frame

    def stop(self):
        """Stop the background receiver thread and close the connection."""
        self._running = False
        self._thread.join(timeout=2.0)
        self._receiver.close()
        print("[INFO] Stopped background iPhone frame receiver thread")


def rgbd_to_point_cloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float = 1000.0,
    max_depth: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert an RGBD image and intrinsics into a point cloud in the camera frame.

    Args:
        rgb: HxWx3 uint8 array, assumed RGB.
        depth: HxW (or HxWx1) array, uint16 in depth_scale units or float32 meters.
        intrinsics: 3x3 matrix or array-like [fx, fy, cx, cy].
        depth_scale: Scale factor from uint16 depth units to meters (default: 1000).
        max_depth: Optional maximum depth in meters for filtering points.

    Returns:
        points: Nx3 float32 array of 3D points in the camera frame.
        colors: Nx3 float32 array of RGB colors in [0, 1].

    """
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    if depth.dtype == np.uint16 or depth.dtype == np.int32:
        depth_m = depth.astype(np.float32) / float(depth_scale)
    else:
        depth_m = depth.astype(np.float32)

    H, W = depth_m.shape
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_NEAREST)

    K = np.asarray(intrinsics, dtype=np.float32)
    if K.shape == (3, 3):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
    elif K.size == 4:
        fx, fy, cx, cy = K.ravel().tolist()
    else:
        raise ValueError(f"Intrinsics must be 3x3 or length-4, got shape {K.shape}")

    u_coords, v_coords = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )

    z = depth_m
    valid = z > 0
    if max_depth is not None:
        valid &= z <= float(max_depth)

    x = (u_coords - cx) / fx * z
    y = (v_coords - cy) / fy * z

    points = np.stack((x, y, z), axis=-1)[valid]

    rgb_float = rgb.astype(np.float32) / 255.0
    colors = rgb_float.reshape(-1, 3)[valid.ravel()]

    return points.astype(np.float32), colors.astype(np.float32)


@dataclass
class SpotFrame:
    """RGBD + pose from Spot's in-hand camera at a single time."""

    rgb_path: str
    depth_path: str
    camera_matrix: List[List[float]]  # 3x3 intrinsics
    T_body_hand: List[List[float]]  # 4x4 BODY->hand camera
    rgb: np.ndarray
    depth: np.ndarray


@dataclass
class IphoneFrame:
    """RGBD + intrinsics from iPhone at a single time.

    NOTE: You are responsible for implementing the data source. For example:
      - A helper function that talks to your Kiwi receiver in-process, or
      - Reading the most recent RGB/depth/intrinsics from disk that the
        Kiwi receiver script has written.
    """

    rgb_path: str
    depth_path: str
    camera_matrix: List[List[float]]  # 3x3 intrinsics
    rgb: np.ndarray
    depth: Optional[np.ndarray]

    @property
    def intrinsics(self) -> List[List[float]]:
        """Alias for camera_matrix, used by skill code."""
        return self.camera_matrix


@dataclass
class CalibrationSample:
    """One paired observation of the Charuco board from both cameras."""

    # Board pose in Spot hand camera frame (OpenCV rvec/tvec)
    rvec_hand: List[float]
    tvec_hand: List[float]

    # Board pose in iPhone camera frame (OpenCV rvec/tvec)
    rvec_iphone: List[float]
    tvec_iphone: List[float]

    # BODY -> hand camera transform at capture time
    T_body_hand: List[List[float]]


def init_robot(hostname: str) -> Robot:
    """Initialize Spot robot for read-only camera access (no lease required)."""
    sdk = create_standard_sdk("CalibrateIphoneClient")
    robot = sdk.create_robot(hostname)
    authenticate(robot)
    verify_estop(robot)
    robot.time_sync.wait_for_sync()
    print("[INFO] Robot connection and time sync successful.")
    return robot


def _capture_spot_hand_frame(
    robot: Robot,
    save_dir: Path,
    sample_idx: int,
) -> SpotFrame:
    """Capture RGBD + BODY->hand transform from Spot's in-hand camera.

    This version uses the Image service directly and does not require a lease
    or SpotLocalizer. It relies on the transforms snapshot attached to the
    image to compute BODY->hand.
    """
    camera_name = "hand_color_image"
    image_client = robot.ensure_client(ImageClient.default_service_name)

    # Build RGB + depth image requests for the hand camera.
    rgb_req = build_image_request(
        camera_name,
        quality_percent=100,
        pixel_format=None,
    )
    depth_req = build_image_request(
        "hand_depth_in_hand_color_frame",
        quality_percent=100,
        pixel_format=None,
    )
    responses = image_client.get_image([rgb_req, depth_req])
    name_to_resp = {r.source.name: r for r in responses}
    rgb_resp = name_to_resp[camera_name]
    depth_resp = name_to_resp["hand_depth_in_hand_color_frame"]

    rgb = _image_response_to_image(rgb_resp)
    depth = _image_response_to_image(depth_resp)

    sample_dir = save_dir / f"sample_{sample_idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = sample_dir / "spot_rgb.png"
    depth_path = sample_dir / "spot_depth.png"
    intrinsics_path = sample_dir / "spot_intrinsics.json"

    # Save images to disk
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(depth_path), depth)

    cam = rgb_resp.source.pinhole.intrinsics
    K = np.array(
        [
            [cam.focal_length.x, 0.0, cam.principal_point.x],
            [0.0, cam.focal_length.y, cam.principal_point.y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Save intrinsics alongside the images for later use (e.g., point-cloud generation).
    H, W = rgb.shape[:2]
    with open(intrinsics_path, "w") as f:
        json.dump(
            {
                "K": K.tolist(),
                "width": int(W),
                "height": int(H),
            },
            f,
            indent=2,
        )

    # BODY -> hand camera using transforms snapshot attached to the image.
    T_body_hand = get_a_tform_b(
        rgb_resp.shot.transforms_snapshot,
        BODY_FRAME_NAME,
        rgb_resp.shot.frame_name_image_sensor,
    ).to_matrix()

    return SpotFrame(
        rgb_path=str(rgb_path),
        depth_path=str(depth_path),
        camera_matrix=K.tolist(),
        T_body_hand=T_body_hand.tolist(),
        rgb=rgb,
        depth=depth,
    )


def _capture_iphone_frame(
    receiver: ThreadedKiwiReceiver,
    save_dir: Path,
    sample_idx: int,
) -> IphoneFrame:
    """Capture RGBD + intrinsics from the iPhone via ThreadedKiwiReceiver.

    For each sample, we:
      - Get the latest frame from the background receiver thread.
      - Save RGB, depth, and intrinsics to disk under sample_XXXX/.
      - Return an IphoneFrame pointing to those saved files.
    """
    sample_dir = save_dir / f"sample_{sample_idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = sample_dir / "iphone_rgb.png"
    depth_path = sample_dir / "iphone_depth.npy"
    intrinsics_path = sample_dir / "iphone_intrinsics.json"

    frame = receiver.get_latest_frame()
    if frame is None:
        raise RuntimeError("No iPhone frame received yet. Ensure iPhone is streaming.")
    rgb = frame.rgb
    depth = frame.depth

    # Save RGB image (frame.rgb is RGB; OpenCV expects BGR)
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Save depth if available
    if depth is not None:
        np.save(str(depth_path), depth.astype(np.float32))
    else:
        # Create an empty placeholder depth if none is provided
        depth = np.zeros(rgb.shape[:2], dtype=np.float32)
        np.save(str(depth_path), depth.astype(np.float32))

    # Save intrinsics
    K = np.asarray(frame.intrinsics, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics from Kiwi, got shape {K.shape}")
    with open(intrinsics_path, "w") as f:
        json.dump({"K": K.tolist()}, f, indent=2)

    return IphoneFrame(
        rgb_path=str(rgb_path),
        depth_path=str(depth_path),
        camera_matrix=K.tolist(),
        rgb=rgb,
        depth=depth,
    )


def _build_charuco_board():
    """Create a Charuco board object matching the physical Calib.io board.

    Board spec (from print):
      - 9 x 14  (short side x long side)
      - Checker Size: 20 mm
      - Marker Size: 15 mm
      - Dictionary: AruCo DICT_5x5
    """
    # OpenCV expects squaresX along the X axis (long side in our usage).
    # The physical board has 14 squares along the long side and 9 along the short.
    squaresX = 14
    squaresY = 9
    squareLength = 0.02  # meters (20 mm)
    markerLength = 0.015  # meters (15 mm)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)

    # New-style API in OpenCV 4.7+ (which you have: 4.9.0)
    board = cv2.aruco.CharucoBoard(
        (squaresX, squaresY),
        squareLength,
        markerLength,
        dictionary,
    )
    return board


def _detect_charuco_pose(
    rgb_path: str,
    board,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Estimate Charuco board pose in the given camera frame.

    Returns:
        (rvec, tvec) if successful, otherwise None.

    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    img_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[WARN] Could not read image at {rgb_path}")
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Obtain the ArUco dictionary from the board, accounting for API changes.
    if hasattr(board, "dictionary"):
        aruco_dict = board.dictionary
    elif hasattr(board, "getDictionary"):
        aruco_dict = board.getDictionary()
    else:
        raise AttributeError(
            "Charuco board has neither 'dictionary' attribute nor 'getDictionary()' method."
        )

    # OpenCV ArUco API differs across versions:
    # - Older: DetectorParameters_create + detectMarkers(...)
    # - Newer (4.7+): DetectorParameters class + ArucoDetector(...)
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=params
        )
    else:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        print(f"[WARN] No ArUco markers detected in {rgb_path}")
        return None

    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board,
    )
    if charuco_ids is None or len(charuco_ids) < 4:
        print(f"[WARN] Not enough Charuco corners in {rgb_path}")
        return None

    # OpenCV ArUco API has changed across versions; some builds require rvec/tvec
    # arguments without defaults. Use positional arguments and explicitly pass
    # None for rvec/tvec so both older and newer versions work.
    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board,
        K,
        dist_coeffs,
        None,
        None,
    )
    if not success:
        print(f"[WARN] Charuco pose estimation failed for {rgb_path}")
        return None

    return rvec, tvec


def collect_calibration_data(
    robot: Robot,
    output_dir: Path,
    num_samples: int,
) -> None:
    """Interactively collect paired Spot/iPhone Charuco observations.

    For each sample:
      - Waits for user to position the arm/board and press Enter.
      - Captures Spot hand RGBD and saves to disk.
      - Gets the latest iPhone RGBD + intrinsics from the background receiver.
      - Estimates Charuco poses in both cameras.
      - Saves a CalibrationSample into samples.json.

    The iPhone receiver runs in a background thread, continuously draining
    the TCP stream to ensure we always get the freshest frame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    board = _build_charuco_board()

    samples: List[CalibrationSample] = []
    receiver = ThreadedKiwiReceiver()

    for idx in range(num_samples):
        print()
        input(
            f"[DATA COLLECTION] Sample {idx+1}/{num_samples}: "
            f"Position the arm/board, ensure iPhone data is saved for this sample, "
            f"then press Enter to capture Spot data and run Charuco detection..."
        )

        # 1) Spot hand frame
        spot_frame = _capture_spot_hand_frame(robot, output_dir, idx)
        K_hand = np.array(spot_frame.camera_matrix, dtype=np.float64)

        # 2) iPhone frame (pulled directly from KiwiReceiver)
        try:
            iphone_frame = _capture_iphone_frame(receiver, output_dir, idx)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Skipping sample {idx}: could not load iPhone data: {e}")
            continue
        K_iphone = np.array(iphone_frame.camera_matrix, dtype=np.float64)

        # 2b) Log RGB + point clouds for both cameras to Rerun for this sample.
        rr.set_time_sequence("sample", idx)

        # Spot hand camera logs: RGB + point cloud (no depth image)
        rr.log("spot/hand/rgb", rr.Image(spot_frame.rgb))

        points_spot_cam, colors_spot = rgbd_to_point_cloud(
            spot_frame.rgb, spot_frame.depth, K_hand, depth_scale=1000.0
        )
        if points_spot_cam.size > 0:
            rr.log(
                "spot/hand/points3d_cam",
                rr.Points3D(
                    positions=points_spot_cam,
                    colors=(colors_spot * 255).astype(np.uint8),
                ),
            )

        # iPhone logs: RGB + point cloud (no depth image)
        rr.log("iphone/rgb", rr.Image(iphone_frame.rgb))
        if iphone_frame.depth is not None:
            # The iPhone intrinsics K are defined for the RGB resolution, which
            # is currently assumed to be 720x960 (H x W). The depth is lower
            # resolution (e.g. 192x256), so we scale K to the depth resolution
            # before constructing the point cloud.
            depth_h, depth_w = iphone_frame.depth.shape[:2]
            base_h, base_w = 720.0, 960.0
            sx = depth_w / base_w
            sy = depth_h / base_h

            fx, fy, cx, cy = (
                K_iphone[0, 0],
                K_iphone[1, 1],
                K_iphone[0, 2],
                K_iphone[1, 2],
            )
            K_scaled = np.array(
                [
                    [fx * sx, 0.0, cx * sx],
                    [0.0, fy * sy, cy * sy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

            points_iphone_cam, colors_iphone = rgbd_to_point_cloud(
                iphone_frame.rgb, iphone_frame.depth, K_scaled, depth_scale=1.0
            )
            if points_iphone_cam.size > 0:
                rr.log(
                    "iphone/points3d_cam",
                    rr.Points3D(
                        positions=points_iphone_cam,
                        colors=(colors_iphone * 255).astype(np.uint8),
                    ),
                )

        # 3) Charuco pose in hand camera
        res_hand = _detect_charuco_pose(
            spot_frame.rgb_path,
            board,
            K_hand,
        )
        if res_hand is None:
            print(f"[WARN] Skipping sample {idx}: no Charuco pose in hand camera.")
            continue
        rvec_hand, tvec_hand = res_hand

        # 4) Charuco pose in iPhone camera
        res_iphone = _detect_charuco_pose(
            iphone_frame.rgb_path,
            board,
            K_iphone,
        )
        if res_iphone is None:
            print(f"[WARN] Skipping sample {idx}: no Charuco pose in iPhone camera.")
            continue
        rvec_iphone, tvec_iphone = res_iphone

        sample = CalibrationSample(
            rvec_hand=rvec_hand.flatten().tolist(),
            tvec_hand=tvec_hand.flatten().tolist(),
            rvec_iphone=rvec_iphone.flatten().tolist(),
            tvec_iphone=tvec_iphone.flatten().tolist(),
            T_body_hand=spot_frame.T_body_hand,
        )
        samples.append(sample)

        # Append to per-sample JSON as well (for debugging)
        with open(output_dir / f"sample_{idx:04d}" / "calib_sample.json", "w") as f:
            json.dump(dataclasses.asdict(sample), f, indent=2)

        print(f"[INFO] Recorded calibration sample {idx}")

    # Save all samples into a single JSON file
    samples_json_path = output_dir / "samples.json"
    with open(samples_json_path, "w") as f:
        json.dump([dataclasses.asdict(s) for s in samples], f, indent=2)

    print(f"[INFO] Saved {len(samples)} valid calibration samples to {samples_json_path}")

    # Stop the background receiver thread
    receiver.stop()


def _load_samples(samples_json_path: Path) -> List[CalibrationSample]:
    with open(samples_json_path, "r") as f:
        raw = json.load(f)
    samples: List[CalibrationSample] = []
    for item in raw:
        samples.append(
            CalibrationSample(
                rvec_hand=item["rvec_hand"],
                tvec_hand=item["tvec_hand"],
                rvec_iphone=item["rvec_iphone"],
                tvec_iphone=item["tvec_iphone"],
                T_body_hand=item["T_body_hand"],
            )
        )
    return samples


def _solve_hand_eye(samples: List[CalibrationSample]) -> np.ndarray:
    """Solve for the fixed transform between hand camera and iPhone camera.

    Uses OpenCV's calibrateHandEye on camera motions relative to the Charuco board.

    Returns:
        T_hand_iphone: 4x4 transform matrix mapping points from iPhone frame
                       into the Spot hand camera frame.

    """
    if len(samples) < 2:
        raise ValueError("Need at least 2 calibration samples for hand–eye.")

    R_hand_wrt_board = []
    t_hand_wrt_board = []
    R_iphone_wrt_board = []
    t_iphone_wrt_board = []

    for s in samples:
        rvec_h = np.array(s.rvec_hand, dtype=np.float64).reshape(3, 1)
        tvec_h = np.array(s.tvec_hand, dtype=np.float64).reshape(3, 1)
        rvec_p = np.array(s.rvec_iphone, dtype=np.float64).reshape(3, 1)
        tvec_p = np.array(s.tvec_iphone, dtype=np.float64).reshape(3, 1)

        R_bh, _ = cv2.Rodrigues(rvec_h)  # board in hand
        R_bp, _ = cv2.Rodrigues(rvec_p)  # board in iphone

        # We want camera-in-board. Invert:
        R_hb = R_bh.T
        t_hb = -R_bh.T @ tvec_h

        R_pb = R_bp.T
        t_pb = -R_bp.T @ tvec_p

        R_hand_wrt_board.append(R_hb)
        t_hand_wrt_board.append(t_hb)
        R_iphone_wrt_board.append(R_pb)
        t_iphone_wrt_board.append(t_pb)

    # Solve A_i X = X B_i
    # Where A_i, B_i are motions between successive camera poses.
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_hand_wrt_board,
        t_hand_wrt_board,
        R_iphone_wrt_board,
        t_iphone_wrt_board,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS,
    )

    T_hand_iphone = np.eye(4, dtype=np.float64)
    T_hand_iphone[:3, :3] = R_cam2gripper
    T_hand_iphone[:3, 3] = t_cam2gripper.flatten()

    return T_hand_iphone


def _compute_T_body_iphone(
    samples: List[CalibrationSample],
    T_hand_iphone: np.ndarray,
) -> np.ndarray:
    """Compose BODY->hand with hand->iPhone to get BODY->iPhone.

    We can compute T_body_iphone for each sample and then average in SE(3).
    """
    T_list = []
    for s in samples:
        T_body_hand = np.array(s.T_body_hand, dtype=np.float64)
        T_body_iphone = T_body_hand @ T_hand_iphone
        T_list.append(T_body_iphone)

    # Simple averaging in Lie algebra (log/exp) for robustness.
    # For a small number of samples and reasonable noise, this is fine.
    def se3_log(T: np.ndarray) -> np.ndarray:
        R = T[:3, :3]
        t = T[:3, 3]
        theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        if theta < 1e-8:
            omega = np.zeros(3)
        else:
            omega = theta / (2.0 * np.sin(theta)) * np.array(
                [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
            )
        return np.concatenate([omega, t])

    def se3_exp(xi: np.ndarray) -> np.ndarray:
        omega = xi[:3]
        t = xi[3:]
        theta = np.linalg.norm(omega)
        if theta < 1e-8:
            R = np.eye(3)
        else:
            k = omega / theta
            K = np.array(
                [
                    [0.0, -k[2], k[1]],
                    [k[2], 0.0, -k[0]],
                    [-k[1], k[0], 0.0],
                ]
            )
            R = (
                np.eye(3)
                + np.sin(theta) * K
                + (1.0 - np.cos(theta)) * (K @ K)
            )
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    xis = [se3_log(T) for T in T_list]
    xi_mean = np.mean(xis, axis=0)
    T_mean = se3_exp(xi_mean)
    return T_mean


def _visualize_aligned_point_clouds(
    samples: List[CalibrationSample],
    T_body_iphone: np.ndarray,
    samples_root: Path,
    max_samples: int = 3,
) -> None:
    """Reload a few samples and visualize Spot/iPhone clouds in BODY frame.

    For each sample, this:
      - Reconstructs point clouds in each camera frame from the saved RGBD.
      - Transforms Spot hand and iPhone clouds into the BODY frame.
      - Logs both clouds to Rerun for visual inspection of the calibration.
    """
    T_iphone_body = np.linalg.inv(T_body_iphone)

    def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply a 4x4 transform to an Nx3 point cloud."""
        if pts.size == 0:
            return pts
        pts_h = np.concatenate(
            [pts.astype(np.float64), np.ones((pts.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        pts_body_h = (T @ pts_h.T).T
        return pts_body_h[:, :3].astype(np.float32)

    for idx, s in enumerate(samples[:max_samples]):
        sample_dir = samples_root / f"sample_{idx:04d}"

        # --- Spot hand camera data ---
        spot_rgb_path = sample_dir / "spot_rgb.png"
        spot_depth_path = sample_dir / "spot_depth.png"
        spot_intrinsics_path = sample_dir / "spot_intrinsics.json"

        if not spot_rgb_path.exists() or not spot_depth_path.exists() or not spot_intrinsics_path.exists():
            continue

        with open(spot_intrinsics_path, "r") as f:
            intr_hand = json.load(f)
        K_hand = np.array(intr_hand["K"], dtype=np.float32)

        rgb_hand_bgr = cv2.imread(str(spot_rgb_path), cv2.IMREAD_COLOR)
        if rgb_hand_bgr is None:
            continue
        rgb_hand = cv2.cvtColor(rgb_hand_bgr, cv2.COLOR_BGR2RGB)
        depth_hand = cv2.imread(str(spot_depth_path), cv2.IMREAD_UNCHANGED)
        if depth_hand is None:
            continue

        pts_hand_cam, colors_hand = rgbd_to_point_cloud(
            rgb_hand, depth_hand, K_hand, depth_scale=1000.0
        )

        # --- iPhone data ---
        iphone_rgb_path = sample_dir / "iphone_rgb.png"
        iphone_depth_path = sample_dir / "iphone_depth.npy"
        iphone_intrinsics_path = sample_dir / "iphone_intrinsics.json"

        if not iphone_rgb_path.exists() or not iphone_depth_path.exists() or not iphone_intrinsics_path.exists():
            continue

        with open(iphone_intrinsics_path, "r") as f:
            intr_iphone = json.load(f)
        K_iphone = np.array(intr_iphone["K"], dtype=np.float32)

        rgb_iphone_bgr = cv2.imread(str(iphone_rgb_path), cv2.IMREAD_COLOR)
        if rgb_iphone_bgr is None:
            continue
        rgb_iphone = cv2.cvtColor(rgb_iphone_bgr, cv2.COLOR_BGR2RGB)
        depth_iphone = np.load(str(iphone_depth_path))

        # Scale iPhone intrinsics from assumed RGB resolution (720x960) to
        # the actual depth resolution before constructing the point cloud.
        depth_h, depth_w = depth_iphone.shape[:2]
        base_h, base_w = 720.0, 960.0
        sx = depth_w / base_w
        sy = depth_h / base_h

        fx_i, fy_i, cx_i, cy_i = (
            K_iphone[0, 0],
            K_iphone[1, 1],
            K_iphone[0, 2],
            K_iphone[1, 2],
        )
        K_iphone_scaled = np.array(
            [
                [fx_i * sx, 0.0, cx_i * sx],
                [0.0, fy_i * sy, cy_i * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        pts_iphone_cam, colors_iphone = rgbd_to_point_cloud(
            rgb_iphone, depth_iphone, K_iphone_scaled, depth_scale=1.0
        )

        # --- Transform into BODY frame ---
        T_body_hand = np.array(s.T_body_hand, dtype=np.float64)

        pts_hand_body = _transform_points(T_body_hand, pts_hand_cam)
        pts_iphone_body = _transform_points(T_iphone_body, pts_iphone_cam)

        # --- Log to Rerun ---
        rr.set_time_sequence("calib_sample", idx)

        if pts_hand_body.size > 0:
            rr.log(
                "body/spot_hand_points",
                rr.Points3D(
                    positions=pts_hand_body,
                    colors=(colors_hand * 255).astype(np.uint8),
                ),
            )

        if pts_iphone_body.size > 0:
            rr.log(
                "body/iphone_points",
                rr.Points3D(
                    positions=pts_iphone_body,
                    colors=(colors_iphone * 255).astype(np.uint8),
                ),
            )


def run_calibration(
    samples_json_path: Path,
    output_extrinsics_path: Path,
) -> None:
    """Load samples.json, run hand–eye, and save T_body_iphone."""
    samples = _load_samples(samples_json_path)
    print(f"[INFO] Loaded {len(samples)} calibration samples from {samples_json_path}")
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples for calibration.")

    T_hand_iphone = _solve_hand_eye(samples)
    print("[INFO] Estimated T_hand_iphone (hand -> iPhone):")
    print(T_hand_iphone)

    T_body_iphone = _compute_T_body_iphone(samples, T_hand_iphone)
    print("[INFO] Estimated T_body_iphone (BODY -> iPhone):")
    print(T_body_iphone)

    output_extrinsics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_extrinsics_path, "w") as f:
        json.dump(
            {
                "T_hand_iphone": T_hand_iphone.tolist(),
                "T_body_iphone": T_body_iphone.tolist(),
            },
            f,
            indent=2,
        )

    print(f"[INFO] Saved extrinsics to {output_extrinsics_path}")

    # Visualize a few aligned point clouds in the BODY frame to verify calibration.
    samples_root = samples_json_path.parent
    _visualize_aligned_point_clouds(samples, T_body_iphone, samples_root)


def main() -> None:
    """Parse arguments and run data collection or calibration solving."""
    parser = argparse.ArgumentParser(
        description="Calibrate external iPhone camera w.r.t. Spot body frame."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Data collection subcommand
    collect_parser = subparsers.add_parser(
        "collect", help="Collect paired Spot/iPhone Charuco observations."
    )
    collect_parser.add_argument(
        "--hostname",
        type=str,
        required=True,
        help="Spot hostname/IP (e.g., 192.168.80.3)",
    )
    collect_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store captured samples.",
    )
    collect_parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of calibration samples to attempt.",
    )

    # Calibration subcommand
    calib_parser = subparsers.add_parser(
        "solve", help="Run calibration to estimate T_body_iphone from samples."
    )
    calib_parser.add_argument(
        "--samples_json",
        type=str,
        required=True,
        help="Path to samples.json produced by the 'collect' command.",
    )
    calib_parser.add_argument(
        "--output_extrinsics",
        type=str,
        required=True,
        help="Path to output JSON file containing T_body_iphone.",
    )

    args = parser.parse_args()

    if args.command == "collect":
        output_dir = Path(args.output_dir)
        robot = init_robot(args.hostname)
        collect_calibration_data(
            robot,
            output_dir=output_dir,
            num_samples=args.num_samples,
        )

    elif args.command == "solve":
        samples_json_path = Path(args.samples_json)
        output_extrinsics_path = Path(args.output_extrinsics)
        run_calibration(samples_json_path, output_extrinsics_path)

def get_point_cloud(dirpath: str, visualize: bool = False):
    """Build a point cloud from iPhone RGB-D images and intrinsics."""
    # dirpath = "/Users/aditya/research/phd/code/spot/see-spot-plan/wipe_online_images_iphone"
    # rgb_image_path = os.path.join(dirpath, "rgb_20251203_155935.png")
    # depth_image_path = os.path.join(dirpath, "depth_20251203_154258.npy")
    # intrinsics_path = os.path.join(dirpath, "intrinsics_20251203_155935.json")
    # K = np.array(json.load(open(intrinsics_path))["K"], dtype=np.float64)
    # rgb = cv2.cvtColor(cv2.imread(rgb_image_path), cv2.COLOR_BGR2RGB)
    # depth = np.load(depth_image_path)
    # points, colors = rgbd_to_point_cloud(rgb, depth, K)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])

    rgb_path = os.path.join(dirpath, "iphone_rgb.png")
    depth_path = os.path.join(dirpath, "iphone_depth.npy")
    intrinsics_path = os.path.join(dirpath, "iphone_intrinsics.json")
    K = np.array(json.load(open(intrinsics_path))["K"], dtype=np.float64)
    rgb_raw = cv2.imread(rgb_path)
    assert rgb_raw is not None, f"Failed to read image: {rgb_path}"
    rgb = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)
    depth = np.load(depth_path)
    points, colors = rgbd_to_point_cloud(rgb, depth, K)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if visualize:
        o3d.visualization.draw_geometries([pcd])

    return pcd

"""
Example usage: 
First collect data: 
python calibrate_iphone.py collect --hostname 192.168.80.3 --output_dir /home/ubuntu/calib_data --num_samples 10

Then run calibration:
python calibrate_iphone.py solve --samples_json /home/ubuntu/calib_data/samples.json --output_extrinsics /home/ubuntu/calib_data/T_body_iphone.json
"""

if __name__ == "__main__":
    rr.init("calibrate_iphone", spawn=True)
    main()