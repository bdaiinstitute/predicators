"""Kiwi iPhone RGBD receiver usable as a library.

This module implements a small TCP server that receives ARKit frame bundles
from the Kiwi iOS app via Protocol Buffers (FrameBundle proto), and exposes
an easy-to-use Python API:

    receiver = KiwiReceiver(host="0.0.0.0", port=8888)
    frame = receiver.recv_frame()

`frame` contains:
  - rgb:       HxWx3 uint8 RGB image
  - depth:     HxW float32 depth array, or None if not provided
  - intrinsics: 3x3 float32 camera intrinsics matrix
  - transform:  4x4 float32 camera pose matrix (ARKit world_T_camera)

This is a refactoring of your original standalone receiver script into a
reusable module that can be imported by `calibrate_iphone.py` (or others).
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image

from predicators.spot_utils.skills.frame_bundle_pb2 import FrameBundle


@dataclass
class IphoneRGBDFrame:
    """Single RGBD frame + intrinsics and transform from Kiwi."""

    rgb: np.ndarray  # HxWx3 uint8, RGB
    depth: Optional[np.ndarray]  # HxW float32, meters (or consistent units), or None
    intrinsics: np.ndarray  # 3x3 float32
    transform: np.ndarray  # 4x4 float32 (world_T_camera from ARKit)
    frame_number: int


class KiwiReceiver:
    """TCP server that receives ARKit frames from the Kiwi iOS app.

    Usage:
        receiver = KiwiReceiver(host="0.0.0.0", port=8888)
        frame = receiver.recv_frame()  # blocks until one frame is received
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8888) -> None:
        """Initialize the TCP server and accept a connection from the iPhone."""
        self._host = host
        self._port = port
        self._server_sock: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._frame_count: int = 0

        self._start_server()

    def _start_server(self) -> None:
        """Bind, listen, and accept a single connection from the iPhone."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self._host, self._port))
        server_sock.listen(1)

        print("=" * 60)
        print("Kiwi Frame Receiver (TCP + Protobuf)")
        print("=" * 60)
        print(f"Listening on {self._host}:{self._port}")
        print(f"Configure iPhone to send to: {get_local_ip()}:{self._port}")
        print("Waiting for connection...\n")

        conn, addr = server_sock.accept()
        print(f"Connected from {addr[0]}:{addr[1]}\n")

        self._server_sock = server_sock
        self._conn = conn

    def close(self) -> None:
        """Close sockets."""
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None

    def recv_frame(self) -> IphoneRGBDFrame:
        """Blocking receive of a single FrameBundle from the connected iPhone."""
        if self._conn is None:
            raise RuntimeError("KiwiReceiver has no active connection.")

        # 1) Read length prefix (4 bytes, big-endian)
        length_data = recv_exact(self._conn, 4)
        if not length_data:
            raise RuntimeError("Connection closed while reading length prefix.")
        length = struct.unpack(">I", length_data)[0]

        # 2) Read Protobuf payload
        protobuf_data = recv_exact(self._conn, length)
        if not protobuf_data:
            raise RuntimeError("Connection closed while reading protobuf payload.")

        # 3) Decode Protobuf
        frame_proto = FrameBundle()
        frame_proto.ParseFromString(protobuf_data)

        self._frame_count += 1

        # 4) Decode RGB
        rgb_data = frame_proto.rgb_image_data
        rgb_image = Image.open(BytesIO(rgb_data))
        rgb = np.array(rgb_image)  # HxWx3 uint8, RGB

        # 5) Decode depth if present
        depth = None
        if frame_proto.depth_data:
            depth_data = frame_proto.depth_data
            depth_width = frame_proto.depth_width
            depth_height = frame_proto.depth_height

            depth = np.frombuffer(depth_data, dtype=np.float32)
            depth = depth.reshape((depth_height, depth_width))

        # 6) Transform (ARKit uses column-major)
        transform = np.array(frame_proto.transform, dtype=np.float32).reshape(4, 4).T

        # 7) Intrinsics (3x3) - also transposed like transform
        intrinsics = np.array(frame_proto.intrinsics, dtype=np.float32).reshape(3, 3).T

        return IphoneRGBDFrame(
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            transform=transform,
            frame_number=frame_proto.frame_number,
        )


def recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    """Receive exactly num_bytes from socket (TCP requires this)."""
    data = b""
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            return b""
        data += chunk
    return data


def get_local_ip() -> str:
    """Get local IP address for display/logging purposes."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:  # noqa: BLE001
        return "127.0.0.1"


__all__ = ["KiwiReceiver", "IphoneRGBDFrame"]

