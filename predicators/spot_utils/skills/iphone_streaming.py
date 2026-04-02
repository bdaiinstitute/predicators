"""Persistent iPhone streaming receiver manager with inter-process support.

This module provides a streaming receiver that can persist across multiple
script executions by using file-based inter-process communication.

Usage from command line:
    python iphone_streaming.py start    # Start streaming (keeps running)
    python iphone_streaming.py stop     # Stop streaming
    python iphone_streaming.py status   # Check if streaming is active

Usage in skills:
    from iphone_streaming import get_latest_frame
    frame = get_latest_frame()  # Returns the most recent IphoneFrame

Or use the convenience wrapper:
    from iphone_streaming import StreamingManager
    with StreamingManager():  # Ensures streaming is active
        frame = get_latest_frame()
"""

import argparse
import atexit
import os
import pickle
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from predicators.spot_utils.skills.calibrate_iphone import IphoneFrame, ThreadedKiwiReceiver

# File paths for inter-process communication
_STREAM_DIR = Path("/tmp/iphone_stream")
_FRAME_FILE = _STREAM_DIR / "latest_frame.pkl"
_PID_FILE = _STREAM_DIR / "streaming.pid"
_LOCK_FILE = _STREAM_DIR / "frame.lock"

# In-process receiver (only used by the streaming process)
_receiver: Optional[ThreadedKiwiReceiver] = None
_keep_running = True


def _ensure_stream_dir():
    """Ensure the streaming directory exists."""
    _STREAM_DIR.mkdir(parents=True, exist_ok=True)


def is_streaming_active() -> bool:
    """Check if streaming is currently active in another process."""
    if not _PID_FILE.exists():
        return False

    try:
        with open(_PID_FILE, "r") as f:
            pid = int(f.read().strip())

        # Check if the process is still running
        os.kill(pid, 0)  # Doesn't actually kill, just checks if PID exists
        return True
    except (OSError, ValueError, ProcessLookupError):
        # PID file exists but process is dead, clean up
        _PID_FILE.unlink(missing_ok=True)
        return False


def start_streaming() -> None:
    """Start the iPhone streaming receiver and keep it running.

    This function blocks and continuously updates the latest frame file.
    It should be run in its own process (e.g., via `python iphone_streaming.py start`).
    """
    global _receiver, _keep_running

    _ensure_stream_dir()

    if is_streaming_active():
        print("[ERROR] Streaming is already active in another process")
        sys.exit(1)

    # Write our PID
    with open(_PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    print("[INFO] Starting iPhone streaming receiver...")
    _receiver = ThreadedKiwiReceiver()
    print("[INFO] iPhone streaming started successfully")
    print(f"[INFO] Frame data will be written to: {_FRAME_FILE}")
    print("\nPress Ctrl+C to stop streaming...")

    def signal_handler(sig, frame_):
        global _keep_running
        print("\n[INFO] Received interrupt signal, stopping...")
        _keep_running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Cleanup function
    def cleanup():
        stop_streaming()

    atexit.register(cleanup)

    # Continuously update the frame file
    try:
        while _keep_running:
            frame = _receiver.get_latest_frame()
            if frame is not None:
                # Write frame with basic file locking
                temp_file = _FRAME_FILE.with_suffix(".tmp")
                with open(temp_file, "wb") as f:
                    pickle.dump(frame, f)
                temp_file.replace(_FRAME_FILE)  # Atomic operation

            time.sleep(0.01)  # Update at ~100 Hz
    finally:
        cleanup()


def stop_streaming() -> None:
    """Stop the iPhone streaming receiver and clean up resources."""
    global _receiver

    if _receiver is not None:
        print("[INFO] Stopping iPhone streaming receiver...")
        _receiver.stop()
        _receiver = None

    # Clean up files
    _PID_FILE.unlink(missing_ok=True)
    _FRAME_FILE.unlink(missing_ok=True)
    print("[INFO] iPhone streaming stopped successfully")


def get_latest_frame(timeout: float = 5.0, max_age: float = 2.0) -> IphoneFrame:
    """Get the latest frame from the streaming receiver.

    This reads from the shared frame file updated by the streaming process.

    Args:
        timeout: Maximum time to wait for a frame if streaming just started
        max_age: Maximum age of frame in seconds before it's considered stale

    Returns:
        The latest IphoneFrame

    Raises:
        RuntimeError: If streaming is not active or no frame is available

    """
    if not is_streaming_active():
        raise RuntimeError(
            "iPhone streaming is not active. Start it with:\n"
            "  python iphone_streaming.py start"
        )

    # Wait for frame file to appear (in case streaming just started)
    start_time = time.time()
    while not _FRAME_FILE.exists():
        if time.time() - start_time > timeout:
            raise RuntimeError(
                f"No frame file found after {timeout}s. "
                "Streaming may have just started, try again."
            )
        time.sleep(0.1)

    # Check frame freshness
    frame_age = time.time() - _FRAME_FILE.stat().st_mtime
    if frame_age > max_age:
        raise RuntimeError(
            f"Frame is {frame_age:.1f}s old (max allowed: {max_age}s). "
            "The iPhone may have stopped streaming. Restart with:\n"
            "  python iphone_streaming.py stop\n"
            "  python iphone_streaming.py start"
        )

    # Read the latest frame
    try:
        with open(_FRAME_FILE, "rb") as f:
            frame = pickle.load(f)
        return frame
    except Exception as e:
        raise RuntimeError(f"Failed to read frame file: {e}")


class StreamingManager:
    """Context manager to ensure streaming is active and provide helpful errors."""

    def __enter__(self):
        """Enter the context manager and verify streaming is active."""
        if not is_streaming_active():
            print(
                "[ERROR] iPhone streaming is not active!\n"
                "[ERROR] Start streaming in a separate terminal with:\n"
                "[ERROR]   python iphone_streaming.py start"
            )
            raise RuntimeError("iPhone streaming not active")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        pass  # Don't stop streaming on exit


def main() -> None:
    """Command-line interface for managing iPhone streaming."""
    parser = argparse.ArgumentParser(
        description="Manage iPhone streaming receiver"
    )
    parser.add_argument(
        "command",
        choices=["start", "stop", "status"],
        help="Command to execute",
    )
    args = parser.parse_args()

    if args.command == "start":
        start_streaming()

    elif args.command == "stop":
        if is_streaming_active():
            try:
                with open(_PID_FILE, "r") as f:
                    pid = int(f.read().strip())
                print(f"[INFO] Sending stop signal to process {pid}...")
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                if not is_streaming_active():
                    print("[INFO] Streaming stopped successfully")
                else:
                    print("[WARN] Process may still be running")
            except Exception as e:
                print(f"[ERROR] Failed to stop streaming: {e}")
                sys.exit(1)
        else:
            print("[INFO] No active streaming process found")

    elif args.command == "status":
        if is_streaming_active():
            try:
                with open(_PID_FILE, "r") as f:
                    pid = int(f.read().strip())
                print(f"[INFO] iPhone streaming is ACTIVE (PID: {pid})")
                if _FRAME_FILE.exists():
                    age = time.time() - _FRAME_FILE.stat().st_mtime
                    print(f"[INFO] Latest frame is {age:.2f}s old")
            except Exception as e:
                print(f"[INFO] Streaming active but could not read details: {e}")
        else:
            print("[INFO] iPhone streaming is NOT ACTIVE")


if __name__ == "__main__":
    main()
