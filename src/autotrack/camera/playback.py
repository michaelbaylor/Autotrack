"""Hardware-out-of-the-loop playback camera for testing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .base import CameraConfig, PTZCamera

logger = logging.getLogger(__name__)


@dataclass
class CameraCommand:
    """A single recorded PTZ command, used for introspection in tests."""

    kind: str  # "move", "stop", "zoom"
    pan: float = 0.0
    tilt: float = 0.0
    zoom_value: int = 0


class PlaybackCamera(PTZCamera):
    """Simulated PTZ camera for hardware-out-of-the-loop testing.

    Reads frames from a local video file instead of a capture device.  All
    PTZ commands (move, stop, zoom) are no-ops; their arguments are recorded
    in :attr:`command_log` so tests can assert on camera behaviour without
    real hardware.

    Usage::

        cam = PlaybackCamera("flight.mp4")
        cam.connect()  # validates the file is readable
        source = cam.get_video_source()  # returns "flight.mp4"
    """

    def __init__(self, video_path: str, config: CameraConfig | None = None) -> None:
        """
        Args:
            video_path: Path to the video file to use as the frame source.
            config:     Optional camera config.  A minimal default config is
                        used when not provided.
        """
        if config is None:
            config = CameraConfig(ip="localhost")
        super().__init__(config)
        self.video_path = video_path
        self.command_log: list[CameraCommand] = []

    # ------------------------------------------------------------------
    # PTZCamera abstract methods
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Verify the video file is readable.

        Returns:
            ``True`` if ``cv2.VideoCapture`` can open the file, ``False``
            otherwise.
        """
        if not Path(self.video_path).exists():
            logger.error("Playback video not found: %s", self.video_path)
            return False

        cap = cv2.VideoCapture(self.video_path)
        ok = cap.isOpened()
        cap.release()

        if not ok:
            logger.error("cv2.VideoCapture could not open: %s", self.video_path)
        return ok

    def move(self, pan_speed: float, tilt_speed: float) -> None:
        """Record a move command (no-op)."""
        self.command_log.append(CameraCommand("move", pan=pan_speed, tilt=tilt_speed))

    def stop(self) -> None:
        """Record a stop command (no-op)."""
        self.command_log.append(CameraCommand("stop"))

    def zoom(self, zoom_value: int) -> None:
        """Record a zoom command (no-op)."""
        self.command_log.append(CameraCommand("zoom", zoom_value=zoom_value))

    def get_status(self) -> dict:
        """Return the current (simulated) camera state."""
        return {
            "pan": self.state.pan,
            "tilt": self.state.tilt,
            "zoom": self.state.zoom,
            "focus": self.state.focus,
        }

    def get_video_source(self) -> str | int:
        """Return the video file path."""
        return self.video_path
