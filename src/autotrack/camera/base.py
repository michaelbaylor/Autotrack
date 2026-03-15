"""Abstract PTZ camera base class with PID-based target tracking."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import yaml
from simple_pid import PID


@dataclass
class CameraConfig:
    """Configuration for a PTZ camera loaded from a YAML file."""

    ip: str
    port: int = 80
    username: str = ""
    password: str = ""
    pan_speed_max: float = 1.0
    tilt_speed_max: float = 1.0
    zoom_min: int = 0
    zoom_max: int = 9999
    focus_min: int = 0
    focus_max: int = 9999
    pid_x_p: float = 0.03
    pid_x_i: float = 0.01
    pid_x_d: float = 0.01
    pid_y_p: float = 0.03
    pid_y_i: float = 0.01
    pid_y_d: float = 0.01
    zoom_coefficient: float = 300.0
    # Edelkrone wireless link IDs (assigned by the Edelkrone hub)
    ptz_link_id: str = ""
    focus_link_id: str = ""
    # Edelkrone module MAC addresses (printed on each device)
    head_module_mac: str = ""
    zoom_module_mac: str = ""
    focus_module_mac: str = ""
    # Local capture device: integer index (0, 1, …) or path (/dev/video0)
    video_device: str = "0"

    @classmethod
    def from_yaml(cls, path: str) -> "CameraConfig":
        """Load a :class:`CameraConfig` from a YAML file.

        Only keys that correspond to known dataclass fields are read; unknown
        keys (e.g. ``type``) are silently ignored.

        Args:
            path: Filesystem path to the YAML configuration file.

        Returns:
            A populated :class:`CameraConfig` instance.
        """
        with open(path, "r") as fh:
            raw: dict = yaml.safe_load(fh) or {}

        known_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class CameraState:
    """Mutable runtime state of the camera."""

    pan: float = 0.0
    tilt: float = 0.0
    zoom: int = 0
    focus: int = 0
    iris: int = 0
    is_tracking: bool = False
    track_id: Optional[int] = None


class PTZCamera(ABC):
    """Abstract base class for pan-tilt-zoom cameras.

    Provides PID-based target tracking on top of the concrete camera
    implementation.  Subclasses must implement :meth:`connect`,
    :meth:`move`, :meth:`stop`, :meth:`zoom`, and :meth:`get_status`.
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.state = CameraState()
        self.ping_ms: float = 0.0

        self._pid_x = PID(
            config.pid_x_p,
            config.pid_x_i,
            config.pid_x_d,
            setpoint=0,
            output_limits=(-1.0, 1.0),
        )
        self._pid_y = PID(
            config.pid_y_p,
            config.pid_y_i,
            config.pid_y_d,
            setpoint=0,
            output_limits=(-1.0, 1.0),
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the camera.

        Returns:
            ``True`` on success, ``False`` otherwise.
        """

    @abstractmethod
    def move(self, pan_speed: float, tilt_speed: float) -> None:
        """Send a continuous pan/tilt move command.

        Args:
            pan_speed:  Normalised speed in ``[-1.0, 1.0]``; positive = right.
            tilt_speed: Normalised speed in ``[-1.0, 1.0]``; positive = down.
        """

    @abstractmethod
    def stop(self) -> None:
        """Halt all camera motion."""

    @abstractmethod
    def zoom(self, zoom_value: int) -> None:
        """Set absolute zoom position.

        Args:
            zoom_value: Target zoom position within ``[zoom_min, zoom_max]``.
        """

    @abstractmethod
    def get_status(self) -> dict:
        """Poll and return current camera state as a dictionary.

        Implementations should populate :attr:`state` and return a dict
        containing at least ``pan``, ``tilt``, ``zoom``, and ``focus`` keys.
        """

    @abstractmethod
    def get_video_source(self) -> str | int:
        """Return the video source to pass to ``cv2.VideoCapture``.

        Returns either an RTSP/file URL string or an integer device index for
        local capture hardware (e.g. HDMI capture cards).
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def track_target(
        self,
        target_x: float,
        target_y: float,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float]:
        """Calculate PID pan/tilt speeds to move the target to frame centre.

        The error is normalised to the range ``[-0.5, 0.5]`` so the PID gains
        are frame-size independent.

        Args:
            target_x: Horizontal pixel position of the target.
            target_y: Vertical pixel position of the target.
            frame_width:  Width of the video frame in pixels.
            frame_height: Height of the video frame in pixels.

        Returns:
            A ``(pan_speed, tilt_speed)`` tuple of PID outputs in ``[-1, 1]``.
        """
        # Normalised error: positive when target is right of / below centre.
        # simple_pid computes output = Kp * (setpoint - input), so we negate
        # the input to get positive output for positive error.
        x_error = (target_x - frame_width / 2) / frame_width
        y_error = (target_y - frame_height / 2) / frame_height

        pan_speed = self._pid_x(-x_error)
        tilt_speed = self._pid_y(-y_error)

        return float(pan_speed), float(tilt_speed)

    def calculate_zoom(self, bbox_area: int, frame_area: int) -> int:
        """Compute a target zoom position based on how large the target is.

        A target that occupies a small fraction of the frame drives a higher
        zoom value, proportional to ``zoom_coefficient``.

        Args:
            bbox_area:  Bounding box area in pixels.
            frame_area: Total frame area in pixels.

        Returns:
            Clamped zoom value within ``[zoom_min, zoom_max]``.
        """
        if bbox_area <= 0 or frame_area <= 0:
            return self.config.zoom_min

        ratio = bbox_area / frame_area
        # Smaller target → higher zoom. The formula is heuristic.
        target_zoom = int(self.config.zoom_coefficient / max(ratio, 1e-6))
        return int(
            max(self.config.zoom_min, min(self.config.zoom_max, target_zoom))
        )

    def reset_pid(self) -> None:
        """Reset both PID controllers, clearing integral and derivative state."""
        self._pid_x.reset()
        self._pid_y.reset()
