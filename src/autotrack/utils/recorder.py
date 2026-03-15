"""Video recording and CSV telemetry logging."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TelemetryFrame:
    """One row of telemetry data captured alongside a video frame."""

    timestamp: float
    fps: float
    ping_ms: float
    pan: float
    tilt: float
    zoom: int
    is_tracking: bool
    track_id: Optional[int]
    target_x: Optional[float]
    target_y: Optional[float]
    confidence: Optional[float]


class Recorder:
    """Records video frames to MP4 and telemetry data to CSV.

    Usage::

        rec = Recorder("recordings")
        rec.start(1280, 720)
        rec.write_frame(frame)
        rec.log_telemetry(telemetry_frame)
        path = rec.stop()
    """

    def __init__(self, output_dir: str = "recordings", fps: float = 30.0) -> None:
        self._output_dir = Path(output_dir)
        self._fps = fps

        self._video_writer: Optional[cv2.VideoWriter] = None
        self._csv_path: Optional[Path] = None
        self._video_path: Optional[Path] = None
        self._telemetry_rows: list[TelemetryFrame] = []
        self._recording = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        """``True`` when a recording session is active."""
        return self._recording

    def start(
        self,
        frame_width: int,
        frame_height: int,
        session_name: str = "",
    ) -> None:
        """Begin a new recording session.

        Creates timestamped ``.mp4`` and ``.csv`` files in *output_dir*.

        Args:
            frame_width:  Width of video frames in pixels.
            frame_height: Height of video frames in pixels.
            session_name: Optional label prepended to the filename.
        """
        if self._recording:
            logger.warning("Recorder.start() called while already recording; ignoring.")
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{session_name}_{timestamp}" if session_name else timestamp

        self._video_path = self._output_dir / f"{prefix}.mp4"
        self._csv_path = self._output_dir / f"{prefix}.csv"

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(
            str(self._video_path), fourcc, self._fps, (frame_width, frame_height)
        )

        if not self._video_writer.isOpened():
            self._video_writer = None
            raise RuntimeError(f"Failed to open video writer for {self._video_path}")

        self._telemetry_rows = []
        self._recording = True
        logger.info("Recording started: %s", self._video_path)

    def stop(self) -> Optional[Path]:
        """Finalise the recording session and write the CSV file.

        Returns:
            The path to the video file, or ``None`` if no recording was active.
        """
        if not self._recording:
            logger.warning("Recorder.stop() called while not recording; ignoring.")
            return None

        self._recording = False

        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        if self._csv_path is not None and self._telemetry_rows:
            self._write_csv()

        path = self._video_path
        logger.info("Recording stopped: %s", path)
        return path

    def write_frame(self, frame: np.ndarray) -> None:
        """Write one video frame to the current recording.

        Silently ignored when not recording.

        Args:
            frame: BGR image array (H x W x 3).
        """
        if not self._recording or self._video_writer is None:
            return
        self._video_writer.write(frame)

    def log_telemetry(self, telemetry: TelemetryFrame) -> None:
        """Append one telemetry record to the in-memory buffer.

        Records are written to CSV only when :meth:`stop` is called.

        Args:
            telemetry: Telemetry snapshot for the current frame.
        """
        if not self._recording:
            return
        self._telemetry_rows.append(telemetry)

    def toggle(self, frame_width: int, frame_height: int) -> bool:
        """Start recording if stopped; stop if recording.

        Args:
            frame_width:  Width of video frames in pixels.
            frame_height: Height of video frames in pixels.

        Returns:
            The new :attr:`is_recording` state.
        """
        if self._recording:
            self.stop()
        else:
            self.start(frame_width, frame_height)
        return self._recording

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_csv(self) -> None:
        """Write buffered telemetry rows to the CSV file."""
        column_names = [f.name for f in fields(TelemetryFrame)]
        with open(self._csv_path, "w", newline="") as fh:  # type: ignore[arg-type]
            writer = csv.DictWriter(fh, fieldnames=column_names)
            writer.writeheader()
            for row in self._telemetry_rows:
                writer.writerow(
                    {f.name: getattr(row, f.name) for f in fields(TelemetryFrame)}
                )
        logger.info("Telemetry written: %s (%d rows)", self._csv_path, len(self._telemetry_rows))
