#!/usr/bin/env python3
"""Autotrack - Automated plane tracking for PTZ cameras."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

import cv2
import numpy as np
import pygame

from .camera.base import CameraConfig
from .camera.edelkrone import EdelkroneCamera
from .camera.playback import PlaybackCamera
from .detector import PlaneDetector
from .tracker_manager import Track, TrackerManager
from .utils.recorder import Recorder, TelemetryFrame
from .utils.thread_worker import ThreadWorker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autotrack – automated plane tracking for PTZ cameras"
    )
    parser.add_argument(
        "device_config",
        nargs="?",
        default=None,
        help="Path to device YAML config file (not required with --playback)",
    )
    parser.add_argument(
        "--playback",
        metavar="VIDEO",
        default=None,
        help="Hardware-out-of-the-loop mode: run detection on a video file instead of a live camera",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model path or name (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Detection confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack",
        choices=["strongsort", "bytetrack", "botsort"],
        help="boxmot tracker algorithm (default: bytetrack)",
    )
    parser.add_argument(
        "--output-dir",
        default="recordings",
        help="Directory for video and CSV output (default: recordings)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Inference device: "auto" (default), "mps", "cuda", "cpu"',
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a graphical display",
    )

    args = parser.parse_args()
    if args.playback is None and args.device_config is None:
        parser.error("device_config is required unless --playback is specified")
    return args


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


class AutoTrack:
    """Main autotrack application.

    Connects to an Edelkrone camera head (or plays back a video file in
    hardware-out-of-the-loop mode), runs YOLO plane detection, tracks
    detections with boxmot, and moves the camera to keep the selected plane
    centred in the frame.
    """

    TARGET_FPS = 30
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 720

    def __init__(
        self,
        config_path: Optional[str] = None,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.4,
        tracker_type: str = "bytetrack",
        output_dir: str = "recordings",
        headless: bool = False,
        playback_video: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        if playback_video:
            self.config = CameraConfig(ip="localhost")
            self.camera = PlaybackCamera(playback_video)
        else:
            if config_path is None:
                raise ValueError("config_path is required when not using playback mode")
            self.config = CameraConfig.from_yaml(config_path)
            self.camera = EdelkroneCamera(self.config)
        self.detector = PlaneDetector(model_path, confidence, device=device)
        self.tracker = TrackerManager(tracker_type, device=device)
        self.recorder = Recorder(output_dir)
        self.headless = headless
        # Camera commands run in a background thread so HTTP latency doesn't
        # block the inference loop between frames.
        self._camera_worker = ThreadWorker()

        self._running = False
        self._selected_track_id: Optional[int] = None
        self._last_fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Connect to the camera and enter the main tracking loop."""
        if not self.camera.connect():
            raise RuntimeError("Failed to connect to camera")

        source = self.camera.get_video_source()
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        screen: Optional[pygame.Surface] = None
        if not self.headless:
            pygame.init()
            screen = pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
            pygame.display.set_caption("Autotrack")

        self._running = True
        _t = time.perf_counter  # local alias for brevity
        _step_totals: dict[str, float] = {
            "read": 0.0, "detect": 0.0, "track": 0.0, "display": 0.0
        }
        _profile_timer = time.time()
        _profile_frames = 0

        try:
            while self._running:
                t0 = _t()
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from stream; stopping.")
                    break
                _step_totals["read"] += _t() - t0

                t0 = _t()
                detections = self.detector.detect(frame)
                _step_totals["detect"] += _t() - t0

                t0 = _t()
                tracks = self.tracker.update(detections, frame)
                _step_totals["track"] += _t() - t0

                # Auto-select first track when none is selected
                if self._selected_track_id is None and tracks:
                    self._selected_track_id = tracks[0].track_id

                # Find the actively selected track
                active_track: Optional[Track] = next(
                    (t for t in tracks if t.track_id == self._selected_track_id), None
                )

                # Move camera toward target (dispatched to background thread
                # so the Edelkrone HTTP call doesn't stall the inference loop)
                if active_track:
                    cx, cy = active_track.center
                    h, w = frame.shape[:2]
                    pan_speed, tilt_speed = self.camera.track_target(cx, cy, w, h)
                    self._camera_worker.submit(self.camera.move, pan_speed, tilt_speed)
                else:
                    self._camera_worker.submit(self.camera.stop)
                    if tracks:
                        self._selected_track_id = tracks[0].track_id
                    else:
                        self._selected_track_id = None

                # Update FPS counter
                _profile_frames += 1
                self._frame_count += 1
                elapsed = time.time() - self._fps_timer
                if elapsed >= 1.0:
                    self._last_fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._fps_timer = time.time()

                t0 = _t()
                display_frame = self._draw_overlay(frame, tracks, active_track)
                if self.recorder.is_recording:
                    self.recorder.write_frame(display_frame)
                    self.recorder.log_telemetry(self._build_telemetry(active_track))
                if not self.headless and screen is not None:
                    self._handle_events(display_frame)
                    self._render_frame(screen, display_frame)
                _step_totals["display"] += _t() - t0

                # Log per-step timing breakdown every 5 seconds
                profile_elapsed = time.time() - _profile_timer
                if profile_elapsed >= 5.0 and _profile_frames > 0:
                    n = _profile_frames
                    logger.info(
                        "Timing per frame (ms) over %d frames — "
                        "read: %.1f  detect: %.1f  track: %.1f  display: %.1f  "
                        "total: %.1f  FPS: %.1f",
                        n,
                        _step_totals["read"]    / n * 1000,
                        _step_totals["detect"]  / n * 1000,
                        _step_totals["track"]   / n * 1000,
                        _step_totals["display"] / n * 1000,
                        sum(_step_totals.values()) / n * 1000,
                        n / profile_elapsed,
                    )
                    _step_totals = {k: 0.0 for k in _step_totals}
                    _profile_frames = 0
                    _profile_timer = time.time()

        finally:
            cap.release()
            self.camera.stop()
            if self.recorder.is_recording:
                self.recorder.stop()
            if not self.headless:
                pygame.quit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_overlay(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        active_track: Optional[Track],
    ) -> np.ndarray:
        """Draw bounding boxes, track IDs, and status text onto a copy of *frame*."""
        display = frame.copy()

        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            is_active = track.track_id == self._selected_track_id
            color = (0, 255, 0) if is_active else (255, 165, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{track.track_id} {track.confidence:.2f}"
            cv2.putText(
                display, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

        # Crosshair on active target
        if active_track:
            cx, cy = active_track.center
            cv2.drawMarker(display, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        # Status overlay (top-left)
        status_lines = [
            f"FPS: {self._last_fps:.1f}",
            f"Tracks: {len(tracks)}",
            f"Recording: {'ON' if self.recorder.is_recording else 'OFF'}",
        ]
        for i, line in enumerate(status_lines):
            cv2.putText(
                display, line, (10, 25 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return display

    def _handle_events(self, display_frame: np.ndarray) -> None:
        """Process pygame keyboard and window events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self._running = False

                elif event.key == pygame.K_r:
                    h, w = display_frame.shape[:2]
                    self.recorder.toggle(w, h)

                elif event.key == pygame.K_SPACE:
                    # Deselect current target and reset PID
                    self._selected_track_id = None
                    self.camera.reset_pid()

    def _render_frame(self, screen: pygame.Surface, frame: np.ndarray) -> None:
        """Convert an OpenCV BGR frame and blit it to the pygame display."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        surface = pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1))
        screen.blit(surface, (0, 0))
        pygame.display.flip()

    def _build_telemetry(self, active_track: Optional[Track]) -> TelemetryFrame:
        """Construct a :class:`TelemetryFrame` from the current application state."""
        return TelemetryFrame(
            timestamp=time.time(),
            fps=self._last_fps,
            ping_ms=0.0,
            pan=self.camera.state.pan,
            tilt=self.camera.state.tilt,
            zoom=self.camera.state.zoom,
            is_tracking=active_track is not None,
            track_id=active_track.track_id if active_track else None,
            target_x=float(active_track.center[0]) if active_track else None,
            target_y=float(active_track.center[1]) if active_track else None,
            confidence=active_track.confidence if active_track else None,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    app = AutoTrack(
        config_path=args.device_config,
        model_path=args.model,
        confidence=args.confidence,
        tracker_type=args.tracker,
        output_dir=args.output_dir,
        headless=args.headless,
        playback_video=args.playback,
        device=args.device,
    )
    app.run()


if __name__ == "__main__":
    main()
