"""Unit tests for PlaybackCamera."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autotrack.camera.base import CameraConfig
from autotrack.camera.playback import CameraCommand, PlaybackCamera


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_video(path: Path, frame_count: int = 5, width: int = 64, height: int = 48) -> Path:
    """Write a tiny synthetic MP4 file for testing."""
    writer = None
    try:
        fourcc = -1  # Let OpenCV pick; we'll check isOpened instead
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 30.0, (width, height))
        for _ in range(frame_count):
            writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    finally:
        if writer:
            writer.release()
    return path


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_config_when_none_provided(self):
        cam = PlaybackCamera("test.mp4")
        assert cam.config.ip == "localhost"

    def test_custom_config_is_used(self):
        cfg = CameraConfig(ip="1.2.3.4", pid_x_p=0.1)
        cam = PlaybackCamera("test.mp4", config=cfg)
        assert cam.config.ip == "1.2.3.4"
        assert cam.config.pid_x_p == pytest.approx(0.1)

    def test_video_path_stored(self):
        cam = PlaybackCamera("/some/path/video.mp4")
        assert cam.video_path == "/some/path/video.mp4"

    def test_command_log_starts_empty(self):
        cam = PlaybackCamera("test.mp4")
        assert cam.command_log == []


# ---------------------------------------------------------------------------
# connect()
# ---------------------------------------------------------------------------


class TestConnect:
    def test_returns_false_for_missing_file(self, tmp_path):
        cam = PlaybackCamera(str(tmp_path / "nonexistent.mp4"))
        assert cam.connect() is False

    def test_returns_true_for_valid_video(self, tmp_path):
        video = _make_video(tmp_path / "clip.mp4")
        cam = PlaybackCamera(str(video))
        assert cam.connect() is True

    def test_returns_false_when_opencv_cannot_open(self, tmp_path):
        # Create a file that exists but is not a valid video
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"not a video")
        cam = PlaybackCamera(str(bad))
        assert cam.connect() is False


# ---------------------------------------------------------------------------
# move() / stop() / zoom()
# ---------------------------------------------------------------------------


class TestCommands:
    def test_move_appends_to_log(self):
        cam = PlaybackCamera("test.mp4")
        cam.move(0.5, -0.3)
        assert len(cam.command_log) == 1
        cmd = cam.command_log[0]
        assert cmd.kind == "move"
        assert cmd.pan == pytest.approx(0.5)
        assert cmd.tilt == pytest.approx(-0.3)

    def test_stop_appends_to_log(self):
        cam = PlaybackCamera("test.mp4")
        cam.stop()
        assert len(cam.command_log) == 1
        assert cam.command_log[0].kind == "stop"

    def test_zoom_appends_to_log(self):
        cam = PlaybackCamera("test.mp4")
        cam.zoom(5000)
        assert len(cam.command_log) == 1
        cmd = cam.command_log[0]
        assert cmd.kind == "zoom"
        assert cmd.zoom_value == 5000

    def test_multiple_commands_ordered(self):
        cam = PlaybackCamera("test.mp4")
        cam.move(1.0, 0.0)
        cam.stop()
        cam.zoom(3000)
        kinds = [c.kind for c in cam.command_log]
        assert kinds == ["move", "stop", "zoom"]

    def test_commands_do_not_modify_state(self):
        cam = PlaybackCamera("test.mp4")
        cam.move(1.0, -1.0)
        cam.zoom(9999)
        # State should be unchanged (no hardware to update from)
        assert cam.state.pan == pytest.approx(0.0)
        assert cam.state.tilt == pytest.approx(0.0)
        assert cam.state.zoom == 0


# ---------------------------------------------------------------------------
# get_status()
# ---------------------------------------------------------------------------


class TestGetStatus:
    def test_returns_dict_with_expected_keys(self):
        cam = PlaybackCamera("test.mp4")
        status = cam.get_status()
        assert set(status.keys()) == {"pan", "tilt", "zoom", "focus"}

    def test_reflects_current_state(self):
        cam = PlaybackCamera("test.mp4")
        cam.state.pan = 12.5
        cam.state.tilt = -3.0
        status = cam.get_status()
        assert status["pan"] == pytest.approx(12.5)
        assert status["tilt"] == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# get_video_source()
# ---------------------------------------------------------------------------


class TestGetVideoSource:
    def test_returns_video_path_string(self):
        cam = PlaybackCamera("/path/to/video.mp4")
        assert cam.get_video_source() == "/path/to/video.mp4"

    def test_return_type_is_str(self):
        cam = PlaybackCamera("clip.mp4")
        assert isinstance(cam.get_video_source(), str)


# ---------------------------------------------------------------------------
# PID / tracking helpers still work (inherited from base)
# ---------------------------------------------------------------------------


class TestInheritedBehaviour:
    def test_track_target_centred_returns_zero(self):
        cam = PlaybackCamera("test.mp4")
        pan, tilt = cam.track_target(640, 360, 1280, 720)
        assert abs(pan) < 1e-6
        assert abs(tilt) < 1e-6

    def test_reset_pid_works(self):
        cam = PlaybackCamera("test.mp4")
        for _ in range(10):
            cam.track_target(900, 360, 1280, 720)
        cam.reset_pid()
        pan, tilt = cam.track_target(640, 360, 1280, 720)
        assert abs(pan) < 1e-6
        assert abs(tilt) < 1e-6
