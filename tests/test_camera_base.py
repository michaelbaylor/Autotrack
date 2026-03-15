"""Unit tests for CameraConfig, CameraState, and PTZCamera base class."""

from __future__ import annotations

from typing import Optional

import pytest
import yaml

from autotrack.camera.base import CameraConfig, CameraState, PTZCamera


# ---------------------------------------------------------------------------
# Concrete test double for PTZCamera
# ---------------------------------------------------------------------------


class FakeCamera(PTZCamera):
    """Minimal concrete implementation of PTZCamera for testing."""

    def __init__(self, config: CameraConfig) -> None:
        super().__init__(config)
        self.connected = False
        self.last_move: Optional[tuple[float, float]] = None
        self.last_zoom: Optional[int] = None
        self.stopped = False

    def connect(self) -> bool:
        self.connected = True
        return True

    def move(self, pan_speed: float, tilt_speed: float) -> None:
        self.last_move = (pan_speed, tilt_speed)

    def stop(self) -> None:
        self.stopped = True

    def zoom(self, zoom_value: int) -> None:
        self.last_zoom = zoom_value

    def get_status(self) -> dict:
        return {
            "pan": self.state.pan,
            "tilt": self.state.tilt,
            "zoom": self.state.zoom,
            "focus": self.state.focus,
        }

    def get_video_source(self) -> str | int:
        return 0


# ---------------------------------------------------------------------------
# CameraConfig
# ---------------------------------------------------------------------------


class TestCameraConfig:
    def test_from_yaml_loads_all_known_fields(self, tmp_path, camera_config_dict):
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(yaml.dump(camera_config_dict))
        cfg = CameraConfig.from_yaml(str(config_file))

        assert cfg.ip == "192.168.1.100"
        assert cfg.port == 80
        assert cfg.username == "root"
        assert cfg.password == "pass"
        assert cfg.pan_speed_max == pytest.approx(1.0)
        assert cfg.tilt_speed_max == pytest.approx(1.0)
        assert cfg.zoom_min == 0
        assert cfg.zoom_max == 9999
        assert cfg.pid_x_p == pytest.approx(0.03)
        assert cfg.pid_y_d == pytest.approx(0.005)
        assert cfg.zoom_coefficient == pytest.approx(300.0)

    def test_from_yaml_ignores_unknown_keys(self, tmp_path, camera_config_dict):
        camera_config_dict["unknown_field"] = "should_be_ignored"
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(yaml.dump(camera_config_dict))
        # Should not raise
        cfg = CameraConfig.from_yaml(str(config_file))
        assert not hasattr(cfg, "unknown_field")

    def test_from_yaml_uses_defaults_for_missing_fields(self, tmp_path):
        minimal = {"ip": "10.0.0.1"}
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(yaml.dump(minimal))
        cfg = CameraConfig.from_yaml(str(config_file))

        assert cfg.ip == "10.0.0.1"
        assert cfg.port == 80  # default
        assert cfg.username == ""  # default

    def test_from_yaml_video_device_default(self, tmp_path):
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(yaml.dump({"ip": "1.2.3.4"}))
        cfg = CameraConfig.from_yaml(str(config_file))
        assert cfg.video_device == "0"

    def test_from_yaml_custom_pid_gains(self, tmp_path):
        data = {"ip": "1.2.3.4", "pid_x_p": 0.1, "pid_y_p": 0.2}
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(yaml.dump(data))
        cfg = CameraConfig.from_yaml(str(config_file))
        assert cfg.pid_x_p == pytest.approx(0.1)
        assert cfg.pid_y_p == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# CameraState
# ---------------------------------------------------------------------------


class TestCameraState:
    def test_default_values(self):
        state = CameraState()
        assert state.pan == pytest.approx(0.0)
        assert state.tilt == pytest.approx(0.0)
        assert state.zoom == 0
        assert state.focus == 0
        assert state.iris == 0
        assert state.is_tracking is False
        assert state.track_id is None

    def test_mutation(self):
        state = CameraState()
        state.pan = 45.0
        state.is_tracking = True
        state.track_id = 7
        assert state.pan == pytest.approx(45.0)
        assert state.is_tracking is True
        assert state.track_id == 7


# ---------------------------------------------------------------------------
# PTZCamera (via FakeCamera)
# ---------------------------------------------------------------------------


class TestPTZCamera:
    def test_track_target_returns_tuple(self, camera_config):
        cam = FakeCamera(camera_config)
        result = cam.track_target(640, 360, 1280, 720)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_track_target_centered_returns_near_zero(self, camera_config):
        """When target is exactly at frame centre, PID error is 0 and output ≈ 0."""
        cam = FakeCamera(camera_config)
        pan, tilt = cam.track_target(640, 360, 1280, 720)
        assert abs(pan) < 1e-6
        assert abs(tilt) < 1e-6

    def test_track_target_target_right_of_centre_positive_pan(self, camera_config):
        """Target to the right should produce a positive pan speed."""
        cam = FakeCamera(camera_config)
        pan, _ = cam.track_target(900, 360, 1280, 720)
        assert pan > 0

    def test_track_target_target_left_of_centre_negative_pan(self, camera_config):
        cam = FakeCamera(camera_config)
        pan, _ = cam.track_target(100, 360, 1280, 720)
        assert pan < 0

    def test_track_target_target_below_centre_positive_tilt(self, camera_config):
        cam = FakeCamera(camera_config)
        _, tilt = cam.track_target(640, 600, 1280, 720)
        assert tilt > 0

    def test_track_target_target_above_centre_negative_tilt(self, camera_config):
        cam = FakeCamera(camera_config)
        _, tilt = cam.track_target(640, 100, 1280, 720)
        assert tilt < 0

    def test_reset_pid_zeroes_integral(self, camera_config):
        """After driving integral state then resetting, output at setpoint should be 0."""
        cam = FakeCamera(camera_config)
        # Drive integral accumulation
        for _ in range(20):
            cam.track_target(900, 360, 1280, 720)
        cam.reset_pid()
        pan, tilt = cam.track_target(640, 360, 1280, 720)
        assert abs(pan) < 1e-6
        assert abs(tilt) < 1e-6

    def test_calculate_zoom_small_target_high_zoom(self, camera_config):
        """A small target (1% of frame) should produce a high zoom value."""
        cam = FakeCamera(camera_config)
        frame_area = 1280 * 720
        small_bbox_area = int(frame_area * 0.01)
        zoom_val = cam.calculate_zoom(small_bbox_area, frame_area)
        assert zoom_val > 1000

    def test_calculate_zoom_large_target_low_zoom(self, camera_config):
        """A large target (50% of frame) should produce a lower zoom value."""
        cam = FakeCamera(camera_config)
        frame_area = 1280 * 720
        large_bbox_area = int(frame_area * 0.5)
        zoom_val = cam.calculate_zoom(large_bbox_area, frame_area)
        assert zoom_val < 1000

    def test_calculate_zoom_clamped_to_zoom_max(self, camera_config):
        """Result should never exceed zoom_max."""
        cam = FakeCamera(camera_config)
        zoom_val = cam.calculate_zoom(1, 1280 * 720)
        assert zoom_val <= camera_config.zoom_max

    def test_calculate_zoom_clamped_to_zoom_min(self, camera_config):
        """Result should never be below zoom_min."""
        cam = FakeCamera(camera_config)
        # Huge bbox fills the whole frame → minimum zoom
        frame_area = 1280 * 720
        zoom_val = cam.calculate_zoom(frame_area, frame_area)
        assert zoom_val >= camera_config.zoom_min

    def test_calculate_zoom_zero_area_returns_zoom_min(self, camera_config):
        cam = FakeCamera(camera_config)
        zoom_val = cam.calculate_zoom(0, 1280 * 720)
        assert zoom_val == camera_config.zoom_min

    def test_connect_sets_flag(self, camera_config):
        cam = FakeCamera(camera_config)
        result = cam.connect()
        assert result is True
        assert cam.connected is True

    def test_move_stored(self, camera_config):
        cam = FakeCamera(camera_config)
        cam.move(0.5, -0.3)
        assert cam.last_move == pytest.approx((0.5, -0.3))

    def test_stop_called(self, camera_config):
        cam = FakeCamera(camera_config)
        cam.stop()
        assert cam.stopped is True
