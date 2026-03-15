"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import yaml


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A blank 720p BGR frame."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def camera_config_dict() -> dict:
    """Raw dictionary with base camera config fields."""
    return {
        "ip": "192.168.1.100",
        "port": 80,
        "username": "root",
        "password": "pass",
        "pan_speed_max": 1.0,
        "tilt_speed_max": 1.0,
        "zoom_min": 0,
        "zoom_max": 9999,
        "focus_min": 0,
        "focus_max": 9999,
        "pid_x_p": 0.03,
        "pid_x_i": 0.01,
        "pid_x_d": 0.005,
        "pid_y_p": 0.03,
        "pid_y_i": 0.01,
        "pid_y_d": 0.005,
        "zoom_coefficient": 300.0,
    }


@pytest.fixture
def edelkrone_config_dict(camera_config_dict) -> dict:
    """Raw dictionary with all Edelkrone-specific fields included."""
    return {
        **camera_config_dict,
        "type": "edelkrone",
        "ptz_link_id": "207138435631",
        "focus_link_id": "204238735631",
        "head_module_mac": "1C:9D:C2:D6:FF:1E",
        "zoom_module_mac": "8C:4B:14:00:FF:DA",
        "focus_module_mac": "8C:4B:14:04:BA:82",
        "video_device": "0",
    }


@pytest.fixture
def camera_config(tmp_path, camera_config_dict):
    """A :class:`~autotrack.camera.base.CameraConfig` loaded from a temp YAML file."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml.dump(camera_config_dict))

    from autotrack.camera.base import CameraConfig

    return CameraConfig.from_yaml(str(config_file))


@pytest.fixture
def edelkrone_config(tmp_path, edelkrone_config_dict):
    """A :class:`~autotrack.camera.base.CameraConfig` with Edelkrone fields."""
    config_file = tmp_path / "edelkrone_config.yaml"
    config_file.write_text(yaml.dump(edelkrone_config_dict))

    from autotrack.camera.base import CameraConfig

    return CameraConfig.from_yaml(str(config_file))
