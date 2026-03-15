"""Unit tests for Recorder and TelemetryFrame."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autotrack.utils.recorder import Recorder, TelemetryFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_telemetry(**kwargs) -> TelemetryFrame:
    defaults = dict(
        timestamp=1.0,
        fps=30.0,
        ping_ms=5.0,
        pan=10.0,
        tilt=-5.0,
        zoom=100,
        is_tracking=True,
        track_id=1,
        target_x=640.0,
        target_y=360.0,
        confidence=0.88,
    )
    defaults.update(kwargs)
    return TelemetryFrame(**defaults)


def _blank_frame(w=1280, h=720) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# TelemetryFrame dataclass
# ---------------------------------------------------------------------------


class TestTelemetryFrame:
    def test_fields_stored(self):
        t = _make_telemetry()
        assert t.fps == pytest.approx(30.0)
        assert t.pan == pytest.approx(10.0)
        assert t.track_id == 1

    def test_optional_fields_can_be_none(self):
        t = _make_telemetry(track_id=None, target_x=None, target_y=None, confidence=None)
        assert t.track_id is None
        assert t.target_x is None


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class TestRecorderIsRecording:
    def test_not_recording_initially(self, tmp_path):
        rec = Recorder(str(tmp_path))
        assert rec.is_recording is False

    def test_recording_after_start(self, tmp_path):
        rec = Recorder(str(tmp_path))
        with patch("autotrack.utils.recorder.cv2.VideoWriter") as MockVW:
            MockVW.return_value.isOpened.return_value = True
            rec.start(1280, 720)
        assert rec.is_recording is True

    def test_not_recording_after_stop(self, tmp_path):
        rec = Recorder(str(tmp_path))
        with patch("autotrack.utils.recorder.cv2.VideoWriter") as MockVW:
            MockVW.return_value.isOpened.return_value = True
            rec.start(1280, 720)
            rec.stop()
        assert rec.is_recording is False


class TestRecorderStart:
    def test_creates_output_directory(self, tmp_path):
        output_dir = tmp_path / "new_subdir"
        rec = Recorder(str(output_dir))
        with patch("autotrack.utils.recorder.cv2.VideoWriter") as MockVW:
            MockVW.return_value.isOpened.return_value = True
            rec.start(1280, 720)
        assert output_dir.exists()

    def test_start_twice_ignored(self, tmp_path):
        rec = Recorder(str(tmp_path))
        with patch("autotrack.utils.recorder.cv2.VideoWriter") as MockVW:
            MockVW.return_value.isOpened.return_value = True
            rec.start(1280, 720)
            first_path = rec._video_path
            rec.start(1280, 720)  # second call should be ignored
        assert rec._video_path == first_path  # unchanged

    def test_raises_if_video_writer_fails_to_open(self, tmp_path):
        rec = Recorder(str(tmp_path))
        with patch("autotrack.utils.recorder.cv2.VideoWriter") as MockVW:
            MockVW.return_value.isOpened.return_value = False
            with pytest.raises(RuntimeError, match="Failed to open video writer"):
                rec.start(1280, 720)

    def test_session_name_in_filename(self, tmp_path):
        rec = Recorder(str(tmp_path))
        with patch("autotrack.utils.recorder.cv2.VideoWriter") as MockVW:
            MockVW.return_value.isOpened.return_value = True
            rec.start(1280, 720, session_name="mysession")
        assert "mysession" in rec._video_path.name


class TestRecorderWriteFrame:
    def test_write_frame_calls_video_writer(self, tmp_path):
        rec = Recorder(str(tmp_path))
        with patch("autotrack.utils.recorder.cv2.VideoWriter") as MockVW:
            mock_writer = MagicMock()
            mock_writer.isOpened.return_value = True
            MockVW.return_value = mock_writer
            rec.start(1280, 720)
            frame = _blank_frame()
            rec.write_frame(frame)
            mock_writer.write.assert_called_once_with(frame)

    def test_write_frame_ignored_when_not_recording(self, tmp_path):
        rec = Recorder(str(tmp_path))
        with patch("autotrack.utils.recorder.cv2.VideoWriter") as MockVW:
            mock_writer = MagicMock()
            mock_writer.isOpened.return_value = True
            MockVW.return_value = mock_writer
            # Never started recording
            rec.write_frame(_blank_frame())
            mock_writer.write.assert_not_called()


class TestRecorderLogTelemetry:
    def test_log_telemetry_accumulates_records(self, tmp_path):
        rec = Recorder(str(tmp_path))
        with patch("autotrack.utils.recorder.cv2.VideoWriter") as MockVW:
            MockVW.return_value.isOpened.return_value = True
            rec.start(1280, 720)
            for i in range(5):
                rec.log_telemetry(_make_telemetry(timestamp=float(i)))
        assert len(rec._telemetry_rows) == 5

    def test_log_telemetry_ignored_when_not_recording(self, tmp_path):
        rec = Recorder(str(tmp_path))
        rec.log_telemetry(_make_telemetry())
        assert len(rec._telemetry_rows) == 0


class TestRecorderStop:
    def test_stop_returns_video_path(self, tmp_path):
        rec = Recorder(str(tmp_path))
        with patch("autotrack.utils.recorder.cv2.VideoWriter") as MockVW:
            MockVW.return_value.isOpened.return_value = True
            rec.start(1280, 720)
            path = rec.stop()
        assert path is not None
        assert str(path).endswith(".mp4")

    def test_stop_returns_none_when_not_recording(self, tmp_path):
        rec = Recorder(str(tmp_path))
        assert rec.stop() is None

    def test_stop_writes_csv_with_correct_rows(self, tmp_path):
        rec = Recorder(str(tmp_path))
        with patch("autotrack.utils.recorder.cv2.VideoWriter") as MockVW:
            MockVW.return_value.isOpened.return_value = True
            rec.start(1280, 720)
            for i in range(3):
                rec.log_telemetry(_make_telemetry(timestamp=float(i)))
            rec.stop()

        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 1

        with open(csv_files[0]) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 3

    def test_stop_csv_contains_expected_columns(self, tmp_path):
        rec = Recorder(str(tmp_path))
        with patch("autotrack.utils.recorder.cv2.VideoWriter") as MockVW:
            MockVW.return_value.isOpened.return_value = True
            rec.start(1280, 720)
            rec.log_telemetry(_make_telemetry())
            rec.stop()

        csv_files = list(tmp_path.glob("*.csv"))
        with open(csv_files[0]) as fh:
            reader = csv.DictReader(fh)
            columns = reader.fieldnames

        assert "timestamp" in columns
        assert "fps" in columns
        assert "pan" in columns
        assert "track_id" in columns


class TestRecorderToggle:
    def test_toggle_starts_recording(self, tmp_path):
        rec = Recorder(str(tmp_path))
        with patch("autotrack.utils.recorder.cv2.VideoWriter") as MockVW:
            MockVW.return_value.isOpened.return_value = True
            new_state = rec.toggle(1280, 720)
        assert new_state is True
        assert rec.is_recording is True

    def test_toggle_stops_recording(self, tmp_path):
        rec = Recorder(str(tmp_path))
        with patch("autotrack.utils.recorder.cv2.VideoWriter") as MockVW:
            MockVW.return_value.isOpened.return_value = True
            rec.start(1280, 720)
            new_state = rec.toggle(1280, 720)
        assert new_state is False
        assert rec.is_recording is False

    def test_toggle_returns_bool(self, tmp_path):
        rec = Recorder(str(tmp_path))
        with patch("autotrack.utils.recorder.cv2.VideoWriter") as MockVW:
            MockVW.return_value.isOpened.return_value = True
            result = rec.toggle(1280, 720)
        assert isinstance(result, bool)
