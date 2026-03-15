"""Unit tests for TrackerManager and Track."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autotrack.detector import Detection
from autotrack.tracker_manager import Track, TrackerManager


# ---------------------------------------------------------------------------
# Track dataclass
# ---------------------------------------------------------------------------


class TestTrack:
    def test_center_square_box(self):
        track = Track(track_id=1, bbox=(0, 0, 100, 100), confidence=0.8)
        assert track.center == (50, 50)

    def test_center_offset_box(self):
        track = Track(track_id=2, bbox=(200, 100, 400, 300), confidence=0.9)
        assert track.center == (300, 200)

    def test_center_non_square(self):
        track = Track(track_id=3, bbox=(0, 0, 200, 100), confidence=0.7)
        cx, cy = track.center
        assert cx == 100
        assert cy == 50

    def test_fields_stored(self):
        track = Track(track_id=42, bbox=(10, 20, 30, 40), confidence=0.55)
        assert track.track_id == 42
        assert track.bbox == (10, 20, 30, 40)
        assert track.confidence == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# TrackerManager._to_boxmot_array
# ---------------------------------------------------------------------------


class TestToBoxmotArray:
    def test_empty_detections_gives_empty_array(self):
        arr = TrackerManager._to_boxmot_array([])
        assert arr.shape == (0, 6)
        assert arr.dtype == np.float32

    def test_single_detection_shape(self):
        det = Detection(bbox=(10, 20, 110, 120), confidence=0.9, class_id=4)
        arr = TrackerManager._to_boxmot_array([det])
        assert arr.shape == (1, 6)

    def test_detection_values(self):
        det = Detection(bbox=(10, 20, 110, 120), confidence=0.85, class_id=4)
        arr = TrackerManager._to_boxmot_array([det])
        assert arr[0, 0] == pytest.approx(10)
        assert arr[0, 1] == pytest.approx(20)
        assert arr[0, 2] == pytest.approx(110)
        assert arr[0, 3] == pytest.approx(120)
        assert arr[0, 4] == pytest.approx(0.85)
        assert arr[0, 5] == pytest.approx(4)

    def test_multiple_detections(self):
        dets = [
            Detection(bbox=(0, 0, 50, 50), confidence=0.9, class_id=4),
            Detection(bbox=(100, 100, 200, 200), confidence=0.7, class_id=4),
        ]
        arr = TrackerManager._to_boxmot_array(dets)
        assert arr.shape == (2, 6)

    def test_output_dtype_is_float32(self):
        det = Detection(bbox=(10, 20, 110, 120), confidence=0.9, class_id=4)
        arr = TrackerManager._to_boxmot_array([det])
        assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# TrackerManager (with mocked boxmot)
# ---------------------------------------------------------------------------


def _make_tracker_manager(tracker_type="bytetrack"):
    """Build a TrackerManager with boxmot fully mocked."""
    with patch("autotrack.tracker_manager.boxmot") as mock_boxmot:
        mock_tracker_cls = MagicMock()
        mock_tracker_instance = MagicMock()
        mock_tracker_cls.return_value = mock_tracker_instance
        setattr(mock_boxmot, "ByteTrack", mock_tracker_cls)
        setattr(mock_boxmot, "StrongSort", mock_tracker_cls)
        setattr(mock_boxmot, "BotSort", mock_tracker_cls)

        manager = TrackerManager.__new__(TrackerManager)
        manager._tracker = mock_tracker_instance
        manager._device = "cpu"

    return manager, mock_tracker_instance


class TestTrackerManager:
    def test_update_empty_detections_returns_empty_list(self):
        manager, mock_tracker = _make_tracker_manager()
        mock_tracker.update.return_value = np.empty((0, 6), dtype=np.float32)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = manager.update([], frame)

        assert result == []

    def test_update_none_result_returns_empty_list(self):
        manager, mock_tracker = _make_tracker_manager()
        mock_tracker.update.return_value = None

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = manager.update([], frame)

        assert result == []

    def test_update_returns_track_objects(self):
        manager, mock_tracker = _make_tracker_manager()
        # boxmot output: [x1, y1, x2, y2, track_id, conf, class_id]
        raw = np.array([[10, 20, 110, 120, 7, 0.88, 4]], dtype=np.float32)
        mock_tracker.update.return_value = raw

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        det = Detection(bbox=(10, 20, 110, 120), confidence=0.88, class_id=4)
        tracks = manager.update([det], frame)

        assert len(tracks) == 1
        t = tracks[0]
        assert isinstance(t, Track)
        assert t.track_id == 7
        assert t.bbox == (10, 20, 110, 120)
        assert t.confidence == pytest.approx(0.88)

    def test_update_multiple_tracks(self):
        manager, mock_tracker = _make_tracker_manager()
        raw = np.array(
            [
                [10, 20, 110, 120, 1, 0.9, 4],
                [200, 200, 300, 300, 2, 0.7, 4],
            ],
            dtype=np.float32,
        )
        mock_tracker.update.return_value = raw

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        tracks = manager.update([], frame)

        assert len(tracks) == 2
        ids = {t.track_id for t in tracks}
        assert ids == {1, 2}

    def test_update_passes_det_array_to_boxmot(self):
        manager, mock_tracker = _make_tracker_manager()
        mock_tracker.update.return_value = np.empty((0, 6), dtype=np.float32)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        det = Detection(bbox=(10, 20, 110, 120), confidence=0.9, class_id=4)
        manager.update([det], frame)

        args, kwargs = mock_tracker.update.call_args
        passed_array = args[0]
        assert passed_array.shape == (1, 6)
        assert passed_array.dtype == np.float32

    def test_reset_clears_list_attribute(self):
        manager, mock_tracker = _make_tracker_manager()
        mock_tracker.trackers = [MagicMock(), MagicMock()]
        # Remove reset attr to trigger fallback path
        del mock_tracker.reset

        manager.reset()

        assert mock_tracker.trackers == []

    def test_reset_calls_tracker_reset_if_available(self):
        manager, mock_tracker = _make_tracker_manager()
        mock_tracker.reset = MagicMock()

        manager.reset()

        mock_tracker.reset.assert_called_once()

    @patch("autotrack.tracker_manager.boxmot")
    def test_invalid_tracker_type_raises(self, mock_boxmot):
        with pytest.raises(ValueError, match="Unknown tracker"):
            TrackerManager(tracker_type="nonexistent_tracker")
