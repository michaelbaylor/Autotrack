"""Unit tests for PlaneDetector and Detection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autotrack.detector import Detection, PlaneDetector


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------


class TestDetection:
    def test_center_mid_box(self):
        det = Detection(bbox=(100, 200, 300, 400), confidence=0.9, class_id=4)
        assert det.center == (200, 300)

    def test_center_zero_origin(self):
        det = Detection(bbox=(0, 0, 10, 10), confidence=0.5, class_id=4)
        assert det.center == (5, 5)

    def test_center_non_square(self):
        det = Detection(bbox=(0, 0, 100, 50), confidence=0.8, class_id=4)
        cx, cy = det.center
        assert cx == 50
        assert cy == 25

    def test_area_standard(self):
        det = Detection(bbox=(0, 0, 100, 200), confidence=0.7, class_id=4)
        assert det.area == 20_000

    def test_area_zero_size(self):
        det = Detection(bbox=(50, 50, 50, 50), confidence=0.5, class_id=4)
        assert det.area == 0

    def test_confidence_stored(self):
        det = Detection(bbox=(0, 0, 10, 10), confidence=0.42, class_id=4)
        assert det.confidence == pytest.approx(0.42)

    def test_class_id_stored(self):
        det = Detection(bbox=(0, 0, 10, 10), confidence=0.5, class_id=4)
        assert det.class_id == 4


# ---------------------------------------------------------------------------
# PlaneDetector
# ---------------------------------------------------------------------------


def _make_mock_box(x1=10, y1=20, x2=110, y2=120, conf=0.85, cls_id=4):
    """Build a mock Ultralytics box object."""
    box = MagicMock()
    box.cls = MagicMock()
    box.cls.__getitem__ = MagicMock(return_value=MagicMock(item=MagicMock(return_value=float(cls_id))))
    box.conf = MagicMock()
    box.conf.__getitem__ = MagicMock(return_value=MagicMock(item=MagicMock(return_value=conf)))
    box.xyxy = MagicMock()
    box.xyxy.__getitem__ = MagicMock(
        return_value=MagicMock(tolist=MagicMock(return_value=[x1, y1, x2, y2]))
    )
    return box


def _make_mock_result(boxes):
    result = MagicMock()
    result.boxes = boxes
    return result


class TestPlaneDetector:
    @patch("autotrack.detector.YOLO")
    def test_init_loads_model(self, MockYOLO):
        """PlaneDetector should instantiate the YOLO model on init."""
        detector = PlaneDetector("yolov8n.pt", confidence=0.4, device="cpu")
        MockYOLO.assert_called_once_with("yolov8n.pt")

    @patch("autotrack.detector.YOLO")
    def test_detect_returns_plane_detection(self, MockYOLO):
        mock_box = _make_mock_box(cls_id=4, conf=0.9, x1=10, y1=20, x2=110, y2=120)
        mock_result = _make_mock_result([mock_box])
        MockYOLO.return_value.predict.return_value = [mock_result]

        detector = PlaneDetector()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 1
        assert detections[0].class_id == PlaneDetector.PLANE_CLASS_ID
        assert detections[0].confidence == pytest.approx(0.9)
        assert detections[0].bbox == (10, 20, 110, 120)

    @patch("autotrack.detector.YOLO")
    def test_detect_filters_non_plane_classes(self, MockYOLO):
        """Detections with class_id != 4 should be discarded."""
        # cls_id=0 → person, should be filtered
        mock_box = _make_mock_box(cls_id=0, conf=0.95)
        mock_result = _make_mock_result([mock_box])
        MockYOLO.return_value.predict.return_value = [mock_result]

        detector = PlaneDetector()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert detections == []

    @patch("autotrack.detector.YOLO")
    def test_detect_empty_when_no_planes(self, MockYOLO):
        """detect() should return an empty list when YOLO finds nothing."""
        mock_result = _make_mock_result([])
        MockYOLO.return_value.predict.return_value = [mock_result]

        detector = PlaneDetector()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert detector.detect(frame) == []

    @patch("autotrack.detector.YOLO")
    def test_detect_empty_when_boxes_is_none(self, MockYOLO):
        """detect() should handle results where boxes is None."""
        mock_result = _make_mock_result(None)
        MockYOLO.return_value.predict.return_value = [mock_result]

        detector = PlaneDetector()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert detector.detect(frame) == []

    @patch("autotrack.detector.YOLO")
    def test_detect_multiple_planes(self, MockYOLO):
        boxes = [
            _make_mock_box(cls_id=4, conf=0.9, x1=0, y1=0, x2=50, y2=50),
            _make_mock_box(cls_id=4, conf=0.7, x1=100, y1=100, x2=200, y2=200),
        ]
        mock_result = _make_mock_result(boxes)
        MockYOLO.return_value.predict.return_value = [mock_result]

        detector = PlaneDetector()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = detector.detect(frame)

        assert len(detections) == 2

    @patch("autotrack.detector.YOLO")
    def test_load_custom_model_replaces_model(self, MockYOLO):
        detector = PlaneDetector()
        detector.load_custom_model("custom_weights.pt")
        # YOLO should have been called twice: once in __init__, once in load_custom_model
        assert MockYOLO.call_count == 2
        MockYOLO.assert_called_with("custom_weights.pt")

    @patch("autotrack.detector.YOLO")
    def test_coreml_mlpackage_forces_cpu_device(self, MockYOLO):
        """CoreML models must use device='cpu' regardless of the requested device."""
        detector = PlaneDetector("model.mlpackage", device="auto")
        assert detector.device == "cpu"

    @patch("autotrack.detector.YOLO")
    def test_coreml_mlmodel_forces_cpu_device(self, MockYOLO):
        detector = PlaneDetector("model.mlmodel", device="mps")
        assert detector.device == "cpu"

    @patch("autotrack.detector.YOLO")
    def test_pt_model_does_not_force_cpu(self, MockYOLO):
        detector = PlaneDetector("yolov8n.pt", device="cpu")
        assert detector.device == "cpu"  # explicitly set, not forced

    @patch("autotrack.detector.YOLO")
    def test_load_custom_coreml_model_sets_cpu_device(self, MockYOLO):
        detector = PlaneDetector("yolov8n.pt", device="cpu")
        detector.load_custom_model("yolov8n.mlpackage")
        assert detector.device == "cpu"

    @patch("autotrack.detector.YOLO")
    def test_detect_calls_predict_with_plane_class(self, MockYOLO):
        mock_result = _make_mock_result([])
        MockYOLO.return_value.predict.return_value = [mock_result]

        detector = PlaneDetector(confidence=0.5, device="cpu")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector.detect(frame)

        call_kwargs = MockYOLO.return_value.predict.call_args
        assert call_kwargs.kwargs["classes"] == [PlaneDetector.PLANE_CLASS_ID]
        assert call_kwargs.kwargs["conf"] == pytest.approx(0.5)
