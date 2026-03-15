"""Plane detection wrapper around Ultralytics YOLO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # allow importing the module without ultralytics installed (e.g. for tests)
    YOLO = None  # type: ignore[assignment,misc]

from .utils.device import resolve_device

# CoreML models handle their own compute-unit placement (Neural Engine / GPU / CPU)
# internally. Passing device="mps" causes a float64→MPS crash because CoreML
# returns float64 numpy arrays which MPS cannot accept.
_COREML_EXTENSIONS = {".mlpackage", ".mlmodel"}


@dataclass
class Detection:
    """A single plane detection result."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int

    @property
    def center(self) -> tuple[int, int]:
        """Return the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        """Return the area of the bounding box in pixels."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class PlaneDetector:
    """Wraps Ultralytics YOLO and filters detections to airplane class only."""

    # COCO class index for "airplane"
    PLANE_CLASS_ID = 4

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.4,
        device: str = "auto",
    ) -> None:
        """Load a YOLO model and configure detection thresholds.

        Args:
            model_path: Path to a .pt weights file or a model name resolvable
                        by Ultralytics (e.g. ``"yolov8n.pt"``).
            confidence: Minimum detection confidence in [0, 1].
            device: Torch device string or ``"auto"`` to select the best
                    available device (MPS → CUDA → CPU).
        """
        self.confidence = confidence
        self.device = self._select_device(model_path, device)
        self._model = YOLO(model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on *frame* and return only airplane detections.

        Args:
            frame: BGR image as a ``numpy`` array (H x W x 3).

        Returns:
            List of :class:`Detection` objects, one per airplane found.
        """
        results = self._model.predict(
            frame,
            conf=self.confidence,
            classes=[self.PLANE_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id != self.PLANE_CLASS_ID:
                    continue
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf, class_id=cls_id))

        return detections

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_device(model_path: str, requested: str) -> str:
        """Return the correct inference device for *model_path*.

        CoreML models (``.mlpackage`` / ``.mlmodel``) route inference to the
        Neural Engine internally.  They must be run with ``device="cpu"`` from
        PyTorch's perspective because CoreML returns ``float64`` numpy arrays
        which MPS cannot accept.

        All other model formats use the caller's requested device (defaulting
        to auto-selection).
        """
        from pathlib import Path
        if Path(model_path).suffix.lower() in _COREML_EXTENSIONS:
            return "cpu"
        return resolve_device(requested)

    def load_custom_model(self, model_path: str, device: str = "auto") -> None:
        """Replace the current model with a custom weights file or CoreML bundle.

        Args:
            model_path: Filesystem path to a ``.pt`` file or ``.mlpackage`` bundle.
            device:     Desired device, or ``"auto"``.  Ignored for CoreML models.
        """
        self.device = self._select_device(model_path, device)
        self._model = YOLO(model_path)
