"""Multi-object tracker wrapper around boxmot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import boxmot
except ImportError:  # allow importing the module without boxmot installed (e.g. for tests)
    boxmot = None  # type: ignore[assignment]

from .detector import Detection
from .utils.device import resolve_device, supports_half


@dataclass
class Track:
    """A single tracked object returned by the tracker."""

    track_id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        """Return the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


# Mapping of human-readable tracker names to boxmot tracker classes.
_TRACKER_MAP: dict[str, str] = {
    "strongsort": "StrongSort",
    "bytetrack": "ByteTrack",
    "botsort": "BotSort",
}


class TrackerManager:
    """Wraps a boxmot tracker and converts between autotrack and boxmot formats.

    boxmot trackers expect detections as a float32 numpy array with columns
    ``[x1, y1, x2, y2, confidence, class_id]``.  Their ``update()`` method
    returns a float32 array with columns
    ``[x1, y1, x2, y2, track_id, confidence, class_id, ...]``.
    """

    def __init__(self, tracker_type: str = "strongsort", device: str = "auto") -> None:
        """Initialise the boxmot tracker.

        Args:
            tracker_type: One of ``"strongsort"``, ``"bytetrack"``, ``"botsort"``.
            device: Torch device string or ``"auto"`` to select the best
                    available device (MPS → CUDA → CPU).
        """
        tracker_name = _TRACKER_MAP.get(tracker_type.lower())
        if tracker_name is None:
            raise ValueError(
                f"Unknown tracker '{tracker_type}'. "
                f"Choose from: {list(_TRACKER_MAP.keys())}"
            )

        self._device = resolve_device(device)
        tracker_cls = getattr(boxmot, tracker_name)

        # StrongSort and BotSort run a ReID model; ByteTrack is Kalman-filter only.
        if tracker_type.lower() in ("strongsort", "botsort"):
            self._tracker = tracker_cls(
                reid_weights=Path("osnet_x0_25_msmt17.pt"),
                device=self._device,
                half=supports_half(self._device),
            )
        else:
            self._tracker = tracker_cls()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        """Update the tracker with new detections and return current tracks.

        Args:
            detections: Plane detections from :class:`~autotrack.detector.PlaneDetector`.
            frame: The current BGR video frame (required by some re-id models).

        Returns:
            List of active :class:`Track` objects.
        """
        det_array = self._to_boxmot_array(detections)
        raw = self._tracker.update(det_array, frame)

        if raw is None or len(raw) == 0:
            return []

        tracks: list[Track] = []
        for row in raw:
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            track_id = int(row[4])
            conf = float(row[5])
            tracks.append(Track(track_id=track_id, bbox=(x1, y1, x2, y2), confidence=conf))

        return tracks

    def reset(self) -> None:
        """Clear all tracked objects by reinitialising the internal tracker."""
        # boxmot trackers do not expose a reset method; re-creating achieves the same effect.
        if hasattr(self._tracker, "reset"):
            self._tracker.reset()
        else:
            # Re-initialise in-place by clearing internal state attributes where possible.
            for attr in ("trackers", "tracks", "_tracks", "active_tracks"):
                if hasattr(self._tracker, attr):
                    container = getattr(self._tracker, attr)
                    if isinstance(container, list):
                        container.clear()
                    elif isinstance(container, dict):
                        container.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_boxmot_array(detections: list[Detection]) -> np.ndarray:
        """Convert a list of :class:`Detection` objects to a boxmot-compatible array.

        Returns:
            Float32 array of shape ``(N, 6)`` with columns
            ``[x1, y1, x2, y2, confidence, class_id]``, or an empty
            ``(0, 6)`` array when *detections* is empty.
        """
        if not detections:
            return np.empty((0, 6), dtype=np.float32)

        rows = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            rows.append([x1, y1, x2, y2, d.confidence, float(d.class_id)])

        return np.array(rows, dtype=np.float32)
