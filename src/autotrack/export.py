"""Model export utilities.

Export a YOLO .pt weights file to CoreML (.mlpackage) format so that
inference runs on the Apple Neural Engine rather than the MPS GPU.  On
M-series Macs the Neural Engine is typically 2–4× faster than MPS for
YOLOv8n-scale models with no change in accuracy.

Usage (CLI)::

    autotrack-export                          # exports yolov8n.pt → yolov8n.mlpackage
    autotrack-export --model yolov8s.pt
    autotrack-export --model yolov8n.pt --imgsz 320

Then use the exported model::

    autotrack --playback video.mp4 --model yolov8n.mlpackage
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_coreml(model_path: str = "yolov8n.pt", imgsz: int = 640) -> Path:
    """Export a YOLO model to CoreML format for Apple Neural Engine inference.

    The exported ``.mlpackage`` is placed in the same directory as the source
    weights file.  NMS is baked into the CoreML graph so the model is
    ready-to-use directly with Ultralytics.

    Args:
        model_path: Path to the source ``.pt`` weights file.
        imgsz:      Inference image size (square).  Must match the ``--imgsz``
                    value you plan to use at runtime.

    Returns:
        Path to the exported ``.mlpackage`` bundle.

    Raises:
        ImportError: If ``ultralytics`` is not installed.
        FileNotFoundError: If *model_path* does not exist and cannot be
                           downloaded by Ultralytics.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required for export") from exc

    logger.info("Loading %s …", model_path)
    model = YOLO(model_path)

    logger.info("Exporting to CoreML (imgsz=%d) …", imgsz)
    exported = model.export(format="coreml", imgsz=imgsz, nms=True)

    out = Path(exported)
    logger.info("Exported: %s", out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Export a YOLO model to CoreML for Apple Neural Engine inference"
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Source .pt weights file (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size in pixels, square (default: 640)",
    )
    args = parser.parse_args()

    out = export_coreml(args.model, args.imgsz)
    print(f"\nExport complete: {out}")
    print(f"Run with:  autotrack --model {out.name} ...")


if __name__ == "__main__":
    main()
