"""Device selection helpers for PyTorch inference."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Preference order for auto-selection
_DEVICE_PRIORITY = ["mps", "cuda", "cpu"]


def get_best_device() -> str:
    """Return the fastest PyTorch device available on this machine.

    Priority order:
    1. ``mps``  – Apple Silicon GPU (Metal Performance Shaders)
    2. ``cuda`` – NVIDIA GPU
    3. ``cpu``  – fallback

    Returns:
        A device string suitable for passing to ``torch`` or Ultralytics.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to the best available device; pass others through.

    Args:
        device: A device string or ``"auto"``.

    Returns:
        A concrete device string, e.g. ``"mps"``, ``"cuda"``, ``"cpu"``.
    """
    if device == "auto":
        chosen = get_best_device()
        logger.info("Device auto-selected: %s", chosen)
        return chosen
    return device


def supports_half(device: str) -> bool:
    """Return whether FP16 (half precision) inference is safe on *device*.

    CUDA supports FP16 reliably.  MPS has partial FP16 support in PyTorch but
    some operations used by ReID models are not yet covered; we conservatively
    return False for MPS to avoid runtime errors.  CPU does not benefit from
    FP16.

    Args:
        device: Concrete device string (not ``"auto"``).

    Returns:
        ``True`` only for CUDA devices.
    """
    return device.startswith("cuda")
