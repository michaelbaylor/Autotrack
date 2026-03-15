"""Unit tests for device selection utilities."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from autotrack.utils.device import get_best_device, resolve_device, supports_half


class TestGetBestDevice:
    def test_returns_mps_when_available(self):
        with patch("autotrack.utils.device.torch.backends.mps.is_available", return_value=True), \
             patch("autotrack.utils.device.torch.cuda.is_available", return_value=False):
            assert get_best_device() == "mps"

    def test_mps_preferred_over_cuda(self):
        with patch("autotrack.utils.device.torch.backends.mps.is_available", return_value=True), \
             patch("autotrack.utils.device.torch.cuda.is_available", return_value=True):
            assert get_best_device() == "mps"

    def test_returns_cuda_when_mps_unavailable(self):
        with patch("autotrack.utils.device.torch.backends.mps.is_available", return_value=False), \
             patch("autotrack.utils.device.torch.cuda.is_available", return_value=True):
            assert get_best_device() == "cuda"

    def test_returns_cpu_as_fallback(self):
        with patch("autotrack.utils.device.torch.backends.mps.is_available", return_value=False), \
             patch("autotrack.utils.device.torch.cuda.is_available", return_value=False):
            assert get_best_device() == "cpu"


class TestResolveDevice:
    def test_auto_delegates_to_get_best_device(self):
        with patch("autotrack.utils.device.get_best_device", return_value="mps") as mock:
            result = resolve_device("auto")
        mock.assert_called_once()
        assert result == "mps"

    def test_explicit_device_passed_through(self):
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("mps") == "mps"
        assert resolve_device("cuda") == "cuda"
        assert resolve_device("cuda:0") == "cuda:0"

    def test_explicit_device_skips_auto_detection(self):
        with patch("autotrack.utils.device.get_best_device") as mock:
            resolve_device("cpu")
        mock.assert_not_called()


class TestSupportsHalf:
    def test_cuda_supports_half(self):
        assert supports_half("cuda") is True

    def test_cuda_with_index_supports_half(self):
        assert supports_half("cuda:0") is True
        assert supports_half("cuda:1") is True

    def test_mps_does_not_support_half(self):
        assert supports_half("mps") is False

    def test_cpu_does_not_support_half(self):
        assert supports_half("cpu") is False
