"""Edelkrone wireless camera head implementation."""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from .base import CameraConfig, PTZCamera

logger = logging.getLogger(__name__)

# Encoder value at zoom position 0.0 (wide end of the lens servo)
_ZOOM_ENC_OFFSET = 24400
_ZOOM_ENC_SCALE = 24000


class EdelkroneCamera(PTZCamera):
    """Controls an Edelkrone wireless camera head via its REST API.

    The Edelkrone hub exposes a JSON REST API at ``http://{ip}:{port}/v1``.
    Two wireless links are used:

    * **PTZ link** – pan, tilt, and (optionally) a zoom servo
    * **Focus link** – a dedicated follow-focus ring module

    Devices are paired wirelessly by MAC address.  If pairing has already
    been established the hub confirms "Link is already paired." and no
    re-pairing is needed.
    """

    _API_VERSION = "v1"
    # Duration (seconds) given to the hub to complete a wireless scan
    _PAIRING_SCAN_DELAY = 3.0

    def __init__(self, config: CameraConfig) -> None:
        super().__init__(config)
        self._base_url = f"http://{config.ip}:{config.port}/{self._API_VERSION}"
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._head_paired = False
        self._focus_paired = False

    # ------------------------------------------------------------------
    # PTZCamera abstract methods
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Check wireless pairing status and pair devices if necessary.

        Returns:
            ``True`` if the PTZ link is paired and responding, ``False``
            otherwise.
        """
        self._check_pairing(self.config.ptz_link_id)
        self._check_pairing(self.config.focus_link_id)

        if not self._head_paired and self.config.head_module_mac:
            macs = [self.config.head_module_mac]
            if self.config.zoom_module_mac:
                macs.append(self.config.zoom_module_mac)
            self._pair(macs, self.config.ptz_link_id)

        if not self._focus_paired and self.config.focus_module_mac:
            self._pair([self.config.focus_module_mac], self.config.focus_link_id)

        status = self.get_status()
        return bool(status)

    def move(self, pan_speed: float, tilt_speed: float) -> None:
        """Send a continuous joystick move command.

        Args:
            pan_speed:  Normalised speed in ``[-1.0, 1.0]``; positive = right.
            tilt_speed: Normalised speed in ``[-1.0, 1.0]``; positive = down.
        """
        self._call(
            "bundle", "",
            {"command": "joystickMove", "headPan": pan_speed, "headTilt": tilt_speed},
            self.config.ptz_link_id,
        )

    def stop(self) -> None:
        """Halt all camera motion."""
        self._call(
            "bundle", "",
            {"command": "joystickMove", "headPan": 0.0, "headTilt": 0.0},
            self.config.ptz_link_id,
        )

    def zoom(self, zoom_value: int) -> None:
        """Seek the zoom servo toward *zoom_value* by sending a delta command.

        The Edelkrone zoom servo does not support absolute positioning; this
        method approximates it by computing a proportional delta from the
        current position.  The result converges with repeated calls.

        Args:
            zoom_value: Target position within ``[zoom_min, zoom_max]``.
        """
        clamped = max(self.config.zoom_min, min(self.config.zoom_max, zoom_value))
        zoom_range = max(self.config.zoom_max - self.config.zoom_min, 1)
        target_norm = (clamped - self.config.zoom_min) / zoom_range
        current_norm = (self.state.zoom - self.config.zoom_min) / zoom_range

        delta = (target_norm - current_norm) * 10.0

        # Don't drive against hard stops
        at_max = self.state.zoom >= self.config.zoom_max and delta > 0
        at_min = self.state.zoom <= self.config.zoom_min and delta < 0
        if at_max or at_min or delta == 0:
            return

        self._call(
            "bundle", "",
            {"command": "focusManualMove", "deltaEnc": delta},
            self.config.ptz_link_id,
        )

    def get_status(self) -> dict:
        """Poll the PTZ link and update :attr:`state`.

        Returns:
            Dictionary with ``pan``, ``tilt``, ``zoom``, ``focus`` keys, or
            an empty dict on failure.
        """
        resp = self._call("bundle", "status", None, self.config.ptz_link_id)
        if resp is None:
            return {}

        try:
            readings = resp.json()["data"]["readings"]
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse PTZ status: %s", exc)
            return {}

        self.state.pan = float(readings.get("headPan", self.state.pan))
        self.state.tilt = float(readings.get("headTilt", self.state.tilt))

        if "focus" in readings:
            raw_enc = readings["focus"]
            norm = (_ZOOM_ENC_OFFSET + raw_enc) / _ZOOM_ENC_SCALE
            norm = max(0.0, min(1.0, norm))
            self.state.zoom = int(norm * self.config.zoom_max)

        return {
            "pan": self.state.pan,
            "tilt": self.state.tilt,
            "zoom": self.state.zoom,
            "focus": self.state.focus,
        }

    def get_video_source(self) -> str | int:
        """Return the local capture device index or path.

        The value is taken from ``config.video_device``.  A purely numeric
        string (e.g. ``"0"``) is returned as an integer so that
        ``cv2.VideoCapture`` opens a live capture device rather than a file.

        Returns:
            Integer device index or device path string.
        """
        try:
            return int(self.config.video_device)
        except ValueError:
            return self.config.video_device

    # ------------------------------------------------------------------
    # Additional public helpers
    # ------------------------------------------------------------------

    def focus_move(self, delta: float) -> None:
        """Move the follow-focus ring by a relative encoder delta.

        Args:
            delta: Positive values move toward infinity focus; negative toward
                   minimum focus.
        """
        self._call(
            "bundle", "",
            {"command": "focusManualMove", "deltaEnc": delta},
            self.config.focus_link_id,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call(
        self,
        call_type: str,
        command: str,
        payload: Optional[dict],
        link_id: str,
    ) -> Optional[requests.Response]:
        """Send a POST request to the Edelkrone REST API.

        URL format: ``/v1/{call_type}/{link_id}[/{command}]``

        Args:
            call_type: ``"link"`` for pairing commands, ``"bundle"`` for
                       motion/status commands.
            command:   Sub-resource (e.g. ``"status"``); empty string for
                       action commands.
            payload:   JSON body; ``None`` sends an empty body.
            link_id:   Wireless link identifier.

        Returns:
            The :class:`requests.Response` on success, ``None`` on failure.
        """
        path = f"/{command}" if command else ""
        url = f"{self._base_url}/{call_type}/{link_id}{path}"

        t0 = time.monotonic()
        try:
            resp = self._session.post(url, json=payload, timeout=5)
        except requests.RequestException as exc:
            logger.warning("Edelkrone request failed (%s %s): %s", call_type, command, exc)
            return None
        finally:
            self.ping_ms = (time.monotonic() - t0) * 1000

        if call_type == "link":
            self._handle_link_response(resp, link_id)

        if not resp.ok:
            logger.warning("Edelkrone returned HTTP %d for %s", resp.status_code, url)

        return resp

    def _handle_link_response(self, resp: requests.Response, link_id: str) -> None:
        """Parse a *link* response and update the pairing flags."""
        try:
            data = resp.json()
        except ValueError:
            return

        if "Link is already paired." in data.get("message", ""):
            if link_id == self.config.ptz_link_id:
                self._head_paired = True
            elif link_id == self.config.focus_link_id:
                self._focus_paired = True

    def _check_pairing(self, link_id: str) -> None:
        """Query pairing status for *link_id*, updating pairing flags."""
        self._call("link", "", {"command": "wirelessPairingStatus"}, link_id)

    def _pair(self, mac_addresses: list[str], link_id: str) -> None:
        """Run the full wireless pairing sequence for a set of MAC addresses.

        Args:
            mac_addresses: List of device MAC addresses to bundle together.
            link_id:       Link to associate the bundle with.
        """
        self._call("link", "", {"command": "wirelessPairingScanStart"}, link_id)
        time.sleep(self._PAIRING_SCAN_DELAY)
        self._call("link", "", {"command": "wirelessPairingScanResults"}, link_id)
        self._call(
            "link", "",
            {
                "command": "wirelessPairingCreateBundle",
                "deviceCount": len(mac_addresses),
                "forcedMasterDevice": "none",
                "macList": mac_addresses,
            },
            link_id,
        )
        self._call("link", "", {"command": "wirelessPairingStatus"}, link_id)
