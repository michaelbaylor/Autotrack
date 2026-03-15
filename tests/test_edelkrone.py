"""Unit tests for EdelkroneCamera."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from autotrack.camera.edelkrone import EdelkroneCamera, _ZOOM_ENC_OFFSET, _ZOOM_ENC_SCALE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_response(body: dict) -> MagicMock:
    """Build a mock requests.Response with status 200 and a JSON body."""
    resp = MagicMock(spec=requests.Response)
    resp.ok = True
    resp.status_code = 200
    resp.json.return_value = body
    return resp


def _ptz_status_response(pan=0.0, tilt=0.0, focus_enc=None) -> MagicMock:
    readings: dict = {"headPan": pan, "headTilt": tilt}
    if focus_enc is not None:
        readings["focus"] = focus_enc
    return _ok_response({"data": {"readings": readings}})


def _paired_link_response(link_id: str) -> MagicMock:
    return _ok_response({"message": f"Link is already paired. ({link_id})"})


# ---------------------------------------------------------------------------
# EdelkroneCamera._call
# ---------------------------------------------------------------------------


class TestCall:
    def test_posts_to_correct_url_with_command(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        with patch.object(cam._session, "post", return_value=_ok_response({})) as mock_post:
            cam._call("bundle", "status", None, "LINK123")
        mock_post.assert_called_once()
        url = mock_post.call_args.args[0]
        assert url.endswith("/bundle/LINK123/status")

    def test_posts_to_correct_url_without_command(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        with patch.object(cam._session, "post", return_value=_ok_response({})) as mock_post:
            cam._call("bundle", "", {"command": "joystickMove"}, "LINK123")
        url = mock_post.call_args.args[0]
        assert url.endswith("/bundle/LINK123")

    def test_returns_none_on_connection_error(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        with patch.object(cam._session, "post", side_effect=requests.ConnectionError):
            result = cam._call("bundle", "status", None, "LINK123")
        assert result is None

    def test_updates_ping_ms_on_success(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        with patch.object(cam._session, "post", return_value=_ok_response({})):
            cam._call("bundle", "status", None, "LINK123")
        assert cam.ping_ms >= 0

    def test_updates_ping_ms_on_failure(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        with patch.object(cam._session, "post", side_effect=requests.ConnectionError):
            cam._call("bundle", "status", None, "LINK123")
        assert cam.ping_ms >= 0

    def test_calls_handle_link_response_for_link_type(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        resp = _ok_response({"message": "Link is already paired."})
        with patch.object(cam._session, "post", return_value=resp):
            with patch.object(cam, "_handle_link_response") as mock_hlr:
                cam._call("link", "", {}, "LINK123")
        mock_hlr.assert_called_once_with(resp, "LINK123")

    def test_does_not_call_handle_link_response_for_bundle_type(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        with patch.object(cam._session, "post", return_value=_ok_response({})):
            with patch.object(cam, "_handle_link_response") as mock_hlr:
                cam._call("bundle", "status", None, "LINK123")
        mock_hlr.assert_not_called()


# ---------------------------------------------------------------------------
# EdelkroneCamera._handle_link_response
# ---------------------------------------------------------------------------


class TestHandleLinkResponse:
    def test_sets_head_paired_for_ptz_link(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        resp = _paired_link_response(edelkrone_config.ptz_link_id)
        cam._handle_link_response(resp, edelkrone_config.ptz_link_id)
        assert cam._head_paired is True
        assert cam._focus_paired is False

    def test_sets_focus_paired_for_focus_link(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        resp = _paired_link_response(edelkrone_config.focus_link_id)
        cam._handle_link_response(resp, edelkrone_config.focus_link_id)
        assert cam._focus_paired is True
        assert cam._head_paired is False

    def test_no_change_when_not_paired_message(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        resp = _ok_response({"message": "Scan in progress."})
        cam._handle_link_response(resp, edelkrone_config.ptz_link_id)
        assert cam._head_paired is False

    def test_no_crash_on_invalid_json(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        resp = MagicMock(spec=requests.Response)
        resp.json.side_effect = ValueError("not json")
        # Should not raise
        cam._handle_link_response(resp, edelkrone_config.ptz_link_id)


# ---------------------------------------------------------------------------
# EdelkroneCamera.move / stop
# ---------------------------------------------------------------------------


class TestMove:
    def test_move_sends_joystick_move(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        with patch.object(cam, "_call") as mock_call:
            cam.move(0.5, -0.3)
        mock_call.assert_called_once_with(
            "bundle", "",
            {"command": "joystickMove", "headPan": 0.5, "headTilt": -0.3},
            edelkrone_config.ptz_link_id,
        )

    def test_stop_sends_zero_joystick_move(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        with patch.object(cam, "_call") as mock_call:
            cam.stop()
        mock_call.assert_called_once_with(
            "bundle", "",
            {"command": "joystickMove", "headPan": 0.0, "headTilt": 0.0},
            edelkrone_config.ptz_link_id,
        )


# ---------------------------------------------------------------------------
# EdelkroneCamera.zoom
# ---------------------------------------------------------------------------


class TestZoom:
    def test_zoom_sends_focus_manual_move(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        cam.state.zoom = 0  # at wide end
        with patch.object(cam, "_call") as mock_call:
            cam.zoom(5000)
        mock_call.assert_called_once()
        payload = mock_call.call_args.args[2]
        assert payload["command"] == "focusManualMove"
        assert payload["deltaEnc"] > 0  # moving toward tele

    def test_zoom_does_nothing_when_already_at_target(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        cam.state.zoom = 5000
        with patch.object(cam, "_call") as mock_call:
            cam.zoom(5000)
        mock_call.assert_not_called()

    def test_zoom_does_not_drive_past_max(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        cam.state.zoom = edelkrone_config.zoom_max  # already at max
        with patch.object(cam, "_call") as mock_call:
            cam.zoom(edelkrone_config.zoom_max)
        mock_call.assert_not_called()

    def test_zoom_does_not_drive_past_min(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        cam.state.zoom = edelkrone_config.zoom_min  # already at min
        with patch.object(cam, "_call") as mock_call:
            cam.zoom(0)
        mock_call.assert_not_called()

    def test_zoom_clamps_value_above_max(self, edelkrone_config):
        """zoom_value above zoom_max should be clamped, not send out-of-range delta."""
        cam = EdelkroneCamera(edelkrone_config)
        cam.state.zoom = 0
        with patch.object(cam, "_call") as mock_call:
            cam.zoom(99999)
        # Should call with a delta, not raise
        mock_call.assert_called_once()


# ---------------------------------------------------------------------------
# EdelkroneCamera.get_status
# ---------------------------------------------------------------------------


class TestGetStatus:
    def test_returns_dict_with_expected_keys(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        with patch.object(cam, "_call", return_value=_ptz_status_response(pan=1.5, tilt=-0.3)):
            result = cam.get_status()
        assert set(result.keys()) == {"pan", "tilt", "zoom", "focus"}

    def test_updates_state_pan_tilt(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        with patch.object(cam, "_call", return_value=_ptz_status_response(pan=1.5, tilt=-0.3)):
            cam.get_status()
        assert cam.state.pan == pytest.approx(1.5)
        assert cam.state.tilt == pytest.approx(-0.3)

    def test_updates_state_zoom_from_encoder(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        # focus_enc = 0 → norm = 24400/24000 → clamped to 1.0 → zoom_max
        with patch.object(cam, "_call", return_value=_ptz_status_response(focus_enc=0)):
            cam.get_status()
        assert cam.state.zoom == edelkrone_config.zoom_max

    def test_zoom_encoder_negative_maps_to_zero(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        # focus_enc = -24400 → norm = 0/24000 = 0.0 → zoom_min
        with patch.object(cam, "_call", return_value=_ptz_status_response(focus_enc=-24400)):
            cam.get_status()
        assert cam.state.zoom == edelkrone_config.zoom_min

    def test_returns_empty_dict_when_call_fails(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        with patch.object(cam, "_call", return_value=None):
            result = cam.get_status()
        assert result == {}

    def test_returns_empty_dict_on_bad_json_structure(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        bad_resp = _ok_response({"unexpected": "structure"})
        with patch.object(cam, "_call", return_value=bad_resp):
            result = cam.get_status()
        assert result == {}


# ---------------------------------------------------------------------------
# EdelkroneCamera.get_video_source
# ---------------------------------------------------------------------------


class TestGetVideoSource:
    def test_numeric_string_returns_int(self, edelkrone_config):
        edelkrone_config.video_device = "0"
        cam = EdelkroneCamera(edelkrone_config)
        assert cam.get_video_source() == 0
        assert isinstance(cam.get_video_source(), int)

    def test_non_zero_numeric_string(self, edelkrone_config):
        edelkrone_config.video_device = "2"
        cam = EdelkroneCamera(edelkrone_config)
        assert cam.get_video_source() == 2

    def test_device_path_returns_string(self, edelkrone_config):
        edelkrone_config.video_device = "/dev/video0"
        cam = EdelkroneCamera(edelkrone_config)
        assert cam.get_video_source() == "/dev/video0"
        assert isinstance(cam.get_video_source(), str)


# ---------------------------------------------------------------------------
# EdelkroneCamera.focus_move
# ---------------------------------------------------------------------------


class TestFocusMove:
    def test_sends_focus_manual_move_on_focus_link(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        with patch.object(cam, "_call") as mock_call:
            cam.focus_move(0.5)
        mock_call.assert_called_once_with(
            "bundle", "",
            {"command": "focusManualMove", "deltaEnc": 0.5},
            edelkrone_config.focus_link_id,
        )


# ---------------------------------------------------------------------------
# EdelkroneCamera._pair
# ---------------------------------------------------------------------------


class TestPair:
    def test_pair_calls_sequence_in_order(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        macs = ["AA:BB:CC:DD:EE:FF"]
        with patch.object(cam, "_call") as mock_call, \
             patch("autotrack.camera.edelkrone.time.sleep"):
            cam._pair(macs, "LINK123")

        commands = [c.args[2]["command"] for c in mock_call.call_args_list]
        assert commands == [
            "wirelessPairingScanStart",
            "wirelessPairingScanResults",
            "wirelessPairingCreateBundle",
            "wirelessPairingStatus",
        ]

    def test_pair_includes_all_macs_in_bundle(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        macs = ["AA:BB:CC:DD:EE:FF", "11:22:33:44:55:66"]
        with patch.object(cam, "_call") as mock_call, \
             patch("autotrack.camera.edelkrone.time.sleep"):
            cam._pair(macs, "LINK123")

        bundle_call = next(
            c for c in mock_call.call_args_list
            if c.args[2] and c.args[2].get("command") == "wirelessPairingCreateBundle"
        )
        assert bundle_call.args[2]["macList"] == macs
        assert bundle_call.args[2]["deviceCount"] == 2

    def test_pair_sleeps_after_scan_start(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        with patch.object(cam, "_call"), \
             patch("autotrack.camera.edelkrone.time.sleep") as mock_sleep:
            cam._pair(["AA:BB:CC:DD:EE:FF"], "LINK123")
        mock_sleep.assert_called_once_with(EdelkroneCamera._PAIRING_SCAN_DELAY)


# ---------------------------------------------------------------------------
# EdelkroneCamera.connect
# ---------------------------------------------------------------------------


class TestConnect:
    def test_connect_checks_pairing_for_both_links(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        with patch.object(cam, "_check_pairing") as mock_check, \
             patch.object(cam, "get_status", return_value={"pan": 0.0}):
            cam._head_paired = True
            cam._focus_paired = True
            cam.connect()
        assert mock_check.call_count == 2

    def test_connect_pairs_head_when_not_paired(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        with patch.object(cam, "_check_pairing"), \
             patch.object(cam, "_pair") as mock_pair, \
             patch.object(cam, "get_status", return_value={"pan": 0.0}):
            cam._head_paired = False
            cam._focus_paired = True
            cam.connect()
        assert mock_pair.called
        linked = mock_pair.call_args.args[1]
        assert linked == edelkrone_config.ptz_link_id

    def test_connect_returns_true_when_status_ok(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        with patch.object(cam, "_check_pairing"), \
             patch.object(cam, "get_status", return_value={"pan": 0.0}):
            cam._head_paired = True
            cam._focus_paired = True
            assert cam.connect() is True

    def test_connect_returns_false_when_status_empty(self, edelkrone_config):
        cam = EdelkroneCamera(edelkrone_config)
        with patch.object(cam, "_check_pairing"), \
             patch.object(cam, "get_status", return_value={}):
            cam._head_paired = True
            cam._focus_paired = True
            assert cam.connect() is False
