import threading
import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.stream import FrameStream, _next_backoff


class TestNextBackoff:
    def test_doubles_each_call(self):
        assert _next_backoff(1) == 2
        assert _next_backoff(2) == 4
        assert _next_backoff(4) == 8
        assert _next_backoff(8) == 16
        assert _next_backoff(16) == 32
        assert _next_backoff(32) == 60

    def test_caps_at_60(self):
        assert _next_backoff(60) == 60
        assert _next_backoff(120) == 60


class TestFrameStream:
    def test_get_latest_frame_returns_none_before_start(self):
        stream = FrameStream("rtsps://host/stream")
        assert stream.get_latest_frame() is None

    def test_is_connected_false_before_start(self):
        stream = FrameStream("rtsps://host/stream")
        assert stream.is_connected is False

    def test_on_disconnect_called_when_stream_unavailable(self, mocker):
        disconnect_event = threading.Event()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mocker.patch('cv2.VideoCapture', return_value=mock_cap)

        sleep_calls = []

        def controlled_sleep(t):
            sleep_calls.append(t)
            if len(sleep_calls) >= 2:
                stream.stop()

        mocker.patch('src.stream.time.sleep', side_effect=controlled_sleep)

        stream = FrameStream(
            "rtsps://host/stream",
            on_disconnect=lambda: disconnect_event.set(),
        )
        stream.start()
        assert disconnect_event.wait(timeout=2.0)
        stream.stop()

    def test_backoff_sequence_on_repeated_failures(self, mocker):
        sleep_args = []
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mocker.patch('cv2.VideoCapture', return_value=mock_cap)

        def controlled_sleep(t):
            sleep_args.append(t)
            if len(sleep_args) >= 3:
                stream.stop()

        mocker.patch('src.stream.time.sleep', side_effect=controlled_sleep)
        stream = FrameStream("rtsps://host/stream")
        stream.start()
        stream._thread.join(timeout=2.0)

        assert sleep_args[:3] == [1, 2, 4]

    def test_on_reconnect_called_when_stream_connects(self, mocker):
        reconnect_event = threading.Event()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, frame), (False, None)]
        mocker.patch('cv2.VideoCapture', return_value=mock_cap)
        mocker.patch('src.stream.time.sleep')

        stream = FrameStream(
            "rtsps://host/stream",
            on_reconnect=lambda: reconnect_event.set(),
        )
        stream.start()
        assert reconnect_event.wait(timeout=2.0)
        stream.stop()

    def test_latest_frame_set_after_successful_read(self, mocker):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[0, 0] = [42, 43, 44]
        frame_available = threading.Event()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True

        def read_side_effect():
            if not frame_available.is_set():
                frame_available.set()
                return True, frame
            stream.stop()
            return False, None

        mock_cap.read.side_effect = read_side_effect
        mocker.patch('cv2.VideoCapture', return_value=mock_cap)
        mocker.patch('src.stream.time.sleep')

        stream = FrameStream("rtsps://host/stream")
        stream.start()
        assert frame_available.wait(timeout=2.0)
        time.sleep(0.02)

        result = stream.get_latest_frame()
        assert result is not None
        np.testing.assert_array_equal(result, frame)
        stream.stop()
