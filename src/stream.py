"""Background thread that reads frames from an RTSPS stream with exponential-backoff reconnection."""
import cv2
import threading
import time
import logging

logger = logging.getLogger(__name__)


def _next_backoff(current: int) -> int:
    return min(current * 2, 60)


class FrameStream:
    def __init__(
        self,
        rtsps_url: str,
        on_disconnect: callable = None,
        on_reconnect: callable = None,
    ):
        self._url = rtsps_url
        self._on_disconnect = on_disconnect or (lambda: None)
        self._on_reconnect = on_reconnect or (lambda: None)
        self._latest_frame = None
        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self._thread = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_latest_frame(self):
        with self._lock:
            return self._latest_frame

    def _read_loop(self):
        backoff = 1
        while self._running:
            cap = cv2.VideoCapture(self._url)
            if not cap.isOpened():
                cap.release()
                if self._connected:
                    self._connected = False
                    self._on_disconnect()
                elif backoff == 1:
                    # Fire on_disconnect for the very first failed open so the broker
                    # immediately reflects that the camera is unavailable at startup.
                    self._on_disconnect()
                logger.warning("Stream unavailable, retrying in %ds", backoff)
                time.sleep(backoff)
                backoff = _next_backoff(backoff)
                continue

            if not self._connected:
                self._connected = True
                backoff = 1
                logger.info("Stream connected: %s", self._url)
                self._on_reconnect()

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    self._connected = False
                    logger.warning("Frame read failed, reconnecting")
                    self._on_disconnect()
                    break
                with self._lock:
                    self._latest_frame = frame

            cap.release()
