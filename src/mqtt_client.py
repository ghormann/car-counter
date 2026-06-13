import json
import logging
import threading
import time
from collections import deque

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MqttClient:
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        topic: str,
        timeout_seconds: int,
    ):
        self._host = host
        self._port = port
        self._timeout_seconds = timeout_seconds
        self._topic = topic
        self._queue: deque[dict] = deque()
        self._queue_lock = threading.Lock()
        self._connected = False
        self._connect_event = threading.Event()

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.username_pw_set(username, password)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def queue_depth(self) -> int:
        with self._queue_lock:
            return len(self._queue)

    def connect(self):
        deadline = time.monotonic() + self._timeout_seconds
        backoff = 1
        while True:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Failed to connect to MQTT broker {self._host}:{self._port} "
                    f"within {self._timeout_seconds}s"
                )
            try:
                self._client.connect(self._host, self._port, keepalive=60)
                self._client.loop_start()
                remaining = deadline - time.monotonic()
                if self._connect_event.wait(timeout=max(0, remaining)):
                    return
            except Exception as e:
                logger.warning("MQTT connection attempt failed: %s", e)

            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Failed to connect to MQTT broker {self._host}:{self._port} "
                    f"within {self._timeout_seconds}s"
                )
            logger.warning("MQTT retry in %ds", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

    def publish(self, payload: dict):
        with self._queue_lock:
            self._queue.append(payload)
        self._drain_queue()

    def flush(self, timeout: float):
        deadline = time.monotonic() + timeout
        while self.queue_depth > 0 and time.monotonic() < deadline:
            self._drain_queue()
            time.sleep(0.05)

    def disconnect(self):
        self._client.loop_stop()
        self._client.disconnect()

    def _drain_queue(self):
        if not self._connected:
            return
        with self._queue_lock:
            while self._queue:
                payload = self._queue.popleft()
                # retain=True so new subscribers immediately receive the current state
                # without waiting for the next detection or heartbeat publish.
                self._client.publish(
                    self._topic,
                    json.dumps(payload),
                    qos=1,
                    retain=True,
                )

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self._connected = True
            self._connect_event.set()
            logger.info("MQTT connected to %s:%s", self._host, self._port)
            self._drain_queue()
        else:
            logger.warning("MQTT connect refused, reason: %s", reason_code)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self._connected = False
        self._connect_event.clear()
        logger.warning("MQTT disconnected, reason: %s", reason_code)
