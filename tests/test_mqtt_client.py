import json
import time
import threading
import pytest
from unittest.mock import MagicMock, patch, call

from src.mqtt_client import MqttClient


def make_client(mocker, timeout_seconds=60):
    mock_paho = MagicMock()
    mocker.patch('src.mqtt_client.mqtt.Client', return_value=mock_paho)
    return MqttClient('host', 1883, 'user', 'pass', 'driveway', timeout_seconds), mock_paho


class TestMqttClientQueue:
    def test_queue_depth_is_0_on_init(self, mocker):
        client, _ = make_client(mocker)
        assert client.queue_depth == 0

    def test_publish_enqueues_when_disconnected(self, mocker):
        client, _ = make_client(mocker)
        assert not client.is_connected
        client.publish({"count": 1})
        assert client.queue_depth == 1

    def test_publish_drains_immediately_when_connected(self, mocker):
        client, mock_paho = make_client(mocker)
        client._connected = True
        client.publish({"count": 1})
        mock_paho.publish.assert_called_once()
        assert client.queue_depth == 0

    def test_publish_payload_serialized_as_json(self, mocker):
        client, mock_paho = make_client(mocker)
        client._connected = True
        payload = {"camera": "driveway", "count": 3, "timestamp": "2026-06-13T00:00:00Z"}
        client.publish(payload)
        call_args = mock_paho.publish.call_args
        assert call_args[0][0] == 'car-counter/driveway'
        assert json.loads(call_args[0][1]) == payload
        assert call_args[1]['qos'] == 1
        assert call_args[1]['retain'] is True

    def test_flush_drains_queue_when_connected(self, mocker):
        client, mock_paho = make_client(mocker)
        client._queue.append({"count": 1})
        client._queue.append({"count": 2})
        client._connected = True
        client.flush(timeout=1.0)
        assert client.queue_depth == 0
        assert mock_paho.publish.call_count == 2

    def test_flush_leaves_queue_intact_when_disconnected(self, mocker):
        client, _ = make_client(mocker)
        client._queue.append({"count": 1})
        client._connected = False
        start = time.monotonic()
        client.flush(timeout=0.1)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5
        assert client.queue_depth == 1

    def test_multiple_publishes_maintain_order(self, mocker):
        client, mock_paho = make_client(mocker)
        client._connected = True
        for i in range(3):
            client.publish({"count": i})
        published = [json.loads(c[0][1]) for c in mock_paho.publish.call_args_list]
        assert [p['count'] for p in published] == [0, 1, 2]


class TestMqttClientConnect:
    def test_connect_succeeds_when_on_connect_fires(self, mocker):
        mock_paho = MagicMock()

        def loop_start_side_effect():
            # Simulate broker accepting connection
            client._connected = True
            client._connect_event.set()

        mock_paho.loop_start.side_effect = loop_start_side_effect
        mocker.patch('src.mqtt_client.mqtt.Client', return_value=mock_paho)
        client = MqttClient('host', 1883, 'user', 'pass', 'driveway', timeout_seconds=5)
        client.connect()
        assert client.is_connected

    def test_connect_raises_after_timeout(self, mocker):
        mock_paho = MagicMock()
        mocker.patch('src.mqtt_client.mqtt.Client', return_value=mock_paho)
        mocker.patch('src.mqtt_client.time.sleep')

        client = MqttClient('host', 1883, 'user', 'pass', 'driveway', timeout_seconds=0)
        # connect_event never set → should timeout immediately
        with pytest.raises(RuntimeError, match="Failed to connect to MQTT"):
            client.connect()

    def test_on_connect_drains_queued_messages(self, mocker):
        mock_paho = MagicMock()
        mocker.patch('src.mqtt_client.mqtt.Client', return_value=mock_paho)
        client = MqttClient('host', 1883, 'user', 'pass', 'driveway', timeout_seconds=5)
        client._queue.append({"count": 7})

        # Simulate on_connect callback
        client._on_connect(mock_paho, None, None, 0, None)

        assert client.queue_depth == 0
        mock_paho.publish.assert_called_once()

    def test_on_disconnect_clears_connected_flag(self, mocker):
        mock_paho = MagicMock()
        mocker.patch('src.mqtt_client.mqtt.Client', return_value=mock_paho)
        client = MqttClient('host', 1883, 'user', 'pass', 'driveway', timeout_seconds=5)
        client._connected = True

        client._on_disconnect(mock_paho, None, None, 0, None)

        assert not client.is_connected
