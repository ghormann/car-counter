"""Tests for Prometheus metrics: counter/gauge updates and /metrics HTTP endpoint response."""
import time
import urllib.request
import urllib.error
import pytest
import threading

from src.metrics import Metrics


class TestMetrics:
    def test_all_metric_attributes_exist(self):
        m = Metrics('test-cam')
        assert hasattr(m, 'frames_processed')
        assert hasattr(m, 'frame_processing_seconds')
        assert hasattr(m, 'stationary_vehicles')
        assert hasattr(m, 'mqtt_messages_published')
        assert hasattr(m, 'mqtt_queue_depth')
        assert hasattr(m, 'mqtt_connected')
        assert hasattr(m, 'stream_connected')
        assert hasattr(m, 'stream_reconnects')
        assert hasattr(m, 'images_saved')
        assert hasattr(m, 'uptime_minutes')

    def test_counter_increments(self):
        m = Metrics('test-cam')
        m.frames_processed.inc()
        m.frames_processed.inc()

    def test_gauge_set(self):
        m = Metrics('test-cam')
        m.stationary_vehicles.set(3)
        m.mqtt_connected.set(1)

    def test_histogram_observe(self):
        m = Metrics('test-cam')
        with m.frame_processing_seconds.time():
            time.sleep(0.001)

    def test_metrics_http_server_returns_200(self, unused_tcp_port):
        m = Metrics('test-cam')
        m.start_server(unused_tcp_port)
        time.sleep(0.05)
        response = urllib.request.urlopen(f'http://localhost:{unused_tcp_port}/metrics')
        assert response.status == 200
        body = response.read().decode()
        assert 'car_counter_frames_processed_total' in body

    def test_metrics_http_server_404_for_other_paths(self, unused_tcp_port):
        m = Metrics('test-cam')
        m.start_server(unused_tcp_port)
        time.sleep(0.05)
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f'http://localhost:{unused_tcp_port}/health')
        assert exc_info.value.code == 404


@pytest.fixture
def unused_tcp_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]
