import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from prometheus_client import (
    Counter, Gauge, Histogram, CollectorRegistry,
    generate_latest, CONTENT_TYPE_LATEST,
)


class Metrics:
    def __init__(self, camera_name: str):
        self._registry = CollectorRegistry()

        def c(name, doc):
            return Counter(name, doc, ['camera'], registry=self._registry).labels(camera=camera_name)

        def g(name, doc):
            return Gauge(name, doc, ['camera'], registry=self._registry).labels(camera=camera_name)

        def h(name, doc):
            return Histogram(name, doc, ['camera'], registry=self._registry).labels(camera=camera_name)

        self.frames_processed = c('car_counter_frames_processed_total', 'Total frames processed')
        self.frame_processing_seconds = h('car_counter_frame_processing_seconds', 'Inference time per frame')
        self.stationary_vehicles = g('car_counter_stationary_vehicles', 'Current stationary vehicle count')
        self.mqtt_messages_published = c('car_counter_mqtt_messages_published_total', 'Total MQTT messages published')
        self.mqtt_queue_depth = g('car_counter_mqtt_queue_depth', 'Pending messages in MQTT buffer')
        self.mqtt_connected = g('car_counter_mqtt_connected', 'MQTT broker connection status (0 or 1)')
        self.stream_connected = g('car_counter_stream_connected', 'RTSPS stream connection status (0 or 1)')
        self.stream_reconnects = c('car_counter_stream_reconnects_total', 'Total RTSPS stream reconnection attempts')
        self.images_saved = c('car_counter_images_saved_total', 'Total images saved to disk')
        self.uptime_minutes = c('car_counter_uptime_minutes_total', 'Total process uptime in minutes')

        self._registry_ref = self._registry

    def start_server(self, port: int = 9600):
        registry = self._registry_ref

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/metrics':
                    output = generate_latest(registry)
                    self.send_response(200)
                    self.send_header('Content-Type', CONTENT_TYPE_LATEST)
                    self.end_headers()
                    self.wfile.write(output)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, fmt, *args):
                pass

        server = HTTPServer(('', port), _Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
