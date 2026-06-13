"""Entry point for the car-counter service: wires together stream, detector, MQTT, and metrics."""
import logging
import os
import signal
import sys
import time

import torch
torch.backends.nnpack.enabled = False  # NNPACK is unsupported on this hardware; env var NNPACK_DISABLE has no effect in PyTorch
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_app_config, load_mqtt_config
from src.detector import Detector
from src.image_saver import ImageSaver
from src.metrics import Metrics
from src.mqtt_client import MqttClient
from src.stream import FrameStream

logger = logging.getLogger(__name__)

_shutdown = False


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    global _shutdown

    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}',
        level=getattr(logging, log_level, logging.INFO),
        stream=sys.stdout,
    )

    app_config_path = os.environ.get('APP_CONFIG_PATH', '/config/app/config.yaml')
    mqtt_config_path = os.environ.get('MQTT_CONFIG_PATH', '/config/mqtt/config.json')
    liveness_file = os.environ.get('LIVENESS_FILE', '/tmp/healthy')

    try:
        app_config = load_app_config(app_config_path)
        mqtt_config = load_mqtt_config(mqtt_config_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Startup validation failed: %s", e)
        sys.exit(1)

    logger.info("Config loaded: camera=%s", app_config.camera_name)

    metrics = Metrics(app_config.camera_name)
    metrics.start_server(9600)

    try:
        mqtt_client = MqttClient(
            host=mqtt_config.host,
            port=mqtt_config.port,
            username=mqtt_config.username,
            password=mqtt_config.password,
            topic=app_config.mqtt_topic,
            timeout_seconds=app_config.mqtt_timeout_seconds,
        )
        mqtt_client.connect()
    except RuntimeError as e:
        logger.error("MQTT startup failed: %s", e)
        sys.exit(1)

    detector = Detector(
        model_path=app_config.model_path,
        vehicle_classes=app_config.vehicle_classes,
        detection_confidence=app_config.detection_confidence,
        iou_threshold=app_config.iou_threshold,
        stationary_seconds=app_config.stationary_seconds,
        target_fps=app_config.target_fps,
        night_enhancement=app_config.night_enhancement,
        scan_regions=app_config.scan_regions,
        ignore_regions=app_config.ignore_regions,
    )

    image_saver = ImageSaver(
        output_dir=app_config.output_dir,
        camera_name=app_config.camera_name,
        cooldown_seconds=app_config.image_save_cooldown_seconds,
    )

    current_count = 0
    # Tracks camera stream state so every MQTT publish reflects current connectivity.
    # Set by the FrameStream callbacks below, not by MQTT events.
    current_status = "connected"
    last_publish_time = 0.0
    startup_image_saved = False

    # Called by FrameStream when the RTSP camera feed drops or fails to open.
    def on_disconnect():
        nonlocal current_status
        current_status = "disconnected"
        metrics.stream_connected.set(0)
        metrics.stream_reconnects.inc()
        mqtt_client.publish({
            "camera": app_config.camera_name,
            "count": None,
            "timestamp": _utcnow(),
            "status": "disconnected",
        })

    # Called by FrameStream when the RTSP camera feed (re)connects successfully.
    def on_reconnect():
        nonlocal current_status
        current_status = "connected"
        metrics.stream_connected.set(1)
        mqtt_client.publish({
            "camera": app_config.camera_name,
            "count": current_count if current_count >= 0 else None,
            "timestamp": _utcnow(),
            "status": "connected",
        })

    stream = FrameStream(
        rtsps_url=app_config.rtsps_url,
        on_disconnect=on_disconnect,
        on_reconnect=on_reconnect,
    )

    def _sigterm_handler(signum, frame_ref):
        global _shutdown
        logger.info("SIGTERM received — shutting down")
        _shutdown = True

    signal.signal(signal.SIGTERM, _sigterm_handler)

    stream.start()
    frame_interval = 1.0 / app_config.target_fps

    while not _shutdown:
        loop_start = time.monotonic()

        frame = stream.get_latest_frame()
        if frame is None:
            time.sleep(frame_interval)
            continue

        if not startup_image_saved:
            image_saver.save(frame, [], app_config.scan_regions, app_config.ignore_regions, prefix="startup_")
            startup_image_saved = True

        t0 = time.monotonic()
        with metrics.frame_processing_seconds.time():
            count, stationary_vehicles = detector.process_frame(frame)
        processing_seconds = round(time.monotonic() - t0, 3)

        metrics.frames_processed.inc()
        metrics.stationary_vehicles.set(count)

        try:
            Path(liveness_file).touch()
        except OSError:
            pass

        now = time.monotonic()

        if count != current_count:
            logger.info("Car count changed: %d -> %d", current_count, count)
            mqtt_client.publish({
                "camera": app_config.camera_name,
                "count": count,
                "timestamp": _utcnow(),
                "status": current_status,
                "processing_seconds": processing_seconds,
            })
            metrics.mqtt_messages_published.inc()
            last_publish_time = now

            image_saver.save(frame, stationary_vehicles, app_config.scan_regions, app_config.ignore_regions)
            metrics.images_saved.inc()

            current_count = count

        elif (now - last_publish_time) >= app_config.publish_interval_seconds:
            mqtt_client.publish({
                "camera": app_config.camera_name,
                "count": current_count,
                "timestamp": _utcnow(),
                "status": current_status,
                "processing_seconds": processing_seconds,
            })
            metrics.mqtt_messages_published.inc()
            last_publish_time = now

        metrics.mqtt_queue_depth.set(mqtt_client.queue_depth)
        metrics.mqtt_connected.set(1 if mqtt_client.is_connected else 0)

        loop_elapsed = time.monotonic() - loop_start
        metrics.uptime_minutes.inc(loop_elapsed / 60.0)

        sleep_time = frame_interval - loop_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    logger.info("Stopping stream")
    stream.stop()

    logger.info("Flushing MQTT queue")
    mqtt_client.flush(timeout=float(app_config.mqtt_timeout_seconds))

    mqtt_client.publish({
        "camera": app_config.camera_name,
        "count": None,
        "status": "shutdown",
        "timestamp": _utcnow(),
    })
    mqtt_client.flush(timeout=5.0)
    mqtt_client.disconnect()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
