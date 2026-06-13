# Car Counter

Monitors RTSPS camera feeds and counts stationary motor vehicles (cars, trucks, buses). Publishes counts via MQTT and saves annotated screenshots on count changes. Designed to run as a single-process-per-camera Kubernetes service on CPU-only hardware.

## Features

- **YOLOv8** object detection (configurable model size; `yolov8x` default)
- **Stationary vehicle tracking** — only counts vehicles present for a configurable duration (IoU-based across frames)
- **Night/IR preprocessing** — automatic CLAHE enhancement for dark color and IR frames
- **MQTT publishing** — on count change, heartbeat, disconnect, and shutdown
- **Annotated screenshots** — saved on count change with bounding boxes, scan region overlays (green), and ignore region overlays (blue)
- **Prometheus metrics** — exposed on port 9600 at `/metrics`
- **Kubernetes-ready** — liveness/readiness probes, SIGTERM handling, structured JSON logs

## Quick Start

### Docker Compose (local testing)

1. Copy example configs and edit for your environment:

   ```bash
   cp config/example-config.yaml config/app-config.yaml
   cp config/example-mqtt-config.json config/mqtt-config.json
   ```

2. Create the output directory:

   ```bash
   mkdir output
   ```

3. Build and run:

   ```bash
   docker compose up --build
   ```

Metrics are available at `http://localhost:9600/metrics`.

## Configuration

Full field-by-field documentation is in [`docs/config-reference.md`](docs/config-reference.md).

### Environment Variables

| Variable           | Default                    | Description                                      |
| ------------------ | -------------------------- | ------------------------------------------------ |
| `APP_CONFIG_PATH`  | `/config/app/config.yaml`  | Main application YAML config                     |
| `MQTT_CONFIG_PATH` | `/config/mqtt/config.json` | MQTT credentials JSON                            |
| `LOG_LEVEL`        | `INFO`                     | Logging level (`DEBUG`, `INFO`, `WARN`, `ERROR`) |
| `LIVENESS_FILE`    | `/tmp/healthy`             | Path touched on each successful frame process    |

### Application Config (`config.yaml`)

```yaml
camera_name: driveway # Used in MQTT messages, image paths, and metrics
rtsps_url: rtsps://192.168.1.10/stream # Credentials optional: rtsps://user:pass@host/stream

# Scan regions (optional — omit to scan entire frame)
scan_regions:
  - { x: 100, y: 200, width: 400, height: 300 }

# Ignore regions (optional — vehicles ≥95% inside are excluded from counting)
# ignore_regions:
#   - { x: 0, y: 0, width: 100, height: 50 }

vehicle_classes: [car, truck, bus] # COCO class names to detect
detection_confidence: 0.4 # Minimum confidence (lower = better night recall)
stationary_seconds: 3 # Seconds a vehicle must stay in place to count
iou_threshold: 0.5 # IoU overlap to match a vehicle across frames

night_enhancement: true # Enable CLAHE preprocessing for low-light frames
target_fps: 1 # Frames to process per second

model_path: yolov8x.pt # YOLOv8 weights (baked into Docker image)
mqtt_prefix: car-counter # MQTT topic prefix; topic becomes <prefix>/<camera_name>

publish_interval_seconds: 5 # Heartbeat interval when count has not changed
mqtt_timeout_seconds: 60 # Max wait for MQTT before exiting

output_dir: ~/output # Base image output directory (must exist at startup)
image_save_cooldown_seconds: 30 # Minimum seconds between saved images
```

### MQTT Credentials (`config.json`)

```json
{
  "host": "server.com",
  "port": 1883,
  "username": "username",
  "password": "password"
}
```

## MQTT Messages

Published with QoS 1 and `retain: true`.

| Event                     | Payload                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------- |
| Count change or heartbeat | `{"camera": "driveway", "count": 3, "timestamp": "2026-06-13T14:32:01Z"}`             |
| Stream disconnected       | `{"camera": "driveway", "count": null, "status": "disconnected", "timestamp": "..."}` |
| Graceful shutdown         | `{"camera": "driveway", "count": null, "status": "shutdown", "timestamp": "..."}`     |

## Prometheus Metrics

All metrics include a `camera` label.

| Metric                                      | Type      | Description                        |
| ------------------------------------------- | --------- | ---------------------------------- |
| `car_counter_frames_processed_total`        | Counter   | Total frames processed             |
| `car_counter_frame_processing_seconds`      | Histogram | Inference time per frame           |
| `car_counter_stationary_vehicles`           | Gauge     | Current stationary vehicle count   |
| `car_counter_mqtt_messages_published_total` | Counter   | Total MQTT messages published      |
| `car_counter_mqtt_queue_depth`              | Gauge     | Pending messages in MQTT buffer    |
| `car_counter_mqtt_connected`                | Gauge     | MQTT broker connection (0/1)       |
| `car_counter_stream_connected`              | Gauge     | RTSPS stream connection (0/1)      |
| `car_counter_stream_reconnects_total`       | Counter   | Total stream reconnection attempts |
| `car_counter_images_saved_total`            | Counter   | Total images saved to disk         |
| `car_counter_uptime_minutes_total`          | Counter   | Total process uptime in minutes    |

## Image Output

Saved to `{output_dir}/{camera_name}/{year}/{month}/{day}/{timestamp}.jpg` (JPEG, 85% quality).

- A `startup_{timestamp}.jpg` is always saved on process start.
- Annotated with **red** boxes for stationary vehicles and **green** boxes for scan regions.
- Cooldown enforced between saves to limit disk writes on busy scenes.

## Kubernetes Deployment

See [`docs/kubernetes-deployment.md`](docs/kubernetes-deployment.md) for Deployment, ConfigMap, and Secret manifests.

**Health probes:**

- **Liveness**: `LIVENESS_FILE` modified within the last 30 seconds
- **Readiness**: HTTP GET `/metrics` on port 9600

## Project Structure

```
car-counter/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── config/
│   ├── example-config.yaml          # Example app config
│   └── example-mqtt-config.json     # Example MQTT config
├── docs/
│   ├── requirements.md
│   ├── config-reference.md
│   └── kubernetes-deployment.md
├── src/
│   ├── __main__.py                  # Entrypoint
│   ├── config.py                    # Config loading and validation
│   ├── detector.py                  # YOLOv8 inference and stationary tracking
│   ├── stream.py                    # RTSPS capture and reconnection
│   ├── mqtt_client.py               # MQTT with buffering and retry
│   ├── image_saver.py               # Annotated screenshot saving
│   └── metrics.py                   # Prometheus metrics server
└── tests/
```

## Hardware Requirements

- CPU-only (no GPU required)
- Target: ≥ 1 frame processed per second
- Supports variable camera resolutions (1024×768, 2K, 4K)
