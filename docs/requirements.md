# Car Counter — Requirements

## Purpose

Monitor RTSPS camera feeds and count stationary motor vehicles (cars, trucks, buses). Publish counts via MQTT and save annotated screenshots on count changes.

---

## Functional Requirements

### Vehicle Detection

- Use **YOLOv8x** (extra-large) via the `ultralytics` library for object detection; smaller variants are insufficient for night IR classification
- Detect these COCO classes only: `car`, `truck`, `bus` (configurable via YAML)
- Default confidence threshold: `0.4` (configurable); lower than typical to accommodate night conditions
- Only count **stationary** vehicles: a vehicle must be continuously detected with IoU ≥ 0.5 overlap across frames for at least `stationary_seconds` (default: 3s, configurable)
- Count is the **total** across all configured scan regions; vehicles overlapping multiple regions count once
- If no scan regions are configured, scan the entire frame
- Optionally define `ignore_regions`: a vehicle whose bounding box is ≥95% inside any ignore region is excluded from counting

### Night / Low-Light Detection

- Apply preprocessing to improve detection in low-light conditions; controlled by `night_enhancement` config flag
- Two enhancement paths, selected automatically per frame:
  - **Dark color frame** (mean brightness < 80, mean saturation ≥ 30): CLAHE on the LAB L-channel (`clipLimit=2.0`)
  - **Dark IR frame** (mean brightness < 80, mean saturation < 30): grayscale CLAHE (`clipLimit=4.0`) followed by unsharp masking to recover edge detail
- Brightness and saturation thresholds are internal constants, not operator-tunable
- All detection per-frame automatically; no time-of-day logic is used

### Frame Processing

- Read all frames from the RTSPS stream continuously to keep the buffer clear
- Process frames at `target_fps` (default: 1, configurable); skip frames between processing intervals
- Target at minimum 1 FPS processing on CPU-only hardware

### MQTT Publishing

- Publish a JSON message when the stationary vehicle count changes (immediately)
- Also publish a heartbeat every `publish_interval_seconds` (default: 5s, configurable) **only if no count-change publish has occurred** within that interval — the heartbeat timer resets on every count-change publish
- Message payload:
  ```json
  { "camera": "driveway", "count": 3, "timestamp": "2026-06-13T14:32:01Z" }
  ```
- Publish `"status": "disconnected"` when the RTSPS stream is unavailable:
  ```json
  {
    "camera": "driveway",
    "count": null,
    "timestamp": "...",
    "status": "disconnected"
  }
  ```
- Publish on graceful shutdown:
  ```json
  {
    "camera": "driveway",
    "count": null,
    "status": "shutdown",
    "timestamp": "..."
  }
  ```
- QoS level: 1 (at least once delivery)
- Retain flag: `true` (broker retains last message for late subscribers)

### Image Saving

- Save a JPEG screenshot (85% quality) whenever the stationary vehicle count changes, subject to `image_save_cooldown_seconds` (configurable) — no more than one image per cooldown period
- Always save a startup image on process start, prefixed with `startup_`
- Annotate saved images with:
  - **Red** bounding boxes around detected stationary vehicles
  - **Green** bounding boxes showing configured scan regions
  - **Blue** bounding boxes (labeled "exclude") showing configured ignore regions
- Image path: `{output_dir}/{camera_name}/{year}/{month}/{day}/{timestamp}.jpg`
  - Startup image: `{output_dir}/{camera_name}/{year}/{month}/{day}/startup_{timestamp}.jpg`
  - Timestamp format: `YYYYMMDD_HHMMSS`
- `output_dir` must exist at startup or the process exits with a meaningful error
- Date subdirectories are created automatically on demand
- If a directory cannot be created or an image cannot be saved (permissions, disk full), log at ERROR level and continue running — image save failures are non-fatal

### Stream Handling

- Support RTSPS streams; credentials embedded in URL are optional (`rtsps://user:pass@host/stream` or `rtsps://host/stream`)
- On stream disconnect: retry with exponential backoff (1s, 2s, 4s… up to 60s max), log the disconnect, and publish a `"status": "disconnected"` MQTT message
- On reconnect: resume processing normally and publish `"status": "connected"`

### MQTT Reliability

- Buffer all outgoing MQTT messages in memory
- On MQTT broker unavailability: retry connection with exponential backoff, log failures at WARN level
- If an MQTT session cannot be established and the queue cannot be flushed within `mqtt_timeout_seconds` (default: 60s, configurable), exit with a meaningful error message

### Graceful Shutdown

- Handle SIGTERM (sent by k8s on pod termination):
  1. Stop processing new frames
  2. Flush the MQTT message queue (respecting `mqtt_timeout_seconds`)
  3. Publish a final `"status": "shutdown"` message
  4. Exit cleanly

---

## Configuration

### Config File Locations

Specified via environment variables with defaults:

| Environment Variable | Default                    | Description                                   |
| -------------------- | -------------------------- | --------------------------------------------- |
| `APP_CONFIG_PATH`    | `/config/app/config.yaml`  | Main application YAML config                  |
| `MQTT_CONFIG_PATH`   | `/config/mqtt/config.json` | MQTT credentials JSON (shared secret)         |
| `LOG_LEVEL`          | `INFO`                     | Logging level (DEBUG, INFO, WARN, ERROR)      |
| `LIVENESS_FILE`      | `/tmp/healthy`             | Path touched on each successful frame process |

### Application YAML Config (`config.yaml`)

All fields are required unless marked optional.

```yaml
# Camera identity and stream source
camera_name: driveway # Used in MQTT messages, image paths, and metric labels
rtsps_url: rtsps://192.168.1.10/stream # RTSPS stream URL; credentials optional

# Scan regions (optional — defaults to full frame if omitted)
# Pixel coordinates relative to full frame resolution
scan_regions:
  - { x: 100, y: 200, width: 400, height: 300 }
  - { x: 500, y: 100, width: 200, height: 150 }

# Ignore regions (optional — vehicles ≥95% inside are excluded)
# ignore_regions:
#   - { x: 0, y: 0, width: 100, height: 50 }

# Vehicle detection
vehicle_classes: [car, truck, bus] # COCO class names to count
detection_confidence: 0.4 # Minimum detection confidence (0.0–1.0)
stationary_seconds: 3 # Seconds a vehicle must remain in place to be counted
iou_threshold: 0.5 # IoU overlap required to match vehicle across frames

# Night / low-light enhancement
night_enhancement: true # Enable CLAHE preprocessing for low-light frames

# Frame processing
target_fps: 1 # Frames to process per second (read all, process at this rate)

# Model
model_path: yolov8x.pt # Path to YOLOv8 weights file (baked into Docker image)

# MQTT
publish_interval_seconds: 5 # Heartbeat publish interval when count has not changed
mqtt_timeout_seconds: 60 # Max seconds to wait for MQTT connection before exiting

# Image saving
output_dir: ~/output # Base output directory; must exist at startup
image_save_cooldown_seconds: 30 # Minimum seconds between saved images on count change
```

### MQTT Credentials JSON (`config.json`)

Shared with other services. Only `host`, `port`, `username`, and `password` are read; all other fields are ignored.

```json
{
  "username": "username",
  "password": "password",
  "port": 1883,
  "host": "server.com"
}
```

### Startup Validation

The process exits with a meaningful error if any of the following are true:

- `APP_CONFIG_PATH` file does not exist or cannot be parsed
- `MQTT_CONFIG_PATH` file does not exist or cannot be parsed
- Any required YAML field is missing
- `output_dir` does not exist
- RTSPS URL is missing or malformed

---

## Observability

### Logging

- Structured JSON logs to stdout
- Log level controlled via `LOG_LEVEL` environment variable
- Key events logged: startup, config loaded, stream connect/disconnect, count changes, image saves, MQTT publish failures, directory creation

### Prometheus Metrics

Exposed on port **9600** at `/metrics`. All metrics include a `camera` label.

| Metric                                      | Type      | Description                              |
| ------------------------------------------- | --------- | ---------------------------------------- |
| `car_counter_frames_processed_total`        | Counter   | Total frames processed                   |
| `car_counter_frame_processing_seconds`      | Histogram | Inference time per frame                 |
| `car_counter_stationary_vehicles`           | Gauge     | Current stationary vehicle count         |
| `car_counter_mqtt_messages_published_total` | Counter   | Total MQTT messages published            |
| `car_counter_mqtt_queue_depth`              | Gauge     | Pending messages in MQTT buffer          |
| `car_counter_mqtt_connected`                | Gauge     | MQTT broker connection status (0 or 1)   |
| `car_counter_stream_connected`              | Gauge     | RTSPS stream connection status (0 or 1)  |
| `car_counter_stream_reconnects_total`       | Counter   | Total RTSPS stream reconnection attempts |
| `car_counter_images_saved_total`            | Counter   | Total images saved to disk               |
| `car_counter_uptime_minutes_total`          | Counter   | Total process uptime in minutes          |

### k8s Health Probes

- **Liveness probe**: check that `LIVENESS_FILE` (`/tmp/healthy`) was modified within the last 30 seconds — indicates frame processing is active
- **Readiness probe**: HTTP GET `/metrics` on port 9600 returns 200

---

## Project Structure

```
car-counter/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── config/
│   ├── example-config.yaml          # Documented example app config (committed)
│   ├── example-mqtt-config.json     # Documented example MQTT config (committed)
│   ├── app-config.yaml              # Local config for docker-compose (gitignored)
│   └── mqtt-config.json             # Local MQTT config for docker-compose (gitignored)
├── docs/
│   ├── requirements.md              # This document
│   ├── config-reference.md          # Full YAML field documentation
│   └── kubernetes-deployment.md     # k8s deployment instructions
├── src/
│   ├── __main__.py                  # Entrypoint
│   ├── config.py                    # Config loading and validation
│   ├── detector.py                  # YOLOv8 inference and stationary vehicle tracking
│   ├── stream.py                    # RTSPS capture and reconnection
│   ├── mqtt_client.py               # MQTT with buffering and retry
│   ├── image_saver.py               # Screenshot saving with bounding box annotations
│   └── metrics.py                   # Prometheus metrics server
└── tests/
    ├── test_detector.py
    ├── test_config.py
    ├── test_image_saver.py
    ├── test_stream.py
    ├── test_mqtt_client.py
    └── data/
        ├── test_cases.yaml          # Parametrized test definitions
        └── images/
            ├── daylight_sample.jpg  # Placeholder: color, bright scene
            ├── twilight_sample.jpg  # Placeholder: color, dim scene
            ├── ir_sample.jpg        # Placeholder: grayscale IR scene
            └── empty_scene.jpg      # Placeholder: no vehicles
```

---

## Deployment

### Single Process Per Camera

Each process handles exactly one camera. Run multiple instances (containers) for multiple cameras, each with its own config.

### Docker

- Base image: `python:3.12-slim`
- YOLOv8n model weights baked into the image at build time
- Model path configurable via YAML to allow mounting a custom/larger model

### docker-compose (Local Testing)

Mount local config files as volumes. See `docker-compose.yml`.

### Kubernetes

See `docs/kubernetes-deployment.md` for full deployment instructions including Deployment, ConfigMap, and Secret manifests.

---

## Hardware Constraints

- CPU-only (no GPU)
- Target: ≥ 1 frame processed per second
- Supports variable camera resolutions: 1024×768, 2K, 4K
