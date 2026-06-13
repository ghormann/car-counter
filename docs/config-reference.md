# Configuration Reference

## Environment Variables

These control where config files are loaded from and how the process behaves at startup. Set them in your Kubernetes manifest, `docker-compose.yml`, or shell environment.

| Variable           | Default                    | Description                                                                |
| ------------------ | -------------------------- | -------------------------------------------------------------------------- |
| `APP_CONFIG_PATH`  | `/config/app/config.yaml`  | Path to the main application YAML config                                   |
| `MQTT_CONFIG_PATH` | `/config/mqtt/config.json` | Path to the MQTT credentials JSON                                          |
| `LOG_LEVEL`        | `INFO`                     | Logging verbosity: `DEBUG`, `INFO`, `WARN`, or `ERROR`                     |
| `LIVENESS_FILE`    | `/tmp/healthy`             | File touched on each successful frame process (used by k8s liveness probe) |

---

## Application Config (`config.yaml`)

All fields are **required** unless marked optional.

### Camera

| Field         | Type   | Example                       | Description                                                                                                          |
| ------------- | ------ | ----------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `camera_name` | string | `driveway`                    | Identifier used in MQTT messages, image output paths, and Prometheus metric labels. Must be unique across instances. |
| `rtsps_url`   | string | `rtsps://192.168.1.10/stream` | RTSPS stream URL. Credentials are optional: `rtsps://user:pass@host/stream`. Must start with `rtsps://`.             |

### Scan Regions (optional)

```yaml
scan_regions:
  - { x: 100, y: 200, width: 400, height: 300 }
  - { x: 500, y: 100, width: 200, height: 150 }
```

Pixel coordinates relative to the full frame resolution. If omitted, the entire frame is scanned. Vehicles overlapping multiple regions are counted once.

| Sub-field | Type | Description                       |
| --------- | ---- | --------------------------------- |
| `x`       | int  | Left edge of the region in pixels |
| `y`       | int  | Top edge of the region in pixels  |
| `width`   | int  | Width of the region in pixels     |
| `height`  | int  | Height of the region in pixels    |

### Ignore Regions (optional)

```yaml
ignore_regions:
  - { x: 0, y: 0, width: 100, height: 50 }
```

Pixel coordinates relative to the full frame resolution. If a detected vehicle's bounding box is 95% or more inside any ignore region, it is excluded from counting. Useful for masking areas with frequent false positives (e.g., a neighbor's driveway at the edge of frame).

Ignore regions are rendered as **blue** outlines labeled `exclude` on annotated images.

| Sub-field | Type | Description                        |
| --------- | ---- | ---------------------------------- |
| `x`       | int  | Left edge of the region in pixels  |
| `y`       | int  | Top edge of the region in pixels   |
| `width`   | int  | Width of the region in pixels      |
| `height`  | int  | Height of the region in pixels     |

### Vehicle Detection

| Field                  | Type            | Default             | Description                                                                                                                               |
| ---------------------- | --------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `vehicle_classes`      | list of strings | `[car, truck, bus]` | COCO class names to detect and count.                                                                                                     |
| `detection_confidence` | float (0.0–1.0) | `0.4`               | Minimum YOLO confidence score to accept a detection. Lower values improve recall in night conditions at the cost of more false positives. |
| `stationary_seconds`   | int             | `3`                 | A vehicle must be continuously detected with sufficient IoU overlap for this many seconds before it is counted.                           |
| `iou_threshold`        | float (0.0–1.0) | `0.5`               | Minimum IoU (intersection-over-union) to match a detection in the current frame to a tracked vehicle in the previous frame.               |

### Night / Low-Light Enhancement

| Field               | Type | Default | Description                                                                                                    |
| ------------------- | ---- | ------- | -------------------------------------------------------------------------------------------------------------- |
| `night_enhancement` | bool | `true`  | Enable automatic CLAHE preprocessing for frames detected as dark. Applied per-frame with no time-of-day logic. |

Enhancement is selected automatically based on per-frame brightness and saturation:

- **Dark color frame** (mean brightness < 80, saturation ≥ 30): CLAHE on the LAB L-channel (`clipLimit=2.0`)
- **Dark IR frame** (mean brightness < 80, saturation < 30): grayscale CLAHE (`clipLimit=4.0`) followed by unsharp masking

### Frame Processing

| Field        | Type | Default | Description                                                                                                                                  |
| ------------ | ---- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `target_fps` | int  | `1`     | Frames to process per second. All frames are read from the stream to keep the buffer clear; only one per interval is passed to the detector. |

### Model

| Field        | Type   | Default      | Description                                                                                                                          |
| ------------ | ------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| `model_path` | string | `yolov8x.pt` | Path to YOLOv8 weights. The default `yolov8x` weights are baked into the Docker image. Mount a custom path to use a different model. |

### MQTT

| Field                      | Type   | Default | Description                                                                                                                                                        |
| -------------------------- | ------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mqtt_prefix`              | string | —       | MQTT topic prefix. The camera name is appended automatically, so `car-counter` becomes `car-counter/driveway`.                                                     |
| `publish_interval_seconds` | int    | `5`     | Heartbeat publish interval. A heartbeat is sent only if no count-change publish has occurred within this interval; the timer resets on every count-change publish. |
| `mqtt_timeout_seconds`     | int    | `60`    | Maximum seconds to wait for MQTT connection on startup or during flush before exiting with an error.                                                               |

### Image Saving

| Field                         | Type   | Default | Description                                                                                                       |
| ----------------------------- | ------ | ------- | ----------------------------------------------------------------------------------------------------------------- |
| `output_dir`                  | string | —       | Base directory for saved images. Must exist at startup; the process exits if it does not. Supports `~` expansion. |
| `image_save_cooldown_seconds` | int    | `30`    | Minimum seconds between images saved on count change. Prevents excessive disk writes during rapid changes.        |

---

## MQTT Credentials (`config.json`)

Shared with other services. Only the four fields below are read; all other fields in the JSON are ignored.

| Field      | Type   | Description                         |
| ---------- | ------ | ----------------------------------- |
| `host`     | string | MQTT broker hostname or IP          |
| `port`     | int    | MQTT broker port (typically `1883`) |
| `username` | string | MQTT username                       |
| `password` | string | MQTT password                       |

---

## Startup Validation

The process exits with a descriptive error on startup if any of the following are true:

- `APP_CONFIG_PATH` file does not exist or cannot be parsed as YAML
- `MQTT_CONFIG_PATH` file does not exist or cannot be parsed as JSON
- Any required YAML field is missing
- `rtsps_url` does not start with `rtsps://`
- `output_dir` does not exist
