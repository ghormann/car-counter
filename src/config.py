"""Configuration dataclasses and loaders for app settings (YAML) and MQTT credentials (JSON)."""
import json
import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BoxedRegion:
    x: int
    y: int
    width: int
    height: int


@dataclass
class AppConfig:
    camera_name: str
    rtsps_url: str
    scan_regions: list[BoxedRegion]
    ignore_regions: list[BoxedRegion]
    vehicle_classes: list[str]
    detection_confidence: float
    stationary_seconds: int
    iou_threshold: float
    night_enhancement: bool
    target_fps: float
    model_path: str
    publish_interval_seconds: int
    mqtt_timeout_seconds: int
    mqtt_topic: str  # derived from mqtt_prefix/camera_name
    output_dir: Path
    image_save_cooldown_seconds: int


@dataclass
class MqttConfig:
    host: str
    port: int
    username: str
    password: str


_REQUIRED_APP_FIELDS = [
    'camera_name', 'rtsps_url', 'vehicle_classes', 'detection_confidence',
    'stationary_seconds', 'iou_threshold', 'night_enhancement',
    'target_fps', 'model_path', 'publish_interval_seconds', 'mqtt_timeout_seconds',
    'mqtt_prefix', 'output_dir', 'image_save_cooldown_seconds',
]


def load_app_config(path: str) -> AppConfig:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"App config not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}")

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    for field_name in _REQUIRED_APP_FIELDS:
        if field_name not in data:
            raise ValueError(f"Missing required config field: '{field_name}'")

    rtsps_url = data['rtsps_url']
    if not str(rtsps_url).startswith('rtsps://'):
        raise ValueError(f"rtsps_url must start with 'rtsps://': {rtsps_url!r}")

    output_dir = Path(data['output_dir']).expanduser()
    if not output_dir.exists():
        raise ValueError(f"output_dir does not exist: {output_dir}")

    scan_regions = [
        BoxedRegion(x=r['x'], y=r['y'], width=r['width'], height=r['height'])
        for r in data.get('scan_regions', [])
    ]

    ignore_regions = [
        BoxedRegion(x=r['x'], y=r['y'], width=r['width'], height=r['height'])
        for r in data.get('ignore_regions', [])
    ]

    return AppConfig(
        camera_name=str(data['camera_name']),
        rtsps_url=str(rtsps_url),
        scan_regions=scan_regions,
        ignore_regions=ignore_regions,
        vehicle_classes=list(data['vehicle_classes']),
        detection_confidence=float(data['detection_confidence']),
        stationary_seconds=int(data['stationary_seconds']),
        iou_threshold=float(data['iou_threshold']),
        night_enhancement=bool(data['night_enhancement']),
        target_fps=float(data['target_fps']),
        model_path=str(data['model_path']),
        publish_interval_seconds=int(data['publish_interval_seconds']),
        mqtt_timeout_seconds=int(data['mqtt_timeout_seconds']),
        mqtt_topic=f"{data['mqtt_prefix']}/{data['camera_name']}",
        output_dir=output_dir,
        image_save_cooldown_seconds=int(data['image_save_cooldown_seconds']),
    )


def load_mqtt_config(path: str) -> MqttConfig:
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"MQTT config not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")

    for field_name in ('host', 'port', 'username', 'password'):
        if field_name not in data:
            raise ValueError(f"Missing required MQTT config field: '{field_name}'")

    return MqttConfig(
        host=str(data['host']),
        port=int(data['port']),
        username=str(data['username']),
        password=str(data['password']),
    )
