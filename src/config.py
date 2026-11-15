"""Configuration dataclasses and loaders for app settings (YAML) and MQTT credentials (JSON)."""
import json
import math
import yaml
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import Polygon


@dataclass
class BoxedRegion:
    """Axis-aligned rectangular region in full-frame pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def points(self) -> list[tuple[int, int]]:
        """Corner points, clockwise from the top-left."""
        return [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
            (self.x, self.y + self.height),
        ]


@dataclass
class PolygonRegion:
    """Arbitrary simple polygon region in full-frame pixel coordinates."""

    points: list[tuple[int, int]]


Region = BoxedRegion | PolygonRegion


@dataclass
class AppConfig:
    camera_name: str
    rtsps_url: str
    scan_regions: list[Region]
    ignore_regions: list[Region]
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
    inference_imgsz: int = 640  # YOLO input size; larger finds smaller (more distant) vehicles


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


def _parse_region(raw, field_name: str) -> Region:
    """Parse one region entry: either ``{x, y, width, height}`` or ``{points: [[x, y], ...]}``."""
    if not isinstance(raw, dict):
        raise ValueError(f"Each entry in '{field_name}' must be a mapping: {raw!r}")
    if 'points' in raw:
        raw_points = raw['points']
        try:
            points = [(int(p[0]), int(p[1])) for p in raw_points if len(p) == 2]
        except (TypeError, ValueError):
            raise ValueError(f"Invalid points in '{field_name}': {raw_points!r}")
        if len(points) != len(raw_points) or len(points) < 3:
            raise ValueError(
                f"Polygon in '{field_name}' needs at least 3 [x, y] points: {raw_points!r}"
            )
        polygon = Polygon(points)
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError(
                f"Polygon in '{field_name}' is not a simple, non-self-intersecting shape: {raw_points!r}"
            )
        return PolygonRegion(points=points)
    try:
        return BoxedRegion(x=raw['x'], y=raw['y'], width=raw['width'], height=raw['height'])
    except KeyError as e:
        raise ValueError(
            f"Region in '{field_name}' needs either 'points' or x/y/width/height (missing {e}): {raw!r}"
        )


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

    inference_imgsz = int(data.get('inference_imgsz', 640))
    if inference_imgsz <= 0 or inference_imgsz % 32 != 0:
        raise ValueError(f"inference_imgsz must be a positive multiple of 32: {inference_imgsz}")

    # The tracker needs a vehicle in this many consecutive frames before it is
    # stationary. One frame would count every passing car, so require two.
    stationary_frames = math.ceil(float(data['stationary_seconds']) * float(data['target_fps']))
    if stationary_frames < 2:
        raise ValueError(
            "stationary_seconds * target_fps must be > 1 so a vehicle is seen in at least "
            f"2 consecutive frames (got stationary_seconds={data['stationary_seconds']}, "
            f"target_fps={data['target_fps']} -> {stationary_frames} frame)"
        )

    scan_regions = [_parse_region(r, 'scan_regions') for r in data.get('scan_regions') or []]
    ignore_regions = [_parse_region(r, 'ignore_regions') for r in data.get('ignore_regions') or []]

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
        inference_imgsz=inference_imgsz,
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
