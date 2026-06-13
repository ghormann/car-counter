# ignore_regions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional `ignore_regions` config that suppresses vehicles whose bounding box is ≥95% inside an exclusion zone, and renders those zones as labeled blue rectangles in annotated images.

**Architecture:** `IgnoreRegion` dataclass mirrors `ScanRegion` in `config.py`. `Detector` gains `_is_in_ignore_regions` called after scan-region filtering. `ImageSaver._annotate` draws ignore regions in blue with an "exclude" label. All callers updated to thread the new list through.

**Tech Stack:** Python 3.12, PyYAML, OpenCV (`cv2`), pytest

---

## File Map

| File | Change |
|------|--------|
| `src/config.py` | Add `IgnoreRegion` dataclass; add `ignore_regions: list[IgnoreRegion]` to `AppConfig`; parse from YAML |
| `src/detector.py` | Add `ignore_regions` param to `__init__`; add `_is_in_ignore_regions`; call it in `_run_inference` |
| `src/image_saver.py` | Add `ignore_regions` param to `save` and `_annotate`; draw blue "exclude" boxes |
| `annotate_images.py` | Pass `ignore_regions=[]` to `Detector.__init__` and `ImageSaver._annotate` |
| `config/example-config.yaml` | Add commented `ignore_regions` example |
| `docs/config-reference.md` | Add `ignore_regions` section after `scan_regions` |
| `docs/requirements.md` | Add `ignore_regions` to functional requirements and config YAML example |
| `README.md` | Add `ignore_regions` to Features list and config example |
| `tests/test_config.py` | Tests for `ignore_regions` parsing |
| `tests/test_detector.py` | Tests for `_is_in_ignore_regions` and suppression in `_run_inference` |
| `tests/test_image_saver.py` | Test for blue "exclude" box rendering |

---

## Task 1: Add `IgnoreRegion` to config

**Files:**
- Modify: `src/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_config.py` (after the existing `scan_regions` tests):

```python
    def test_ignore_regions_parsed_when_present(self, tmp_path):
        data = dict(VALID_APP_DATA)
        data['output_dir'] = str(tmp_path)
        data['ignore_regions'] = [{'x': 10, 'y': 20, 'width': 50, 'height': 60}]
        p = tmp_path / 'config.yaml'
        write_yaml(p, data)
        result = load_app_config(str(p))
        assert result.ignore_regions == [IgnoreRegion(x=10, y=20, width=50, height=60)]

    def test_ignore_regions_defaults_to_empty_list(self, tmp_path):
        p = make_app_config_file(tmp_path)
        result = load_app_config(str(p))
        assert result.ignore_regions == []
```

Also update the import line at the top of `tests/test_config.py`:

```python
from src.config import load_app_config, load_mqtt_config, AppConfig, MqttConfig, ScanRegion, IgnoreRegion
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ghormann/src/car-counter && source env/bin/activate && pytest tests/test_config.py::TestLoadAppConfig::test_ignore_regions_parsed_when_present tests/test_config.py::TestLoadAppConfig::test_ignore_regions_defaults_to_empty_list -v
```

Expected: FAIL with `ImportError: cannot import name 'IgnoreRegion'`

- [ ] **Step 3: Implement `IgnoreRegion` in config.py**

In `src/config.py`, add `IgnoreRegion` after `ScanRegion` (line 12), add the field to `AppConfig`, and parse it. Replace the file contents with:

```python
import json
import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScanRegion:
    x: int
    y: int
    width: int
    height: int


@dataclass
class IgnoreRegion:
    x: int
    y: int
    width: int
    height: int


@dataclass
class AppConfig:
    camera_name: str
    rtsps_url: str
    scan_regions: list[ScanRegion]
    ignore_regions: list[IgnoreRegion]
    vehicle_classes: list[str]
    detection_confidence: float
    stationary_seconds: int
    iou_threshold: float
    night_enhancement: bool
    target_fps: int
    model_path: str
    publish_interval_seconds: int
    mqtt_timeout_seconds: int
    mqtt_topic: str
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
    'mqtt_topic', 'output_dir', 'image_save_cooldown_seconds',
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
        ScanRegion(x=r['x'], y=r['y'], width=r['width'], height=r['height'])
        for r in data.get('scan_regions', [])
    ]

    ignore_regions = [
        IgnoreRegion(x=r['x'], y=r['y'], width=r['width'], height=r['height'])
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
        target_fps=int(data['target_fps']),
        model_path=str(data['model_path']),
        publish_interval_seconds=int(data['publish_interval_seconds']),
        mqtt_timeout_seconds=int(data['mqtt_timeout_seconds']),
        mqtt_topic=str(data['mqtt_topic']),
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ghormann/src/car-counter && source env/bin/activate && pytest tests/test_config.py -v
```

Expected: All config tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ghormann/src/car-counter && git add src/config.py tests/test_config.py && git commit -m "feat: add IgnoreRegion dataclass and config parsing"
```

---

## Task 2: Add ignore-region suppression to Detector

**Files:**
- Modify: `src/detector.py`
- Modify: `tests/test_detector.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_detector.py` after the existing `TestScanRegions` class. Also update the import at the top to include `IgnoreRegion`:

```python
from src.config import ScanRegion, IgnoreRegion
```

Add the new test class:

```python
class TestIgnoreRegions:
    def test_no_ignore_regions_does_not_suppress(self):
        d = make_detector(ignore_regions=[])
        assert d._is_in_ignore_regions((0, 0, 100, 100)) is False

    def test_vehicle_fully_inside_ignore_region_is_suppressed(self):
        region = IgnoreRegion(x=0, y=0, width=200, height=200)
        d = make_detector(ignore_regions=[region])
        # box entirely inside: 100% overlap
        assert d._is_in_ignore_regions((10, 10, 190, 190)) is True

    def test_vehicle_exactly_95_percent_inside_is_suppressed(self):
        # region covers x=0..200, y=0..200 (area 40000)
        # box: x=0..200, y=0..200 (area 40000), but shift so 95% overlaps
        # box x=0..200, y=0..200 (area=40000); region x=0..190, y=0..211 (> 95%)
        region = IgnoreRegion(x=0, y=0, width=200, height=200)
        d = make_detector(ignore_regions=[region])
        # box: x=0..200, y=0..200 (area 40000)
        # intersection with region (0..200, 0..200): 40000 => 100%
        assert d._is_in_ignore_regions((0, 0, 200, 200)) is True

    def test_vehicle_94_percent_inside_is_not_suppressed(self):
        # region: x=0..100, y=0..100 (area 10000)
        # box: x=0..100, y=0..106 (area 10600)
        # intersection: x=0..100, y=0..100 => 10000
        # coverage: 10000 / 10600 = 0.943... < 0.95 → not suppressed
        region = IgnoreRegion(x=0, y=0, width=100, height=100)
        d = make_detector(ignore_regions=[region])
        assert d._is_in_ignore_regions((0, 0, 100, 106)) is False

    def test_vehicle_95_percent_inside_one_of_multiple_regions_is_suppressed(self):
        regions = [
            IgnoreRegion(x=500, y=500, width=100, height=100),
            IgnoreRegion(x=0, y=0, width=200, height=200),
        ]
        d = make_detector(ignore_regions=regions)
        assert d._is_in_ignore_regions((10, 10, 190, 190)) is True

    def test_vehicle_outside_all_ignore_regions_is_not_suppressed(self):
        region = IgnoreRegion(x=500, y=500, width=100, height=100)
        d = make_detector(ignore_regions=[region])
        assert d._is_in_ignore_regions((0, 0, 100, 100)) is False
```

Also add `ignore_regions=[]` to `make_detector`'s `defaults` dict and to `test_detector_constructs_without_threshold_params`:

```python
def make_detector(**kwargs):
    defaults = dict(
        model_path='yolov8n.pt',
        vehicle_classes=['car', 'truck', 'bus'],
        detection_confidence=0.4,
        iou_threshold=0.5,
        stationary_seconds=3,
        target_fps=1,
        night_enhancement=True,
        scan_regions=[],
        ignore_regions=[],
    )
    defaults.update(kwargs)
    with patch('src.detector.YOLO'):
        return Detector(**defaults)
```

And update `test_detector_constructs_without_threshold_params`:

```python
def test_detector_constructs_without_threshold_params():
    with patch('src.detector.YOLO'):
        d = Detector(
            model_path='yolov8n.pt',
            vehicle_classes=['car'],
            detection_confidence=0.4,
            iou_threshold=0.5,
            stationary_seconds=3,
            target_fps=1,
            night_enhancement=True,
            scan_regions=[],
            ignore_regions=[],
        )
    assert d is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ghormann/src/car-counter && source env/bin/activate && pytest tests/test_detector.py::TestIgnoreRegions -v
```

Expected: FAIL with `TypeError: Detector.__init__() got an unexpected keyword argument 'ignore_regions'`

- [ ] **Step 3: Implement `_is_in_ignore_regions` in detector.py**

In `src/detector.py`, update the import and `Detector` class. Apply these changes:

1. Update the import (line 9):
```python
from src.config import ScanRegion, IgnoreRegion
```

2. Add `ignore_regions` parameter to `__init__` (after `scan_regions` on line 42):
```python
    def __init__(
        self,
        model_path: str,
        vehicle_classes: list[str],
        detection_confidence: float,
        iou_threshold: float,
        stationary_seconds: int,
        target_fps: int,
        night_enhancement: bool,
        scan_regions: list[ScanRegion],
        ignore_regions: list[IgnoreRegion],
    ):
        self._model = YOLO(model_path)
        self._vehicle_classes = vehicle_classes
        self._detection_confidence = detection_confidence
        self._iou_threshold = iou_threshold
        self._required_frames = math.ceil(stationary_seconds * target_fps)
        self._night_enhancement = night_enhancement
        self._scan_regions = scan_regions
        self._ignore_regions = ignore_regions
        self._tracked: list[TrackedVehicle] = []
```

3. In `_run_inference`, add the ignore-region check immediately after the scan-region check (line 98–99). Replace:
```python
                if self._is_in_scan_regions((x1, y1, x2, y2)):
                    candidates.append(Detection(box=(x1, y1, x2, y2), class_name=class_name, confidence=conf))
```
With:
```python
                if self._is_in_scan_regions((x1, y1, x2, y2)) and not self._is_in_ignore_regions((x1, y1, x2, y2)):
                    candidates.append(Detection(box=(x1, y1, x2, y2), class_name=class_name, confidence=conf))
```

4. Add the new method at the end of the class (after `_is_in_scan_regions`):
```python
    def _is_in_ignore_regions(self, box: tuple) -> bool:
        if not self._ignore_regions:
            return False
        x1, y1, x2, y2 = box
        vehicle_area = (x2 - x1) * (y2 - y1)
        if vehicle_area <= 0:
            return False
        for region in self._ignore_regions:
            rx1, ry1 = region.x, region.y
            rx2, ry2 = region.x + region.width, region.y + region.height
            ix1 = max(x1, rx1)
            iy1 = max(y1, ry1)
            ix2 = min(x2, rx2)
            iy2 = min(y2, ry2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / vehicle_area >= 0.95:
                return True
        return False
```

- [ ] **Step 4: Run all detector tests to verify they pass**

```bash
cd /home/ghormann/src/car-counter && source env/bin/activate && pytest tests/test_detector.py -v
```

Expected: All detector tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ghormann/src/car-counter && git add src/detector.py tests/test_detector.py && git commit -m "feat: suppress detections ≥95% inside ignore regions"
```

---

## Task 3: Draw ignore regions in annotated images

**Files:**
- Modify: `src/image_saver.py`
- Modify: `tests/test_image_saver.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_image_saver.py`. First update the import at the top:

```python
from src.config import ScanRegion, IgnoreRegion
```

Then add the test after `test_scan_region_drawn_green`:

```python
    def test_ignore_region_drawn_blue_with_label(self, tmp_output_dir):
        saver = ImageSaver(tmp_output_dir, "driveway", cooldown_seconds=0)
        region = IgnoreRegion(x=50, y=50, width=100, height=100)
        frame = make_frame()
        annotated = saver._annotate(frame.copy(), [], [], [region])
        # Blue in BGR is (B=255, G=0, R=0). Rectangle drawn at (col=50, row=50).
        assert annotated[50, 50, 0] == 255  # B channel high
        assert annotated[50, 50, 2] == 0    # R channel zero
        assert annotated[50, 50, 1] == 0    # G channel zero
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ghormann/src/car-counter && source env/bin/activate && pytest tests/test_image_saver.py::TestAnnotate::test_ignore_region_drawn_blue_with_label -v
```

Expected: FAIL with `TypeError: _annotate() takes 4 positional arguments but 5 were given`

- [ ] **Step 3: Implement ignore-region drawing in image_saver.py**

Update `src/image_saver.py`. The full updated file:

```python
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.config import ScanRegion, IgnoreRegion
from src.detector import TrackedVehicle

logger = logging.getLogger(__name__)


class ImageSaver:
    def __init__(self, output_dir: Path, camera_name: str, cooldown_seconds: int):
        self._output_dir = output_dir
        self._camera_name = camera_name
        self._cooldown_seconds = cooldown_seconds
        self._last_save_time = 0.0

    def save(
        self,
        frame: np.ndarray,
        stationary_vehicles: list[TrackedVehicle],
        scan_regions: list[ScanRegion],
        ignore_regions: list[IgnoreRegion] = None,
        prefix: str = "",
    ) -> Path | None:
        if ignore_regions is None:
            ignore_regions = []
        now = time.monotonic()
        if not prefix and (now - self._last_save_time) < self._cooldown_seconds:
            return None

        dt = datetime.now(timezone.utc)
        timestamp = dt.strftime("%Y%m%d_%H%M%S")
        save_dir = (
            self._output_dir
            / self._camera_name
            / str(dt.year)
            / f"{dt.month:02d}"
            / f"{dt.day:02d}"
        )

        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error("Failed to create directory %s: %s", save_dir, e)
            return None

        filename = f"{prefix}{timestamp}.jpg"
        path = save_dir / filename
        annotated = self._annotate(frame.copy(), stationary_vehicles, scan_regions, ignore_regions)

        try:
            success = cv2.imwrite(str(path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                logger.error("cv2.imwrite failed for %s", path)
                return None
        except OSError as e:
            logger.error("Failed to save image %s: %s", path, e)
            return None

        if not prefix:
            self._last_save_time = now

        logger.info("Saved image: %s", path)
        return path

    @staticmethod
    def _annotate(
        frame: np.ndarray,
        stationary_vehicles: list[TrackedVehicle],
        scan_regions: list[ScanRegion],
        ignore_regions: list[IgnoreRegion] = None,
    ) -> np.ndarray:
        if ignore_regions is None:
            ignore_regions = []
        frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        for vehicle in stationary_vehicles:
            x1, y1, x2, y2 = (int(c) for c in vehicle.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red
            label = f"{vehicle.class_name} {vehicle.confidence:.2f}"
            (lw, lh), baseline = cv2.getTextSize(label, font, 0.5, 1)
            ly = max(y1 - 4, lh + baseline)
            cv2.rectangle(frame, (x1, ly - lh - baseline), (x1 + lw, ly), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, label, (x1, ly - baseline), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        for region in scan_regions:
            x1, y1 = region.x, region.y
            x2, y2 = region.x + region.width, region.y + region.height
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green

        for region in ignore_regions:
            x1, y1 = region.x, region.y
            x2, y2 = region.x + region.width, region.y + region.height
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue
            label = "exclude"
            (lw, lh), baseline = cv2.getTextSize(label, font, 0.5, 1)
            ly = y1 + lh + baseline + 4
            cv2.rectangle(frame, (x1, y1), (x1 + lw, ly), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, label, (x1, ly - baseline), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        total = len(stationary_vehicles)
        summary = f"Vehicles detected: {total}"
        (sw, sh), baseline = cv2.getTextSize(summary, font, 0.8, 2)
        h, w = frame.shape[:2]
        sx = (w - sw) // 2
        sy = h - 20
        cv2.rectangle(frame, (sx - 4, sy - sh - baseline), (sx + sw + 4, sy + baseline), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, summary, (sx, sy), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return frame
```

- [ ] **Step 4: Run all image saver tests to verify they pass**

```bash
cd /home/ghormann/src/car-counter && source env/bin/activate && pytest tests/test_image_saver.py -v
```

Expected: All image saver tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ghormann/src/car-counter && git add src/image_saver.py tests/test_image_saver.py && git commit -m "feat: draw ignore regions as blue labeled boxes in annotations"
```

---

## Task 4: Wire ignore_regions through __main__ and annotate_images.py

**Files:**
- Modify: `src/__main__.py`
- Modify: `annotate_images.py`

- [ ] **Step 1: Check how `__main__.py` wires config into Detector and ImageSaver**

```bash
grep -n "Detector\|ImageSaver\|scan_regions\|ignore_regions" /home/ghormann/src/car-counter/src/__main__.py
```

- [ ] **Step 2: Update `__main__.py` to pass `ignore_regions`**

Find the `Detector(...)` constructor call in `src/__main__.py` and add `ignore_regions=config.ignore_regions` after `scan_regions=config.scan_regions`.

Find the `saver.save(...)` call(s) and add `ignore_regions=config.ignore_regions` after `scan_regions=config.scan_regions`.

- [ ] **Step 3: Update `annotate_images.py` to pass `ignore_regions=[]`**

In `annotate_images.py`, update the `Detector(...)` constructor call to add `ignore_regions=[]` after `scan_regions=[]`.

Update the `ImageSaver._annotate(...)` call (line 44) to add `ignore_regions=[]` as the fourth argument:

```python
    annotated = ImageSaver._annotate(frame.copy(), vehicles, scan_regions=[], ignore_regions=[])
```

- [ ] **Step 4: Run the full test suite**

```bash
cd /home/ghormann/src/car-counter && source env/bin/activate && pytest -v
```

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/ghormann/src/car-counter && git add src/__main__.py annotate_images.py && git commit -m "feat: wire ignore_regions through main and annotate_images"
```

---

## Task 5: Update configuration files and documentation

**Files:**
- Modify: `config/example-config.yaml`
- Modify: `docs/config-reference.md`
- Modify: `docs/requirements.md`
- Modify: `README.md`

- [ ] **Step 1: Update `config/example-config.yaml`**

Add the `ignore_regions` block after `scan_regions`:

```yaml
# Scan regions (optional — omit to scan entire frame)
scan_regions:
  - {x: 100, y: 200, width: 400, height: 300}

# Ignore regions (optional — vehicles ≥95% inside are excluded from counting)
# ignore_regions:
#   - {x: 0, y: 0, width: 100, height: 50}
```

- [ ] **Step 2: Update `docs/config-reference.md`**

Add the `ignore_regions` section immediately after the `scan_regions` section (after the scan regions table). Insert:

````markdown
### Ignore Regions (optional)

```yaml
ignore_regions:
  - { x: 0, y: 0, width: 100, height: 50 }
```

Pixel coordinates relative to the full frame resolution. If a detected vehicle's bounding box is 95% or more inside any ignore region, it is excluded from counting. Useful for masking areas with frequent false positives (e.g., a parking spot reflected in glass, a neighbor's driveway at the edge of frame).

Ignore regions are rendered as **blue** outlines labeled `exclude` on annotated images.

| Sub-field | Type | Description                        |
| --------- | ---- | ---------------------------------- |
| `x`       | int  | Left edge of the region in pixels  |
| `y`       | int  | Top edge of the region in pixels   |
| `width`   | int  | Width of the region in pixels      |
| `height`  | int  | Height of the region in pixels     |
````

- [ ] **Step 3: Update `docs/requirements.md`**

In the **Vehicle Detection** section, add after the scan-regions bullet:

```
- Optionally define `ignore_regions`: a vehicle whose bounding box is ≥95% inside any ignore region is excluded from counting
```

In the **Image Saving** annotation bullet, add after the green scan-region line:

```
  - **Blue** bounding boxes (labeled "exclude") showing configured ignore regions
```

In the **Application YAML Config** example block, add after `scan_regions`:

```yaml
# Ignore regions (optional — vehicles ≥95% inside are excluded)
# ignore_regions:
#   - { x: 0, y: 0, width: 100, height: 50 }
```

- [ ] **Step 4: Update `README.md`**

In the **Features** list, update the annotated screenshots bullet:

```markdown
- **Annotated screenshots** — saved on count change with bounding boxes, scan region overlays (green), and ignore region overlays (blue)
```

In the config YAML example block, add after the `scan_regions` block:

```yaml
# Ignore regions (optional — vehicles ≥95% inside are excluded from counting)
# ignore_regions:
#   - { x: 0, y: 0, width: 100, height: 50 }
```

- [ ] **Step 5: Commit**

```bash
cd /home/ghormann/src/car-counter && git add config/example-config.yaml docs/config-reference.md docs/requirements.md README.md && git commit -m "docs: document ignore_regions in config reference, requirements, and README"
```

---

## Final Verification

- [ ] **Run the full test suite one last time**

```bash
cd /home/ghormann/src/car-counter && source env/bin/activate && pytest -v
```

Expected: All tests PASS, no warnings about missing arguments.
