# Image Tiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add overlapping tile-based inference to `Detector._run_inference` so small, distant vehicles that YOLO misses at full-frame resolution are detected reliably.

**Architecture:** The frame is split into overlapping tiles; YOLO runs on each tile plus the full frame; boxes from all passes are remapped to full-frame coordinates and merged; the existing IOU-based NMS deduplicates overlaps before scan-region filtering runs as normal. Tiling is opt-in via three new config fields; when absent, behavior is identical to today.

**Tech Stack:** Python 3.12, OpenCV (cv2), ultralytics YOLO, PyYAML, pytest

---

## File Map

| File | Change |
|------|--------|
| `src/config.py` | Add `tile_width`, `tile_height`, `tile_overlap` optional fields to `AppConfig`; load them from YAML |
| `src/detector.py` | Add `tile_width/height/overlap` params to `__init__`; refactor `_run_inference` to run tiled + full-frame inference and merge results |
| `src/__main__.py` | Pass new tiling fields from `AppConfig` to `Detector.__init__` |
| `config/app-config.yaml` | Add tiling fields |
| `config/example-config.yaml` | Add tiling fields with comments |
| `tests/test_detector.py` | Add unit tests for tile generation, coordinate remapping, and deduplication; update `make_detector` helper; update `TestRealImageDetection` to pass tiling params |
| `tests/data/test_cases.yaml` | Add `tile_width`, `tile_height`, `tile_overlap` to all `detection_cases` |

---

## Task 1: Add tiling fields to `AppConfig` and `load_app_config`

**Files:**
- Modify: `src/config.py`

- [ ] **Step 1: Write failing test**

In `tests/test_config.py` (create if absent, otherwise append):

```python
import pytest
from unittest.mock import patch, mock_open
import yaml
from src.config import load_app_config

MINIMAL_CONFIG = {
    'camera_name': 'test', 'rtsps_url': 'rtsps://host/stream',
    'scan_regions': [], 'vehicle_classes': ['car'],
    'detection_confidence': 0.3, 'stationary_seconds': 3,
    'iou_threshold': 0.5, 'night_enhancement': True, 'target_fps': 1,
    'model_path': 'yolov8x.pt', 'publish_interval_seconds': 5,
    'mqtt_timeout_seconds': 60, 'mqtt_topic': 'test/topic',
    'output_dir': '/tmp', 'image_save_cooldown_seconds': 30,
}

def load_from_dict(d):
    content = yaml.dump(d)
    with patch('builtins.open', mock_open(read_data=content)):
        with patch('src.config.Path.exists', return_value=True):
            return load_app_config('fake.yaml')

def test_tiling_fields_loaded_when_present():
    cfg = {**MINIMAL_CONFIG, 'tile_width': 640, 'tile_height': 480, 'tile_overlap': 0.25}
    result = load_from_dict(cfg)
    assert result.tile_width == 640
    assert result.tile_height == 480
    assert result.tile_overlap == pytest.approx(0.25)

def test_tiling_fields_default_to_none_when_absent():
    result = load_from_dict(MINIMAL_CONFIG)
    assert result.tile_width is None
    assert result.tile_height is None
    assert result.tile_overlap is None
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /home/ghormann/src/car-counter
pytest tests/test_config.py::test_tiling_fields_loaded_when_present tests/test_config.py::test_tiling_fields_default_to_none_when_absent -v
```

Expected: `AttributeError` — `AppConfig` has no `tile_width`

- [ ] **Step 3: Add fields to `AppConfig` dataclass**

In `src/config.py`, add three optional fields to `AppConfig` after `image_save_cooldown_seconds`:

```python
    tile_width: int | None = None
    tile_height: int | None = None
    tile_overlap: float | None = None
```

- [ ] **Step 4: Load fields in `load_app_config`**

In `src/config.py`, in the `return AppConfig(...)` block, add after `image_save_cooldown_seconds=...`:

```python
        tile_width=int(data['tile_width']) if 'tile_width' in data else None,
        tile_height=int(data['tile_height']) if 'tile_height' in data else None,
        tile_overlap=float(data['tile_overlap']) if 'tile_overlap' in data else None,
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/test_config.py::test_tiling_fields_loaded_when_present tests/test_config.py::test_tiling_fields_default_to_none_when_absent -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add tile_width, tile_height, tile_overlap to AppConfig"
```

---

## Task 2: Add tiling params to `Detector.__init__` and wire from `__main__.py`

**Files:**
- Modify: `src/detector.py`
- Modify: `src/__main__.py`

- [ ] **Step 1: Write failing test**

In `tests/test_detector.py`, add to the bottom:

```python
class TestTilingInit:
    def test_detector_accepts_tiling_params(self):
        with patch('src.detector.YOLO'):
            d = Detector(
                model_path='yolov8x.pt',
                vehicle_classes=['car'],
                detection_confidence=0.3,
                iou_threshold=0.5,
                stationary_seconds=3,
                target_fps=1,
                night_enhancement=True,
                scan_regions=[],
                tile_width=640,
                tile_height=640,
                tile_overlap=0.2,
            )
        assert d._tile_width == 640
        assert d._tile_height == 640
        assert d._tile_overlap == pytest.approx(0.2)

    def test_detector_tiling_defaults_to_none(self):
        with patch('src.detector.YOLO'):
            d = Detector(
                model_path='yolov8x.pt',
                vehicle_classes=['car'],
                detection_confidence=0.3,
                iou_threshold=0.5,
                stationary_seconds=3,
                target_fps=1,
                night_enhancement=True,
                scan_regions=[],
            )
        assert d._tile_width is None
        assert d._tile_height is None
        assert d._tile_overlap is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_detector.py::TestTilingInit -v
```

Expected: `TypeError` — unexpected keyword arguments

- [ ] **Step 3: Add params to `Detector.__init__`**

In `src/detector.py`, update the `__init__` signature and body:

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
        tile_width: int | None = None,
        tile_height: int | None = None,
        tile_overlap: float | None = None,
    ):
        self._model = YOLO(model_path)
        self._vehicle_classes = vehicle_classes
        self._detection_confidence = detection_confidence
        self._iou_threshold = iou_threshold
        self._required_frames = math.ceil(stationary_seconds * target_fps)
        self._night_enhancement = night_enhancement
        self._scan_regions = scan_regions
        self._tracked: list[TrackedVehicle] = []
        self._tile_width = tile_width
        self._tile_height = tile_height
        self._tile_overlap = tile_overlap
```

- [ ] **Step 4: Wire tiling params in `src/__main__.py`**

Find the `Detector(...)` call in `src/__main__.py` and add the three new kwargs:

```python
        night_enhancement=app_config.night_enhancement,
        scan_regions=app_config.scan_regions,
        tile_width=app_config.tile_width,
        tile_height=app_config.tile_height,
        tile_overlap=app_config.tile_overlap,
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_detector.py::TestTilingInit -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/detector.py src/__main__.py tests/test_detector.py
git commit -m "feat: add tiling params to Detector constructor"
```

---

## Task 3: Implement tile generation helper `_generate_tiles`

**Files:**
- Modify: `src/detector.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_detector.py`, add to the bottom:

```python
class TestGenerateTiles:
    def test_no_tiling_when_params_are_none(self):
        d = make_detector()
        # tile params default to None in make_detector
        tiles = d._generate_tiles(height=1080, width=1920)
        assert tiles == []

    def test_single_tile_when_image_smaller_than_tile(self):
        d = make_detector(tile_width=640, tile_height=640, tile_overlap=0.2)
        tiles = d._generate_tiles(height=480, width=640)
        assert len(tiles) == 1
        assert tiles[0] == (0, 0, 640, 480)  # (x, y, w, h) clamped to image

    def test_tiles_cover_full_image(self):
        d = make_detector(tile_width=640, tile_height=640, tile_overlap=0.0)
        tiles = d._generate_tiles(height=1280, width=1280)
        # With no overlap and 640-wide tiles: expect 2x2 = 4 tiles
        assert len(tiles) == 4

    def test_tiles_overlap_by_specified_fraction(self):
        d = make_detector(tile_width=640, tile_height=640, tile_overlap=0.5)
        tiles = d._generate_tiles(height=640, width=1280)
        # stride = 640 * (1 - 0.5) = 320; fits: x=0, x=320, x=640 → 3 tiles
        xs = [t[0] for t in tiles]
        assert 0 in xs
        assert 320 in xs

    def test_tile_bounds_never_exceed_image(self):
        d = make_detector(tile_width=640, tile_height=640, tile_overlap=0.2)
        tiles = d._generate_tiles(height=1080, width=1920)
        for x, y, w, h in tiles:
            assert x + w <= 1920
            assert y + h <= 1080
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_detector.py::TestGenerateTiles -v
```

Expected: `AttributeError: 'Detector' object has no attribute '_generate_tiles'`

- [ ] **Step 3: Implement `_generate_tiles`**

Add this method to `Detector` in `src/detector.py` (after `__init__`, before `process_frame`):

```python
    def _generate_tiles(self, height: int, width: int) -> list[tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) tile coords covering the full frame."""
        if self._tile_width is None or self._tile_height is None or self._tile_overlap is None:
            return []
        stride_x = max(1, int(self._tile_width * (1 - self._tile_overlap)))
        stride_y = max(1, int(self._tile_height * (1 - self._tile_overlap)))
        tiles = []
        y = 0
        while y < height:
            x = 0
            while x < width:
                w = min(self._tile_width, width - x)
                h = min(self._tile_height, height - y)
                tiles.append((x, y, w, h))
                if x + self._tile_width >= width:
                    break
                x += stride_x
            if y + self._tile_height >= height:
                break
            y += stride_y
        return tiles
```

- [ ] **Step 4: Update `make_detector` helper to accept tiling params**

In `tests/test_detector.py`, update the `make_detector` defaults dict to include:

```python
    defaults = dict(
        model_path='yolov8n.pt',
        vehicle_classes=['car', 'truck', 'bus'],
        detection_confidence=0.4,
        iou_threshold=0.5,
        stationary_seconds=3,
        target_fps=1,
        night_enhancement=True,
        scan_regions=[],
        tile_width=None,
        tile_height=None,
        tile_overlap=None,
    )
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_detector.py::TestGenerateTiles -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/detector.py tests/test_detector.py
git commit -m "feat: implement Detector._generate_tiles for overlapping tile coordinates"
```

---

## Task 4: Refactor `_run_inference` to use tiling

**Files:**
- Modify: `src/detector.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_detector.py`, add to the bottom:

```python
class TestTiledInference:
    def _make_mock_result(self, boxes_data):
        """boxes_data: list of (x1,y1,x2,y2, conf, cls_id)"""
        mock_result = MagicMock()
        mock_boxes = []
        for x1, y1, x2, y2, conf, cls_id in boxes_data:
            b = MagicMock()
            b.xyxy = [np.array([x1, y1, x2, y2], dtype=np.float32)]
            b.conf = [np.float32(conf)]
            b.cls = [np.float32(cls_id)]
            mock_boxes.append(b)
        mock_result.boxes = mock_boxes
        return mock_result

    def test_tiling_disabled_runs_single_inference(self):
        d = make_detector(vehicle_classes=['car'], detection_confidence=0.3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self._make_mock_result([(10, 10, 100, 100, 0.9, 2)])
        d._model.return_value = [result]
        d._model.names = {2: 'car'}

        detections = d._run_inference(frame)
        assert d._model.call_count == 1
        assert len(detections) == 1

    def test_tiling_enabled_runs_tile_plus_fullframe(self):
        # 1280-wide frame, 640-wide tiles, no overlap → 2 tiles + 1 full = 3 calls
        d = make_detector(
            vehicle_classes=['car'], detection_confidence=0.3,
            tile_width=640, tile_height=480, tile_overlap=0.0,
        )
        frame = np.zeros((480, 1280, 3), dtype=np.uint8)
        empty_result = self._make_mock_result([])
        d._model.return_value = [empty_result]
        d._model.names = {2: 'car'}

        d._run_inference(frame)
        assert d._model.call_count == 3  # 2 tiles + 1 full frame

    def test_tile_detections_remapped_to_full_frame_coords(self):
        # Tile starts at x=640, y=0; detection at tile-local (10,10,100,100)
        # Should appear at full-frame (650,10,740,100)
        d = make_detector(
            vehicle_classes=['car'], detection_confidence=0.3,
            tile_width=640, tile_height=480, tile_overlap=0.0,
        )
        frame = np.zeros((480, 1280, 3), dtype=np.uint8)
        empty_result = self._make_mock_result([])
        tile_result = self._make_mock_result([(10, 10, 100, 100, 0.9, 2)])
        d._model.names = {2: 'car'}

        call_count = 0
        def side_effect(crop, verbose=False):
            nonlocal call_count
            call_count += 1
            # Second call is the right tile (x=640)
            if call_count == 2:
                return [tile_result]
            return [empty_result]

        d._model.side_effect = side_effect

        detections = d._run_inference(frame)
        car_detections = [det for det in detections if det.class_name == 'car']
        assert len(car_detections) == 1
        x1, y1, x2, y2 = car_detections[0].box
        assert x1 == pytest.approx(650)
        assert x2 == pytest.approx(740)

    def test_overlapping_tile_detections_deduplicated(self):
        # Same vehicle detected in two overlapping tiles → NMS keeps only one
        d = make_detector(
            vehicle_classes=['car'], detection_confidence=0.3,
            iou_threshold=0.5,
            tile_width=640, tile_height=480, tile_overlap=0.5,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Both tiles produce nearly the same box; should collapse to 1
        dup_result = self._make_mock_result([(10, 10, 100, 100, 0.9, 2)])
        d._model.return_value = [dup_result]
        d._model.names = {2: 'car'}

        detections = d._run_inference(frame)
        assert len(detections) == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_detector.py::TestTiledInference -v
```

Expected: failures — tiling logic not yet in `_run_inference`

- [ ] **Step 3: Refactor `_run_inference`**

Replace the existing `_run_inference` method in `src/detector.py` with:

```python
    def _run_inference(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        candidates = self._infer_on_crop(frame, offset_x=0, offset_y=0)

        for x, y, w, h in self._generate_tiles(height=height, width=width):
            tile = frame[y:y + h, x:x + w]
            tile_detections = self._infer_on_crop(tile, offset_x=x, offset_y=y)
            candidates.extend(tile_detections)

        candidates.sort(key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []
        for candidate in candidates:
            if not any(self._compute_iou(candidate.box, k.box) >= self._iou_threshold for k in kept):
                kept.append(candidate)

        return [d for d in kept if self._is_in_scan_regions(d.box)]

    def _infer_on_crop(self, crop: np.ndarray, offset_x: int, offset_y: int) -> list[Detection]:
        results = self._model(crop, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self._model.names[class_id]
                if class_name in self._vehicle_classes and conf >= self._detection_confidence:
                    detections.append(Detection(
                        box=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                        class_name=class_name,
                        confidence=conf,
                    ))
        return detections
```

Note: `_is_in_scan_regions` filtering has moved from `_infer_on_crop` into `_run_inference` after NMS, so it applies once across all merged detections.

- [ ] **Step 4: Run all detector tests**

```bash
pytest tests/test_detector.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/detector.py tests/test_detector.py
git commit -m "feat: refactor _run_inference to run tiled + full-frame inference with merged NMS"
```

---

## Task 5: Update config files and test cases

**Files:**
- Modify: `config/app-config.yaml`
- Modify: `config/example-config.yaml`
- Modify: `tests/data/test_cases.yaml`

- [ ] **Step 1: Add tiling fields to `config/app-config.yaml`**

Add after the `model_path` line:

```yaml
# Image tiling (improves detection of small/distant vehicles)
tile_width: 640
tile_height: 640
tile_overlap: 0.2
```

- [ ] **Step 2: Add tiling fields to `config/example-config.yaml`**

Add after the `model_path` line:

```yaml
# Image tiling (improves detection of small/distant vehicles)
tile_width: 640
tile_height: 640
tile_overlap: 0.2
```

- [ ] **Step 3: Add tiling fields to all `detection_cases` in `tests/data/test_cases.yaml`**

For every entry under `detection_cases`, add:

```yaml
    tile_width: 640
    tile_height: 640
    tile_overlap: 0.2
```

- [ ] **Step 4: Update `TestRealImageDetection` to pass tiling params to `Detector`**

In `tests/test_detector.py`, update the `Detector(...)` call inside `test_detects_expected_vehicle_count`:

```python
        detector = Detector(
            model_path='yolov8x.pt',
            vehicle_classes=case.get('vehicle_classes', ['car', 'truck', 'bus']),
            detection_confidence=case.get('detection_confidence', 0.4),
            iou_threshold=0.5,
            stationary_seconds=1,
            target_fps=1,
            night_enhancement=case.get('night_enhancement', True),
            scan_regions=scan_regions,
            tile_width=case.get('tile_width'),
            tile_height=case.get('tile_height'),
            tile_overlap=case.get('tile_overlap'),
        )
```

- [ ] **Step 5: Run the full test suite**

```bash
pytest tests/ -v --ignore=tests/test_detector.py -k "not TestRealImageDetection"
```

Expected: all PASS (real image tests skipped — expected_counts not yet calibrated)

- [ ] **Step 6: Commit**

```bash
git add config/app-config.yaml config/example-config.yaml tests/data/test_cases.yaml tests/test_detector.py
git commit -m "feat: add tiling config to app-config, example-config, and test cases"
```

---

## Task 6: Re-run annotated images and calibrate `expected_count`

**Files:**
- Modify: `annotate_images.py`
- Modify: `tests/data/test_cases.yaml`

- [ ] **Step 1: Update `annotate_images.py` to use tiling**

Replace the `Detector(...)` instantiation in `annotate_images.py` with:

```python
detector = Detector(
    model_path="yolov8x.pt",
    vehicle_classes=["car", "truck", "bus"],
    detection_confidence=0.15,
    iou_threshold=0.5,
    stationary_seconds=3,
    target_fps=1,
    night_enhancement=True,
    scan_regions=[],
    tile_width=640,
    tile_height=640,
    tile_overlap=0.2,
)
```

- [ ] **Step 2: Run annotation script**

```bash
cd /home/ghormann/src/car-counter
python annotate_images.py
```

Review each `annotated_*.jpg/png` in `tests/data/images/` and note the correct vehicle count per image.

- [ ] **Step 3: Update `expected_count` in `tests/data/test_cases.yaml`**

For each detection case, set `expected_count` to the actual correct count observed in the annotated image.

- [ ] **Step 4: Commit**

```bash
git add annotate_images.py tests/data/test_cases.yaml tests/data/images/
git commit -m "chore: update annotated images and calibrate expected_count for tiling"
```
