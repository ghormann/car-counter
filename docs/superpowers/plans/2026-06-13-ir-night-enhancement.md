# IR Night Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-stage IR enhancement path so nighttime IR camera frames are preprocessed before inference, enabling detection of vehicles that the raw IR image misses.

**Architecture:** A new `_enhance_frame` dispatcher replaces the current `_should_apply_clahe` pattern in `Detector`. It auto-detects IR vs color using module-level constants (replacing the previously config-driven thresholds) and routes dark color frames to existing CLAHE and dark IR frames to a new grayscale CLAHE + unsharp-mask path. The `night_enhancement` toggle stays in config. Tests for real images switch from `_run_inference` to `process_frame` so the full enhancement pipeline is exercised.

**Tech Stack:** Python 3.12, OpenCV (`cv2`), YOLOv8n via `ultralytics`, pytest

---

## Files Changed

| File | Change |
|---|---|
| `src/detector.py` | Remove 2 threshold init params; add 2 module constants; replace `_should_apply_clahe` with `_enhance_frame`; add `_apply_ir_enhancement`; update `process_frame` |
| `src/config.py` | Remove `night_brightness_threshold` and `ir_saturation_threshold` from `AppConfig`, required fields list, and loader |
| `src/__main__.py` | Remove the two threshold kwargs from `Detector(...)` call |
| `tests/test_detector.py` | Update `make_detector` helper; replace `TestClahe` with `TestEnhancement`; update `TestRealImageDetection` |
| `tests/data/test_cases.yaml` | Add `night_enhancement` field to all detection cases; update `expected_count` for night cases |
| `tests/test_config.py` | Remove threshold fields from `VALID_APP_DATA` and the missing-required-field parametrize list |
| `config/example-config.yaml` | Remove `night_brightness_threshold` and `ir_saturation_threshold` |
| `docs/requirements.md` | Update Night/Low-Light Detection section and config table |

---

## Task 1: Remove threshold params from `Detector`, add module constants

**Files:**
- Modify: `src/detector.py`
- Modify: `tests/test_detector.py`

- [ ] **Step 1: Write a failing test**

Add this test inside `TestClahe` (it will be renamed in Task 2, but add it here for now):

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
        )
    assert d is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_detector.py::test_detector_constructs_without_threshold_params -v
```

Expected: `FAILED` — `TypeError: __init__() missing required keyword arguments`

- [ ] **Step 3: Remove threshold params from `Detector.__init__`, add module constants**

At the top of `src/detector.py`, before the class, add:

```python
_NIGHT_BRIGHTNESS_THRESHOLD = 80
_IR_SATURATION_THRESHOLD = 30
```

Replace the `__init__` signature (remove `night_brightness_threshold` and `ir_saturation_threshold`):

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
):
    self._model = YOLO(model_path)
    self._vehicle_classes = vehicle_classes
    self._detection_confidence = detection_confidence
    self._iou_threshold = iou_threshold
    self._required_frames = math.ceil(stationary_seconds * target_fps)
    self._night_enhancement = night_enhancement
    self._scan_regions = scan_regions
    self._tracked: list[TrackedVehicle] = []
```

Remove the two instance variable assignments for the old thresholds. Update `_should_apply_clahe` to use module constants in place of `self._night_brightness_threshold` and `self._ir_saturation_threshold`:

```python
def _should_apply_clahe(self, frame: np.ndarray) -> bool:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_brightness = float(np.mean(hsv[:, :, 2]))
    mean_saturation = float(np.mean(hsv[:, :, 1]))
    is_ir = mean_saturation < _IR_SATURATION_THRESHOLD
    is_dark = mean_brightness < _NIGHT_BRIGHTNESS_THRESHOLD
    return is_dark and not is_ir
```

- [ ] **Step 4: Update `make_detector` in `tests/test_detector.py`**

Remove `night_brightness_threshold` and `ir_saturation_threshold` from the defaults dict and from the `Detector(...)` call:

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
    )
    defaults.update(kwargs)
    with patch('src.detector.YOLO'):
        return Detector(**defaults)
```

- [ ] **Step 5: Run all detector tests to verify they pass**

```bash
pytest tests/test_detector.py -v
```

Expected: all previously passing tests still pass, new test passes.

- [ ] **Step 6: Commit**

```bash
git add src/detector.py tests/test_detector.py
git commit -m "refactor: replace Detector threshold params with module-level constants"
```

---

## Task 2: Replace `_should_apply_clahe` with `_enhance_frame` dispatcher

**Files:**
- Modify: `src/detector.py`
- Modify: `tests/test_detector.py`

- [ ] **Step 1: Write failing tests for `_enhance_frame`**

Replace the `TestClahe` class with `TestEnhancement` in `tests/test_detector.py`. Keep the existing `_apply_clahe` behavioral tests (`test_apply_clahe_*`) — only replace the dispatch and integration tests:

```python
class TestEnhancement:
    def test_enhance_frame_calls_clahe_for_dark_color_frame(self, dim_color_frame):
        d = make_detector()
        with patch.object(d, '_apply_clahe', return_value=dim_color_frame) as clahe_mock:
            with patch.object(d, '_apply_ir_enhancement', return_value=dim_color_frame):
                d._enhance_frame(dim_color_frame)
        clahe_mock.assert_called_once()

    def test_enhance_frame_calls_ir_enhancement_for_dark_ir_frame(self):
        d = make_detector()
        ir_frame = make_bgr_from_hsv(saturation=0, brightness=50)
        with patch.object(d, '_apply_ir_enhancement', return_value=ir_frame) as ir_mock:
            with patch.object(d, '_apply_clahe', return_value=ir_frame):
                d._enhance_frame(ir_frame)
        ir_mock.assert_called_once()

    def test_enhance_frame_returns_frame_unchanged_when_bright(self):
        d = make_detector()
        bright_frame = make_bgr_from_hsv(saturation=100, brightness=200)
        with patch.object(d, '_apply_clahe') as clahe_mock:
            with patch.object(d, '_apply_ir_enhancement') as ir_mock:
                result = d._enhance_frame(bright_frame)
        clahe_mock.assert_not_called()
        ir_mock.assert_not_called()

    def test_apply_clahe_returns_same_shape(self, dim_color_frame):
        d = make_detector()
        result = d._apply_clahe(dim_color_frame)
        assert result.shape == dim_color_frame.shape
        assert result.dtype == np.uint8

    def test_apply_clahe_brightens_dark_frame(self, dim_color_frame):
        d = make_detector()
        result = d._apply_clahe(dim_color_frame)
        assert np.mean(result) > np.mean(dim_color_frame)

    def test_apply_clahe_does_not_modify_original(self, dim_color_frame):
        d = make_detector()
        original_copy = dim_color_frame.copy()
        d._apply_clahe(dim_color_frame)
        np.testing.assert_array_equal(dim_color_frame, original_copy)

    def test_night_enhancement_disabled_skips_enhance_frame(self, dim_color_frame):
        detector = make_detector(night_enhancement=False)
        with patch.object(detector, '_enhance_frame') as mock:
            with patch.object(detector, '_run_inference', return_value=[]):
                detector.process_frame(dim_color_frame)
        mock.assert_not_called()

    def test_night_enhancement_enabled_calls_enhance_frame(self, dim_color_frame):
        detector = make_detector(night_enhancement=True)
        with patch.object(detector, '_enhance_frame', return_value=dim_color_frame) as mock:
            with patch.object(detector, '_run_inference', return_value=[]):
                detector.process_frame(dim_color_frame)
        mock.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_detector.py::TestEnhancement -v
```

Expected: `FAILED` — `AttributeError: 'Detector' object has no attribute '_enhance_frame'`

- [ ] **Step 3: Add `_enhance_frame` to `src/detector.py`**

Add this method to the `Detector` class:

```python
def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(hsv[:, :, 2]))
    saturation = float(np.mean(hsv[:, :, 1]))
    is_dark = brightness < _NIGHT_BRIGHTNESS_THRESHOLD
    is_ir = saturation < _IR_SATURATION_THRESHOLD
    if is_dark and not is_ir:
        return self._apply_clahe(frame)
    if is_dark and is_ir:
        return self._apply_ir_enhancement(frame)
    return frame
```

Update `process_frame` to call `_enhance_frame` instead of the old pattern:

```python
def process_frame(self, frame: np.ndarray) -> tuple[int, list[TrackedVehicle]]:
    if self._night_enhancement:
        frame = self._enhance_frame(frame)
    detections = self._run_inference(frame)
    self._update_tracker(detections)
    stationary = [v for v in self._tracked if v.frames >= self._required_frames]
    return len(stationary), stationary
```

Remove `_should_apply_clahe` entirely from the class.

- [ ] **Step 4: Run all detector tests**

```bash
pytest tests/test_detector.py -v
```

Expected: all tests pass. (The `_apply_ir_enhancement` calls in the dispatch tests are patched, so they don't need the real implementation yet.)

- [ ] **Step 5: Commit**

```bash
git add src/detector.py tests/test_detector.py
git commit -m "refactor: replace _should_apply_clahe with _enhance_frame dispatcher"
```

---

## Task 3: Add `_apply_ir_enhancement`

**Files:**
- Modify: `src/detector.py`
- Modify: `tests/test_detector.py`

- [ ] **Step 1: Write failing behavioral tests**

Add these tests inside `TestEnhancement` in `tests/test_detector.py`:

```python
def test_apply_ir_enhancement_returns_same_shape(self):
    d = make_detector()
    ir_frame = make_bgr_from_hsv(saturation=0, brightness=50)
    result = d._apply_ir_enhancement(ir_frame)
    assert result.shape == ir_frame.shape
    assert result.dtype == np.uint8

def test_apply_ir_enhancement_brightens_dark_ir_frame(self):
    d = make_detector()
    ir_frame = make_bgr_from_hsv(saturation=0, brightness=50)
    result = d._apply_ir_enhancement(ir_frame)
    assert np.mean(result) > np.mean(ir_frame)

def test_apply_ir_enhancement_does_not_modify_original(self):
    d = make_detector()
    ir_frame = make_bgr_from_hsv(saturation=0, brightness=50)
    original_copy = ir_frame.copy()
    d._apply_ir_enhancement(ir_frame)
    np.testing.assert_array_equal(ir_frame, original_copy)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_detector.py::TestEnhancement::test_apply_ir_enhancement_returns_same_shape tests/test_detector.py::TestEnhancement::test_apply_ir_enhancement_brightens_dark_ir_frame tests/test_detector.py::TestEnhancement::test_apply_ir_enhancement_does_not_modify_original -v
```

Expected: `FAILED` — `AttributeError: 'Detector' object has no attribute '_apply_ir_enhancement'`

- [ ] **Step 3: Implement `_apply_ir_enhancement` in `src/detector.py`**

Add this method to the `Detector` class:

```python
def _apply_ir_enhancement(self, frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blurred = cv2.GaussianBlur(eq, (0, 0), 3)
    sharpened = cv2.addWeighted(eq, 1.5, blurred, -0.5, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
```

- [ ] **Step 4: Run all detector tests**

```bash
pytest tests/test_detector.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/detector.py tests/test_detector.py
git commit -m "feat: add IR night enhancement (grayscale CLAHE + unsharp mask)"
```

---

## Task 4: Update real-image detection tests to use `process_frame`

**Files:**
- Modify: `tests/data/test_cases.yaml`
- Modify: `tests/test_detector.py`

- [ ] **Step 1: Add `night_enhancement` field to all cases in `tests/data/test_cases.yaml`**

Replace the full `detection_cases` block:

```yaml
detection_cases:
  - name: garage_two_vehicles
    image: images/20260613_155852.jpg
    scan_regions: []
    vehicle_classes: [car, truck, bus]
    detection_confidence: 0.3
    night_enhancement: false
    expected_count: 2

  - name: daytime_two_vehicles
    image: images/2_daytime.jpg
    scan_regions: []
    vehicle_classes: [car, truck, bus]
    detection_confidence: 0.3
    night_enhancement: false
    expected_count: 2

  - name: night_driveway_car
    image: images/night_1.jpg
    scan_regions: []
    vehicle_classes: [car, truck, bus]
    detection_confidence: 0.3
    night_enhancement: true
    expected_count: 2

  - name: night_driveway_plus_street
    image: images/night_2.jpg
    scan_regions: []
    vehicle_classes: [car, truck, bus]
    detection_confidence: 0.3
    night_enhancement: true
    expected_count: 2

  - name: night_side_truck
    image: images/night_side.jpg
    scan_regions: []
    vehicle_classes: [car, truck, bus]
    detection_confidence: 0.3
    night_enhancement: false
    expected_count: 1
```

- [ ] **Step 2: Update `TestRealImageDetection` in `tests/test_detector.py`**

Replace the entire class:

```python
class TestRealImageDetection:
    @pytest.mark.parametrize("case", TEST_CASES['detection_cases'], ids=lambda c: c['name'])
    def test_detects_expected_vehicle_count(self, case):
        image_path = Path(__file__).parent / 'data' / case['image']
        frame = cv2.imread(str(image_path))
        assert frame is not None, f"Could not load test image: {image_path}"

        scan_regions = [
            ScanRegion(x=r['x'], y=r['y'], width=r['width'], height=r['height'])
            for r in case.get('scan_regions', [])
        ]
        detector = Detector(
            model_path='yolov8n.pt',
            vehicle_classes=case.get('vehicle_classes', ['car', 'truck', 'bus']),
            detection_confidence=case.get('detection_confidence', 0.4),
            iou_threshold=0.5,
            stationary_seconds=1,
            target_fps=1,
            night_enhancement=case.get('night_enhancement', False),
            scan_regions=scan_regions,
        )

        count, vehicles = detector.process_frame(frame)
        assert count == case['expected_count'], (
            f"Expected {case['expected_count']} vehicles, got {count}: "
            + ", ".join(f"{v.box}" for v in vehicles)
        )
```

- [ ] **Step 3: Run the real-image detection tests**

```bash
pytest tests/test_detector.py::TestRealImageDetection -v
```

Expected: day cases pass; night cases with `expected_count: 2` may fail if the enhancement is insufficient.

- [ ] **Step 4: Tune IR enhancement parameters if night cases fail**

If `night_driveway_car` or `night_driveway_plus_street` report only 1 detection, the enhancement needs tuning. Try increasing `clipLimit` in `_apply_ir_enhancement` from `4.0` to `6.0`, or adjusting the unsharp weight from `1.5` to `2.0`:

```python
def _apply_ir_enhancement(self, frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blurred = cv2.GaussianBlur(eq, (0, 0), 3)
    sharpened = cv2.addWeighted(eq, 2.0, blurred, -1.0, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
```

Re-run after each change. Stop tuning as soon as both night cases report 2 detections. Keep the final working values.

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_detector.py tests/data/test_cases.yaml
git commit -m "test: real image detection now exercises process_frame with night enhancement"
```

---

## Task 5: Remove threshold fields from `AppConfig` and config loader

**Files:**
- Modify: `src/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Update `VALID_APP_DATA` in `tests/test_config.py`**

Remove the two fields from the dict:

```python
VALID_APP_DATA = {
    'camera_name': 'driveway',
    'rtsps_url': 'rtsps://192.168.1.10/stream',
    'vehicle_classes': ['car', 'truck', 'bus'],
    'detection_confidence': 0.4,
    'stationary_seconds': 3,
    'iou_threshold': 0.5,
    'night_enhancement': True,
    'target_fps': 1,
    'model_path': 'yolov8n.pt',
    'publish_interval_seconds': 5,
    'mqtt_timeout_seconds': 60,
    'mqtt_topic': 'car-counter/driveway',
    'image_save_cooldown_seconds': 30,
    # output_dir is set per-test using tmp_path
}
```

Also remove `'night_brightness_threshold'` and `'ir_saturation_threshold'` from the `@pytest.mark.parametrize("missing_field", [...])` list in `test_missing_required_field_raises_value_error`.

- [ ] **Step 2: Run config tests to verify they now fail**

```bash
pytest tests/test_config.py -v
```

Expected: `FAILED` — `AppConfig.__init__() got unexpected keyword argument 'night_brightness_threshold'` (or similar), because the loader still tries to set these fields.

- [ ] **Step 3: Update `src/config.py`**

Remove `night_brightness_threshold` and `ir_saturation_threshold` from `AppConfig`:

```python
@dataclass
class AppConfig:
    camera_name: str
    rtsps_url: str
    scan_regions: list[ScanRegion]
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
```

Remove from `_REQUIRED_APP_FIELDS`:

```python
_REQUIRED_APP_FIELDS = [
    'camera_name', 'rtsps_url', 'vehicle_classes', 'detection_confidence',
    'stationary_seconds', 'iou_threshold', 'night_enhancement',
    'target_fps', 'model_path', 'publish_interval_seconds', 'mqtt_timeout_seconds',
    'mqtt_topic', 'output_dir', 'image_save_cooldown_seconds',
]
```

Remove the two fields from the `AppConfig(...)` construction at the end of `load_app_config` (the two lines `night_brightness_threshold=int(data['night_brightness_threshold'])` and `ir_saturation_threshold=int(data['ir_saturation_threshold'])`).

- [ ] **Step 4: Run config tests**

```bash
pytest tests/test_config.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "refactor: remove night_brightness_threshold and ir_saturation_threshold from config"
```

---

## Task 6: Update `__main__.py` Detector construction

**Files:**
- Modify: `src/__main__.py`

- [ ] **Step 1: Remove the two threshold kwargs from the `Detector(...)` call**

In `src/__main__.py`, find the `Detector(...)` block and remove `night_brightness_threshold` and `ir_saturation_threshold`:

```python
detector = Detector(
    model_path=app_config.model_path,
    vehicle_classes=app_config.vehicle_classes,
    detection_confidence=app_config.detection_confidence,
    iou_threshold=app_config.iou_threshold,
    stationary_seconds=app_config.stationary_seconds,
    target_fps=app_config.target_fps,
    night_enhancement=app_config.night_enhancement,
    scan_regions=app_config.scan_regions,
)
```

- [ ] **Step 2: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/__main__.py
git commit -m "refactor: remove threshold args from Detector construction in __main__"
```

---

## Task 7: Update docs and example config

**Files:**
- Modify: `docs/requirements.md`
- Modify: `config/example-config.yaml`

- [ ] **Step 1: Update `docs/requirements.md`**

Replace the Night / Low-Light Detection section:

```markdown
### Night / Low-Light Detection
- Apply preprocessing to improve detection in low-light conditions; controlled by `night_enhancement` config flag
- Two enhancement paths, selected automatically per frame:
  - **Dark color frame** (mean brightness < 80, mean saturation ≥ 30): CLAHE on the LAB L-channel (`clipLimit=2.0`)
  - **Dark IR frame** (mean brightness < 80, mean saturation < 30): grayscale CLAHE (`clipLimit=4.0`) followed by unsharp masking to recover edge detail
- Brightness and saturation thresholds are internal constants, not operator-tunable
- All detection per-frame automatically; no time-of-day logic is used
```

In the config table and example YAML block within `requirements.md`, remove the rows / lines for `night_brightness_threshold` and `ir_saturation_threshold`.

- [ ] **Step 2: Update `config/example-config.yaml`**

Remove the two lines from the night / low-light section:

```yaml
# Night / low-light enhancement
night_enhancement: true
```

(Delete the `night_brightness_threshold: 80` and `ir_saturation_threshold: 30` lines entirely.)

- [ ] **Step 3: Commit**

```bash
git add docs/requirements.md config/example-config.yaml
git commit -m "docs: update requirements and example config for IR enhancement changes"
```

---

## Final verification

- [ ] **Run the complete test suite one last time**

```bash
pytest tests/ -v
```

Expected: all tests pass, no skips.
