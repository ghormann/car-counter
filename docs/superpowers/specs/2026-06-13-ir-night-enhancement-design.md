# IR Night Enhancement — Design Spec

**Date:** 2026-06-13

## Problem

The detector misses vehicles in IR nighttime images. IR cameras produce pure grayscale frames
(saturation ≈ 0). The current enhancement logic explicitly skips IR frames on the assumption
that "the camera's IR illumination is already handling the scene." Real-world images show this
assumption is wrong: YOLOv8n detects only 1 of 2 visible vehicles in IR night conditions.

## Goal

Detect all visible vehicles in IR nighttime frames by adding a multi-stage IR enhancement path,
while keeping the existing color CLAHE path unchanged. Remove the two now-unnecessary config
thresholds, making enhancement fully automatic.

---

## Architecture

### Enhancement dispatch (`detector.py`)

Two module-level constants replace the removed config params:

```python
_NIGHT_BRIGHTNESS_THRESHOLD = 80   # mean pixel brightness (0–255)
_IR_SATURATION_THRESHOLD    = 30   # mean saturation (0–255) below which IR mode is inferred
```

`process_frame` calls `_enhance_frame` (replacing the current `_should_apply_clahe` +
`_apply_clahe` inline call):

```python
def process_frame(self, frame):
    if self._night_enhancement:
        frame = self._enhance_frame(frame)
    detections = self._run_inference(frame)
    ...
```

`_enhance_frame` computes brightness and saturation once and dispatches:

```python
def _enhance_frame(self, frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(hsv[:, :, 2]))
    saturation = float(np.mean(hsv[:, :, 1]))
    is_dark = brightness < _NIGHT_BRIGHTNESS_THRESHOLD
    is_ir   = saturation < _IR_SATURATION_THRESHOLD
    if is_dark and not is_ir:
        return self._apply_clahe(frame)
    if is_dark and is_ir:
        return self._apply_ir_enhancement(frame)
    return frame
```

### Color enhancement (unchanged)

`_apply_clahe` is unchanged: CLAHE on the LAB L-channel with `clipLimit=2.0`.

### IR enhancement (new)

Operates on the grayscale channel directly since no color information exists in IR frames.
Uses a higher `clipLimit` than the color path (4.0 vs 2.0) to more aggressively normalize
the extreme contrast typical of IR night scenes (bright headlights, dark surroundings).
Unsharp masking recovers edge detail that CLAHE smooths out.

```python
def _apply_ir_enhancement(self, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blurred = cv2.GaussianBlur(eq, (0, 0), 3)
    sharpened = cv2.addWeighted(eq, 1.5, blurred, -0.5, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
```

### `Detector.__init__` signature

Remove `night_brightness_threshold` and `ir_saturation_threshold` parameters.
`night_enhancement: bool` is kept — it still provides a way to disable all enhancement.

---

## Config changes

### Removed from `AppConfig` and `config.py`

- `night_brightness_threshold`
- `ir_saturation_threshold`

Removed from `_REQUIRED_APP_FIELDS`, the YAML loader, and startup validation.

### `night_enhancement` stays

Still a required YAML field. Controls whether `_enhance_frame` is called at all.

### Files to update

- `src/config.py` — remove fields from dataclass, loader, and required field list
- `src/__main__.py` — remove the two args when constructing `Detector`
- `config/example-config.yaml` — remove the two fields
- `docs/requirements.md` — update Night/Low-Light Detection section and config table

---

## Test changes

### `make_detector()` helper

Remove `night_brightness_threshold` and `ir_saturation_threshold` from defaults dict and
`Detector(...)` call.

### `TestClahe` → `TestEnhancement`

`_should_apply_clahe` no longer exists as a public-ish method. Tests update to target
`_enhance_frame` and the two leaf methods:

| Old test | New test |
|---|---|
| `test_should_apply_clahe_parametrized` | `test_enhance_frame_dispatches_correctly` — parametrize over (saturation, brightness) → which method is called |
| `test_apply_clahe_returns_same_shape` | Keep, unchanged |
| `test_apply_clahe_brightens_dark_frame` | Keep, unchanged |
| `test_apply_clahe_does_not_modify_original` | Keep, unchanged |
| `test_night_enhancement_disabled_skips_clahe` | Update: assert `_enhance_frame` is not called |
| `test_night_enhancement_enabled_applies_clahe_to_dark_frame` | Update: assert `_apply_clahe` called for dark color frame |
| _(new)_ | `test_enhance_frame_calls_ir_enhancement_for_dark_ir_frame` |
| _(new)_ | `test_apply_ir_enhancement_returns_same_shape` |
| _(new)_ | `test_apply_ir_enhancement_brightens_dark_frame` |
| _(new)_ | `test_apply_ir_enhancement_does_not_modify_original` |

### `TestRealImageDetection`

Change from calling `_run_inference` to `process_frame`:

```python
count, _ = detector.process_frame(frame)
assert count == case['expected_count']
```

Detector constructed with `stationary_seconds=1, target_fps=1` (required_frames=1) so a
single frame call is sufficient to register stationary vehicles.

`night_enhancement` is read from the test case YAML (default `false` for existing day cases).

### `test_cases.yaml` changes

Night cases get `night_enhancement: true` and `expected_count: 2`:

```yaml
- name: night_driveway_car        # night_1.jpg — expected_count: 2
- name: night_driveway_plus_street # night_2.jpg — expected_count: 2
```

Day cases get `night_enhancement: false` (explicit, for clarity).

---

## What is not changing

- `_apply_clahe` implementation
- `_run_inference`
- `_update_tracker`, `_compute_iou`, `_is_in_scan_regions`
- All other test classes (`TestComputeIou`, `TestUpdateTracker`, `TestStationaryCount`,
  `TestScanRegions`, `TestRunInference`)
- MQTT, stream, image saver, metrics — no changes

---

## Requirements.md updates

- Night/Low-Light Detection section: replace the IR-skip rationale with the new two-path
  description (color CLAHE for dark color frames, grayscale CLAHE + unsharp for dark IR frames)
- Remove `night_brightness_threshold` and `ir_saturation_threshold` from the config table
  and example YAML block
- Note that thresholds are internal constants, not operator-tunable
