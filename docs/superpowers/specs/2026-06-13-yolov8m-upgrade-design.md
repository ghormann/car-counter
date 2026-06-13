# Design: Upgrade Default Detection Model to YOLOv8m

**Date:** 2026-06-13  
**Status:** Approved

## Problem

YOLOv8n (nano, ~3.2M params) fails to classify a white van as a vehicle in IR nighttime images at distance. The van is detected as `skateboard` at 0.11 confidence — below threshold and wrong class. CLAHE and unsharp mask tuning were exhausted without improvement. The root cause is model classification capacity, not image preprocessing.

## Solution

Replace YOLOv8n with YOLOv8m (~25M params, ~8× nano) as the default baked-in model. The machine is an AMD Ryzen 7 5825U (8 cores / 16 threads) running mostly idle; at 1 FPS the extra compute cost is irrelevant.

YOLOv8m becomes the global default for all deployments. `model_path` remains a config field so operators can override it per camera.

## Verification Gate

Before updating any test expectations, run YOLOv8m on `night_1.jpg` and `night_2.jpg` through the real IR enhancement path (`detector._apply_ir_enhancement()`) and confirm the van box (approximately x1=1301, y1=39, x2=1804, y2=213) classifies as `car`, `truck`, or `bus` at ≥ 0.4 confidence.

- **Detected ≥ 0.4** → proceed; bump `expected_count` to 2 in test cases.
- **Detected at 0.3–0.39** → pause and ask user whether to lower `detection_confidence` or escalate to YOLOv8l.
- **Not detected** → fall back to YOLOv8l and re-verify.

Test expectation changes are conditional on observed evidence, not assumed.

## Files Changed

| File | Change |
|---|---|
| `Dockerfile` | Bake `yolov8m.pt` instead of `yolov8n.pt` |
| `config/example-config.yaml` | `model_path: yolov8m.pt` |
| `docs/requirements.md` | Update Night/Low-Light and Model sections to name YOLOv8m as default; note YOLOv8n is insufficient for night IR |
| `tests/data/test_cases.yaml` | `night_driveway_car` and `night_driveway_plus_street`: `expected_count` 1 → 2 (conditional on verification) |

`docs/config-reference.md` — update model_path default if it names nano.

## Out of Scope

- No CLAHE or unsharp mask tuning changes (proven not to help)
- No per-camera model-switching logic (`model_path` is already per-deployment)
- No new config fields
