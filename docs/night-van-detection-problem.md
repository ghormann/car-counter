# Night Van Detection Problem

## Summary

YOLOv8n (nano) fails to classify a white van as a vehicle in IR nighttime images. The van
is visible and partially detected, but misclassified as `skateboard` at 0.11 confidence —
below the detection threshold and the wrong class.

## Affected Images

- `tests/data/images/night_2.jpg` — van clearly visible, upper-center of frame
- `tests/data/images/night_1.jpg` — same van, two seconds earlier (same misclassification)

Both images are 2688×1512, IR mode (saturation ≈ 0, brightness ≈ 73).

## What the Model Actually Sees

With IR enhancement applied (grayscale CLAHE clipLimit=4.0 + unsharp mask) and confidence
threshold lowered to 0.05, detections in night_2.jpg are:

| Class      | Confidence | Box (x1,y1,x2,y2)     | Size      | Notes                    |
|------------|------------|------------------------|-----------|--------------------------|
| person     | 0.52       | (396, 218, 439, 332)   | 43×114    | Mailbox — not a vehicle  |
| car        | 0.32       | (632, 358, 972, 710)   | 340×352   | Driveway car ✓           |
| person     | 0.18       | (44, 365, 111, 457)    | 67×92     | Unknown object           |
| skateboard | 0.11       | (1301, 39, 1804, 213)  | 503×174   | **The white van**        |

The van box (1301,39,1804,213) is correctly located in the upper-center of the frame.
The model detects the shape but cannot classify it as a vehicle. At operational confidence
(0.3), the van is completely absent from results.

## Root Cause

YOLOv8n has insufficient capacity to classify vehicles in dark IR at distance. Contributing
factors:

- **IR rendering**: No color, front-lit, dark body — the wide flat profile resembles a
  skateboard silhouette to the nano model
- **Distance/scale**: Van occupies the upper frame where it subtends a relatively small
  solid angle despite its physical size
- **Model size**: YOLOv8n has ~3.2M parameters. It is explicitly optimized for speed over
  accuracy and struggles with ambiguous night IR scenes

No amount of CLAHE or unsharp mask tuning corrects this — the issue is classification
capacity, not image contrast.

## What Was Tried

- IR enhancement: grayscale CLAHE at clipLimit 2.0, 4.0, 6.0, 12.0
- Unsharp mask weights: 1.5/−0.5, 2.0/−1.0, 3.0/−2.0
- Gamma correction: γ = 0.3–0.7
- Standard histogram equalization (cv2.equalizeHist)
- Confidence threshold sweep: 0.05–0.40

None produced a `car`, `truck`, or `bus` classification for the van above 0.15 confidence.

## Fix Applied

**Switched to YOLOv8l (large), ~43M parameters.** YOLOv8l runs at ~15ms/frame on a
Ryzen 7 5825U (well within the 1 FPS budget) and correctly classifies the van in
`night_2.jpg` as `car` at 0.66 confidence.

`night_1.jpg` remains at `expected_count: 1` — the van is undetectable in that frame
even with YOLOv8x (~68M params). The two-second-earlier frame has subtly worse
contrast and no YOLO model classifies the van as a vehicle at any useful confidence.

Cross-class NMS was also added to `detector._run_inference()`: when YOLOv8l detects
the same vehicle as both `car` and `truck` (which it does more often than nano), the
lower-confidence duplicate is suppressed using the same IoU threshold as tracking.

## Reference Files

- Annotated images: `tests/data/images/annotated_night_2_lowconf.jpg` (all detections ≥ 0.05)
- Test cases: `tests/data/test_cases.yaml` (night cases currently at `expected_count: 1`)
- Enhancement code: `src/detector.py` — `_apply_ir_enhancement()`
- Requirements: `docs/requirements.md` — Night / Low-Light Detection section
