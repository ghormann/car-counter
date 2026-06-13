# Design: `ignore_regions`

**Date:** 2026-06-13

## Summary

Add an optional `ignore_regions` configuration list that suppresses vehicles from being counted when 95% or more of their bounding box falls inside an exclusion zone. Ignored regions are rendered in blue on annotated images with an "exclude" label.

---

## Config (`config.py`)

Add a new `IgnoreRegion` dataclass with the same `x`, `y`, `width`, `height` integer fields as the existing `ScanRegion`. Add `ignore_regions: list[IgnoreRegion]` to `AppConfig`. Parsing mirrors `scan_regions`: read from `data.get('ignore_regions', [])`, default to empty list (field is optional).

```yaml
ignore_regions:
  - { x: 50, y: 0, width: 200, height: 100 }
```

---

## Detection Logic (`detector.py`)

After a detection passes the `_is_in_scan_regions` check, run a new `_is_in_ignore_regions` check. A detection is suppressed if any ignore region covers ≥ 95% of its bounding box area.

**Coverage calculation:**

```
intersection_area = overlap between bbox and ignore region
vehicle_area = (x2 - x1) * (y2 - y1)
suppressed if intersection_area / vehicle_area >= 0.95
```

`Detector.__init__` accepts a new `ignore_regions: list[IgnoreRegion]` parameter (default `[]`). The check runs in `_run_inference` immediately after `_is_in_scan_regions` returns `True`.

---

## Annotation (`image_saver.py`)

`ImageSaver._annotate` and `ImageSaver.save` receive a new `ignore_regions: list[IgnoreRegion]` parameter (default `[]`). Each ignore region is drawn as a blue rectangle (`(255, 0, 0)` in BGR) with the text label `"exclude"` in the top-left corner of the region, using the same font and style as other labels.

`annotate_images.py` passes `ignore_regions=[]` to match the existing `scan_regions=[]` pattern.

---

## Files Changed

| File | Change |
|------|--------|
| `src/config.py` | Add `IgnoreRegion` dataclass; add `ignore_regions` field to `AppConfig`; parse from YAML |
| `src/detector.py` | Add `ignore_regions` param; add `_is_in_ignore_regions` method; call it in `_run_inference` |
| `src/image_saver.py` | Add `ignore_regions` param to `save` and `_annotate`; draw blue boxes with "exclude" label |
| `annotate_images.py` | Pass `ignore_regions=[]` |
| `config/example-config.yaml` | Add commented-out `ignore_regions` example |
| `docs/config-reference.md` | Add `ignore_regions` section after `scan_regions` |
| `docs/requirements.md` | Add `ignore_regions` to functional requirements and config YAML example |
| `README.md` | Note `ignore_regions` feature |

---

## Tests

- `tests/test_config.py`: parse config with `ignore_regions`; parse config without `ignore_regions` (defaults to `[]`)
- `tests/test_detector.py`: vehicle 95%+ inside ignore region is suppressed; vehicle <95% inside ignore region is not suppressed; ignore regions applied after scan regions pass; no ignore regions → no suppression

---

## Constraints

- 95% overlap threshold is hardcoded (matches spec)
- `ignore_regions` is always optional; absence defaults to `[]`
- Ignore regions are drawn blue (`255, 0, 0` BGR); scan regions remain green
