# Image Tiling for Small Object Detection

**Date:** 2026-06-13
**Status:** Approved

## Problem

Small, distant vehicles in the middle/far field of wide-angle night camera frames are missed by single-pass YOLO inference. The model sees them at very low resolution, producing confidence scores below the detection threshold.

## Solution

Run YOLO inference on overlapping tiles of the full frame in addition to a full-frame pass, then merge all detections with NMS before applying scan region filtering. This gives the model a higher effective resolution view of small distant objects without changing any other pipeline behavior.

## Architecture

Tiling is contained entirely within `Detector._run_inference`. No other methods change.

### Tile Generation

Given `tile_width`, `tile_height`, and `tile_overlap` (fraction), generate a grid of tile coordinates that covers the full frame. Tiles overlap by `tile_overlap` in both dimensions to avoid missing objects that straddle tile boundaries.

### Inference Per Tile

For each tile: crop the (already-enhanced) frame, run YOLO inference, remap detected box coordinates from tile-local space back to full-frame space.

### Full-Frame Pass

Also run inference on the full frame (as today). This ensures large vehicles that span multiple tiles are still detected reliably.

### Deduplication

Collect all detections from tile passes and the full-frame pass into one list. Apply the existing IOU-based NMS (already in `_run_inference`) across the merged list to remove duplicates produced by overlapping tiles.

### Scan Region Filtering

`_is_in_scan_regions` runs after NMS on the merged, deduplicated list — identical to current behavior.

## Configuration

Three new optional fields added to `app-config.yaml` and the `Config` dataclass:

```yaml
tile_width: 640      # pixel width of each tile
tile_height: 640     # pixel height of each tile
tile_overlap: 0.2    # fractional overlap between adjacent tiles (0.0–1.0)
```

If omitted, tiling is disabled and behavior is identical to today (full-frame pass only).

## Files Changed

- `src/config.py` — add `tile_width`, `tile_height`, `tile_overlap` fields (all optional with defaults of `None`)
- `src/detector.py` — refactor `_run_inference` to generate tiles, run per-tile inference, remap coordinates, merge with full-frame results, then apply NMS
- `config/app-config.yaml` — add tiling config fields
- `config/example-config.yaml` — add tiling config fields with comments

## Testing

- Existing detection unit tests continue to pass (no tiling configured in test cases)
- Manually verify annotated images show improved detection of distant vehicles
- New test cases can set `tile_width`/`tile_height`/`tile_overlap` per case if needed in future

## Non-Goals

- No changes to `process_frame`, `_enhance_frame`, `_update_tracker`, or `_is_in_scan_regions`
- No new dependencies
- No per-region tiling configuration
