"""One-shot script to annotate vlcsnap images with detected vehicle bounding boxes."""
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.detector import Detector

IMAGES_DIR = Path("tests/data/images")
NEW_IMAGES = [p.name for p in IMAGES_DIR.iterdir()
              if not p.name.startswith("annotated_")]

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

for filename in NEW_IMAGES:
    path = IMAGES_DIR / filename
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"SKIP (unreadable): {filename}")
        continue

    # Run single-frame detection; frames=1 so nothing is "stationary" yet,
    # so we reach into _run_inference directly to get all detections.
    if detector._night_enhancement:
        enhanced = detector._enhance_frame(frame.copy())
    else:
        enhanced = frame.copy()
    detections = detector._run_inference(enhanced)

    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out_path = IMAGES_DIR / f"annotated_{filename}"
    cv2.imwrite(str(out_path), annotated)
    print(f"{filename}: {len(detections)} detections -> {out_path.name}")
