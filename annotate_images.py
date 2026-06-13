"""One-shot script to annotate vlcsnap images with detected vehicle bounding boxes."""
import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.detector import Detector, TrackedVehicle
from src.image_saver import ImageSaver

IMAGES_DIR = Path("tests/data/images")
NEW_IMAGES = [p.name for p in IMAGES_DIR.iterdir()
              if not p.name.startswith("annotated_")]

detector = Detector(
    model_path="yolov8x.pt",
    vehicle_classes=["car", "truck", "bus"],
    detection_confidence=0.3,
    iou_threshold=0.5,
    stationary_seconds=3,
    target_fps=1,
    night_enhancement=True,
    scan_regions=[],
    ignore_regions=[],
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

    vehicles = [
        TrackedVehicle(box=det.box, frames=1, class_name=det.class_name, confidence=det.confidence)
        for det in detections
    ]
    annotated = ImageSaver._annotate(frame.copy(), vehicles, scan_regions=[])

    out_path = IMAGES_DIR / f"annotated_{filename}"
    cv2.imwrite(str(out_path), annotated)
    print(f"{filename}: {len(detections)} detections -> {out_path.name}")
