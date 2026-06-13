import math
import logging
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import ScanRegion

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    class_name: str
    confidence: float


@dataclass
class TrackedVehicle:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    frames: int


class Detector:
    def __init__(
        self,
        model_path: str,
        vehicle_classes: list[str],
        detection_confidence: float,
        iou_threshold: float,
        stationary_seconds: int,
        target_fps: int,
        night_enhancement: bool,
        night_brightness_threshold: int,
        ir_saturation_threshold: int,
        scan_regions: list[ScanRegion],
    ):
        self._model = YOLO(model_path)
        self._vehicle_classes = vehicle_classes
        self._detection_confidence = detection_confidence
        self._iou_threshold = iou_threshold
        self._required_frames = math.ceil(stationary_seconds * target_fps)
        self._night_enhancement = night_enhancement
        self._night_brightness_threshold = night_brightness_threshold
        self._ir_saturation_threshold = ir_saturation_threshold
        self._scan_regions = scan_regions
        self._tracked: list[TrackedVehicle] = []

    def process_frame(self, frame: np.ndarray) -> tuple[int, list[TrackedVehicle]]:
        if self._night_enhancement and self._should_apply_clahe(frame):
            frame = self._apply_clahe(frame)
        detections = self._run_inference(frame)
        self._update_tracker(detections)
        stationary = [v for v in self._tracked if v.frames >= self._required_frames]
        return len(stationary), stationary

    def _should_apply_clahe(self, frame: np.ndarray) -> bool:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean_brightness = float(np.mean(hsv[:, :, 2]))
        mean_saturation = float(np.mean(hsv[:, :, 1]))
        is_ir = mean_saturation < self._ir_saturation_threshold
        is_dark = mean_brightness < self._night_brightness_threshold
        return is_dark and not is_ir

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_eq = cv2.merge([clahe.apply(l_ch), a_ch, b_ch])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def _run_inference(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(frame, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self._model.names[class_id]
                if class_name in self._vehicle_classes and conf >= self._detection_confidence:
                    if self._is_in_scan_regions((x1, y1, x2, y2)):
                        detections.append(Detection(box=(x1, y1, x2, y2), class_name=class_name, confidence=conf))
        return detections

    def _update_tracker(self, detections: list[Detection]):
        matched_indices: set[int] = set()
        new_tracked: list[TrackedVehicle] = []

        for detection in detections:
            best_iou = 0.0
            best_idx = -1
            for i, tracked in enumerate(self._tracked):
                if i in matched_indices:
                    continue
                iou = self._compute_iou(detection.box, tracked.box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= self._iou_threshold and best_idx >= 0:
                matched_indices.add(best_idx)
                new_tracked.append(TrackedVehicle(
                    box=detection.box,
                    frames=self._tracked[best_idx].frames + 1,
                ))
            else:
                new_tracked.append(TrackedVehicle(box=detection.box, frames=1))

        self._tracked = new_tracked

    def _compute_iou(self, box1, box2) -> float:
        ix1 = max(box1[0], box2[0])
        iy1 = max(box1[1], box2[1])
        ix2 = min(box1[2], box2[2])
        iy2 = min(box1[3], box2[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _is_in_scan_regions(self, box: tuple) -> bool:
        if not self._scan_regions:
            return True
        x1, y1, x2, y2 = box
        for region in self._scan_regions:
            rx1, ry1 = region.x, region.y
            rx2, ry2 = region.x + region.width, region.y + region.height
            if x2 > rx1 and x1 < rx2 and y2 > ry1 and y1 < ry2:
                return True
        return False
