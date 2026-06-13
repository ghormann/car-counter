"""YOLOv8-based vehicle detector with IoU tracking, scan/ignore regions, and night enhancement."""
import math
import logging
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import BoxedRegion

logger = logging.getLogger(__name__)

# Mean V-channel value (0–255) in HSV below which a frame is treated as dark.
# 80 corresponds to roughly 31 % of full brightness, a level where colour
# cameras lose significant detail without contrast enhancement.
_NIGHT_BRIGHTNESS_THRESHOLD = 80

# Mean S-channel value (0–255) in HSV below which a dark frame is assumed to
# come from an IR camera rather than a low-light colour camera. IR sensors
# produce near-grayscale output, so saturation stays very low (<12 %) even
# when the scene is well-exposed by infrared illumination.
_IR_SATURATION_THRESHOLD = 30


@dataclass
class Detection:
    """A single raw detection from the YOLO model for one frame."""

    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    class_name: str
    confidence: float


@dataclass
class TrackedVehicle:
    """A vehicle being tracked across consecutive frames.

    ``frames`` counts how many consecutive frames this vehicle has been
    continuously detected at roughly the same position (via IoU matching).
    A vehicle is considered stationary once ``frames`` reaches the configured
    threshold.
    """

    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    frames: int
    class_name: str = ""
    confidence: float = 0.0


class Detector:
    """Detects and tracks stationary vehicles in video frames.

    Uses a YOLOv8 model for per-frame detection, greedy IoU matching for
    frame-to-frame tracking, and optional night/IR image enhancement.
    Only detections that fall inside configured scan regions (and outside
    ignore regions) are considered.
    """

    def __init__(
        self,
        model_path: str,
        vehicle_classes: list[str],
        detection_confidence: float,
        iou_threshold: float,
        stationary_seconds: int,
        target_fps: int,
        night_enhancement: bool,
        scan_regions: list[BoxedRegion],
        ignore_regions: list[BoxedRegion],
    ):
        """Initialize the detector.

        Args:
            model_path: Path to the YOLOv8 weights file (.pt).
            vehicle_classes: COCO class names to treat as vehicles
                (e.g. ``["car", "truck", "bus"]``).
            detection_confidence: Minimum YOLO confidence score [0, 1] to
                accept a detection.
            iou_threshold: IoU value [0, 1] used for two purposes: NMS
                deduplication of raw detections, and frame-to-frame tracker
                matching.
            stationary_seconds: How many continuous seconds a vehicle must
                remain at the same position before it is counted as stationary.
            target_fps: Expected frame rate of the incoming stream; combined
                with ``stationary_seconds`` to compute the required frame count.
            night_enhancement: When ``True``, apply CLAHE or IR sharpening
                before inference based on frame brightness/saturation.
            scan_regions: Bounding boxes that define the area(s) of interest.
                A detection must overlap at least one scan region to be kept.
                An empty list means the entire frame is scanned.
            ignore_regions: Bounding boxes for areas to suppress. A detection
                whose bounding box is ≥ 95 % covered by an ignore region is
                discarded.
        """
        self._model = YOLO(model_path)
        self._vehicle_classes = vehicle_classes
        self._detection_confidence = detection_confidence
        self._iou_threshold = iou_threshold
        self._required_frames = math.ceil(stationary_seconds * target_fps)
        self._night_enhancement = night_enhancement
        self._scan_regions = scan_regions
        self._ignore_regions = ignore_regions
        self._tracked: list[TrackedVehicle] = []

    def process_frame(self, frame: np.ndarray) -> tuple[int, list[TrackedVehicle]]:
        """Run detection and tracking on a single video frame.

        Optionally enhances the frame for low-light conditions, runs YOLO
        inference, updates the tracker, and returns currently stationary
        vehicles.

        Args:
            frame: BGR image as a NumPy array (H × W × 3).

        Returns:
            A tuple of ``(count, vehicles)`` where ``count`` is the number of
            stationary vehicles and ``vehicles`` is the corresponding list of
            :class:`TrackedVehicle` objects.
        """
        if self._night_enhancement:
            frame = self._enhance_frame(frame)
        detections = self._run_inference(frame)
        self._update_tracker(detections)
        stationary = [v for v in self._tracked if v.frames >= self._required_frames]
        return len(stationary), stationary

    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Select and apply the appropriate low-light enhancement.

        Checks mean HSV brightness and saturation to distinguish three
        conditions:

        * **Normal light** — frame returned unchanged.
        * **Dark, colour camera** — CLAHE applied in LAB space to boost
          perceived luminance without blowing out colours.
        * **Dark, IR camera** (low saturation) — grayscale CLAHE followed by
          unsharp masking to recover edge detail lost in IR imagery.

        Args:
            frame: BGR image as a NumPy array.

        Returns:
            Enhanced BGR image (same shape as input).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = float(np.mean(hsv[:, :, 2]))
        saturation = float(np.mean(hsv[:, :, 1]))
        is_dark = brightness < _NIGHT_BRIGHTNESS_THRESHOLD
        is_ir = saturation < _IR_SATURATION_THRESHOLD
        if is_dark and not is_ir:
            logger.debug("Applying CLAHE enhancement (brightness=%.1f, saturation=%.1f)", brightness, saturation)
            return self._apply_clahe(frame)
        if is_dark and is_ir:
            logger.debug("Applying IR enhancement (brightness=%.1f, saturation=%.1f)", brightness, saturation)
            return self._apply_ir_enhancement(frame)
        logger.debug("No enhancement applied (brightness=%.1f, saturation=%.1f)", brightness, saturation)
        return frame

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the L channel of a BGR frame.

        Converts to LAB colour space so that contrast enhancement acts only on
        luminance, preserving hue and saturation.

        Args:
            frame: BGR image as a NumPy array.

        Returns:
            Contrast-enhanced BGR image.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_eq = cv2.merge([clahe.apply(l_ch), a_ch, b_ch])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def _apply_ir_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE and unsharp masking to an IR (near-monochrome) frame.

        IR cameras produce low-saturation, low-contrast images. This method
        first equalises contrast with a higher clip limit than the colour CLAHE
        path, then sharpens using an unsharp mask (``1.5 × original - 0.5 ×
        blurred``) to recover vehicle edges.

        Args:
            frame: BGR image as a NumPy array (typically near-grayscale).

        Returns:
            Sharpened BGR image suitable for YOLO inference.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        blurred = cv2.GaussianBlur(eq, (0, 0), 3)
        sharpened = cv2.addWeighted(eq, 1.5, blurred, -0.5, 0)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def _run_inference(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLO inference and return filtered, NMS-deduplicated detections.

        Processing steps:

        1. Run the YOLO model on the frame.
        2. Keep only boxes whose class is in ``_vehicle_classes`` and whose
           confidence meets ``_detection_confidence``.
        3. Discard boxes that don't overlap any scan region or that are
           95 %+ covered by an ignore region.
        4. Sort surviving boxes by confidence (descending) and apply greedy
           IoU-based NMS to remove overlapping duplicates.

        Args:
            frame: BGR image as a NumPy array.

        Returns:
            List of :class:`Detection` objects, highest confidence first, with
            duplicates removed.
        """
        results = self._model(frame, verbose=False)
        candidates = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self._model.names[class_id]
                if class_name in self._vehicle_classes and conf >= self._detection_confidence:
                    if self._is_in_scan_regions((x1, y1, x2, y2)) and not self._is_in_ignore_regions((x1, y1, x2, y2)):
                        candidates.append(Detection(box=(x1, y1, x2, y2), class_name=class_name, confidence=conf))
        candidates.sort(key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []
        for candidate in candidates:
            if not any(self._compute_iou(candidate.box, k.box) >= self._iou_threshold for k in kept):
                kept.append(candidate)
        return kept

    def _update_tracker(self, detections: list[Detection]):
        """Match detections to tracked vehicles and update frame counts.

        Uses greedy IoU matching: each detection is paired with the
        highest-IoU unmatched tracked vehicle. A match is accepted when IoU
        meets ``_iou_threshold``. Matched vehicles carry their frame count
        forward (incremented by one); unmatched detections start new tracks at
        frame count 1. Tracked vehicles with no matching detection are dropped,
        which resets any in-progress stationary count for that position.

        Args:
            detections: Detections from the current frame, as returned by
                :meth:`_run_inference`.
        """
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
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                ))
            else:
                new_tracked.append(TrackedVehicle(
                    box=detection.box,
                    frames=1,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                ))

        self._tracked = new_tracked

    def _compute_iou(self, box1, box2) -> float:
        """Compute Intersection over Union (IoU) for two axis-aligned boxes.

        Args:
            box1: ``(x1, y1, x2, y2)`` coordinates of the first box.
            box2: ``(x1, y1, x2, y2)`` coordinates of the second box.

        Returns:
            IoU in ``[0, 1]``. Returns ``0.0`` if the union area is zero.
        """
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
        """Return True if the box overlaps at least one scan region.

        An empty scan region list means the full frame is in scope, so this
        always returns ``True`` in that case.

        Args:
            box: ``(x1, y1, x2, y2)`` detection bounding box.

        Returns:
            ``True`` if the detection should be considered, ``False`` if it
            falls entirely outside all configured scan regions.
        """
        if not self._scan_regions:
            return True
        x1, y1, x2, y2 = box
        for region in self._scan_regions:
            rx1, ry1 = region.x, region.y
            rx2, ry2 = region.x + region.width, region.y + region.height
            if x2 > rx1 and x1 < rx2 and y2 > ry1 and y1 < ry2:
                return True
        return False

    def _is_in_ignore_regions(self, box: tuple) -> bool:
        """Return True if the box is substantially covered by an ignore region.

        A detection is suppressed when ≥ 95 % of its area overlaps an ignore
        region. This threshold avoids discarding vehicles that merely touch the
        edge of a masked area (e.g. a parked-car zone that borders a travel
        lane).

        Args:
            box: ``(x1, y1, x2, y2)`` detection bounding box.

        Returns:
            ``True`` if the detection should be suppressed, ``False`` otherwise.
        """
        if not self._ignore_regions:
            return False
        x1, y1, x2, y2 = box
        vehicle_area = (x2 - x1) * (y2 - y1)
        if vehicle_area <= 0:
            return False
        for region in self._ignore_regions:
            rx1, ry1 = region.x, region.y
            rx2, ry2 = region.x + region.width, region.y + region.height
            ix1 = max(x1, rx1)
            iy1 = max(y1, ry1)
            ix2 = min(x2, rx2)
            iy2 = min(y2, ry2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / vehicle_area >= 0.95:
                return True
        return False
