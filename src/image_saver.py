"""Saves annotated frames to disk when stationary vehicles are detected, with a per-camera cooldown."""
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.config import BoxedRegion
from src.detector import TrackedVehicle

logger = logging.getLogger(__name__)


class ImageSaver:
    """Saves annotated JPEG frames to a date-partitioned directory tree.

    Attributes:
        _output_dir: Root directory under which per-camera subdirectories are created.
        _camera_name: Camera identifier used as the first subdirectory level.
        _cooldown_seconds: Minimum seconds between normal (non-prefixed) saves.
        _last_save_time: Monotonic timestamp of the most recent normal save.
    """

    def __init__(self, output_dir: Path, camera_name: str, cooldown_seconds: int):
        """Initialize the saver.

        Args:
            output_dir: Root path where images are stored.
            camera_name: Camera label used to namespace saved images.
            cooldown_seconds: Minimum interval between unprefixed saves; prevents
                flooding disk during sustained detections.
        """
        self._output_dir = output_dir
        self._camera_name = camera_name
        self._cooldown_seconds = cooldown_seconds
        self._last_save_time = 0.0

    def save(
        self,
        frame: np.ndarray,
        stationary_vehicles: list[TrackedVehicle],
        scan_regions: list[BoxedRegion],
        ignore_regions: list[BoxedRegion] = None,
        prefix: str = "",
    ) -> Path | None:
        """Annotate and save a frame to disk if the cooldown has elapsed.

        The output path is ``<output_dir>/<camera>/<year>/<month>/<day>/<prefix><timestamp>.jpg``.
        Cooldown is bypassed when *prefix* is non-empty, allowing forced saves
        (e.g. debug snapshots) without resetting the normal save timer.

        Args:
            frame: Raw BGR frame from the camera.
            stationary_vehicles: Vehicles to highlight with bounding boxes.
            scan_regions: Detection zones drawn in green.
            ignore_regions: Exclusion zones drawn in blue.
            prefix: Optional filename prefix; non-empty values skip cooldown enforcement.

        Returns:
            Path to the saved file, or ``None`` if the cooldown blocked the save
            or an I/O error occurred.
        """
        if ignore_regions is None:
            ignore_regions = []
        now = time.monotonic()
        if not prefix and (now - self._last_save_time) < self._cooldown_seconds:
            return None

        dt = datetime.now(timezone.utc)
        timestamp = dt.strftime("%Y%m%d_%H%M%S")
        save_dir = (
            self._output_dir
            / self._camera_name
            / str(dt.year)
            / f"{dt.month:02d}"
            / f"{dt.day:02d}"
        )

        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error("Failed to create directory %s: %s", save_dir, e)
            return None

        filename = f"{prefix}{timestamp}.jpg"
        path = save_dir / filename
        annotated = self._annotate(frame.copy(), stationary_vehicles, scan_regions, ignore_regions)

        try:
            success = cv2.imwrite(str(path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                logger.error("cv2.imwrite failed for %s", path)
                return None
        except OSError as e:
            logger.error("Failed to save image %s: %s", path, e)
            return None

        if not prefix:
            self._last_save_time = now

        logger.info("Saved image: %s", path)
        return path

    @staticmethod
    def _annotate(
        frame: np.ndarray,
        stationary_vehicles: list[TrackedVehicle],
        scan_regions: list[BoxedRegion],
        ignore_regions: list[BoxedRegion] = None,
    ) -> np.ndarray:
        """Draw bounding boxes and a summary overlay onto *frame*.

        Stationary vehicles are outlined in red with a confidence label.
        Scan regions are outlined in green. Ignore regions are outlined in blue
        with an "exclude" label. A centered vehicle count is rendered at the
        bottom of the frame.

        Args:
            frame: BGR image to annotate (will be copied internally).
            stationary_vehicles: Detected vehicles to highlight.
            scan_regions: Active detection zones to outline.
            ignore_regions: Exclusion zones to outline.

        Returns:
            Annotated copy of *frame*.
        """
        if ignore_regions is None:
            ignore_regions = []
        frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        for vehicle in stationary_vehicles:
            x1, y1, x2, y2 = (int(c) for c in vehicle.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red
            label = f"{vehicle.class_name} {vehicle.confidence:.2f}"
            (lw, lh), baseline = cv2.getTextSize(label, font, 0.5, 1)
            ly = max(y1 - 4, lh + baseline)
            cv2.rectangle(frame, (x1, ly - lh - baseline), (x1 + lw, ly), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, label, (x1, ly - baseline), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        for region in scan_regions:
            x1, y1 = region.x, region.y
            x2, y2 = region.x + region.width, region.y + region.height
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green

        for region in ignore_regions:
            x1, y1 = region.x, region.y
            x2, y2 = region.x + region.width, region.y + region.height
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue
            label = "exclude"
            (lw, lh), baseline = cv2.getTextSize(label, font, 0.5, 1)
            ly = y1 + lh + baseline + 4
            cv2.rectangle(frame, (x1, y1), (x1 + lw, ly), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, label, (x1, ly - baseline), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        total = len(stationary_vehicles)
        summary = f"Vehicles detected: {total}"
        (sw, sh), baseline = cv2.getTextSize(summary, font, 0.8, 2)
        h, w = frame.shape[:2]
        sx = (w - sw) // 2
        sy = h - 20
        cv2.rectangle(frame, (sx - 4, sy - sh - baseline), (sx + sw + 4, sy + baseline), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, summary, (sx, sy), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return frame
