import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.config import ScanRegion
from src.detector import TrackedVehicle

logger = logging.getLogger(__name__)


class ImageSaver:
    def __init__(self, output_dir: Path, camera_name: str, cooldown_seconds: int):
        self._output_dir = output_dir
        self._camera_name = camera_name
        self._cooldown_seconds = cooldown_seconds
        self._last_save_time = 0.0

    def save(
        self,
        frame: np.ndarray,
        stationary_vehicles: list[TrackedVehicle],
        scan_regions: list[ScanRegion],
        prefix: str = "",
    ) -> Path | None:
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
        annotated = self._annotate(frame.copy(), stationary_vehicles, scan_regions)

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

    def _annotate(
        self,
        frame: np.ndarray,
        stationary_vehicles: list[TrackedVehicle],
        scan_regions: list[ScanRegion],
    ) -> np.ndarray:
        frame = frame.copy()
        for vehicle in stationary_vehicles:
            x1, y1, x2, y2 = (int(c) for c in vehicle.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red

        for region in scan_regions:
            x1, y1 = region.x, region.y
            x2, y2 = region.x + region.width, region.y + region.height
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green

        return frame
