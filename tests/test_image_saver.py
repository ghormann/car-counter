import time
import numpy as np
import cv2
import pytest
from pathlib import Path

from src.config import ScanRegion
from src.detector import TrackedVehicle
from src.image_saver import ImageSaver


def make_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_vehicle(x1=10, y1=10, x2=200, y2=200, frames=10):
    return TrackedVehicle(box=(float(x1), float(y1), float(x2), float(y2)), frames=frames)


class TestImageSaverPaths:
    def test_save_creates_jpeg_file(self, tmp_output_dir):
        saver = ImageSaver(tmp_output_dir, "driveway", cooldown_seconds=0)
        path = saver.save(make_frame(), [], [])
        assert path is not None
        assert path.exists()
        assert path.suffix == ".jpg"

    def test_path_has_date_subdirectory_structure(self, tmp_output_dir):
        saver = ImageSaver(tmp_output_dir, "driveway", cooldown_seconds=0)
        path = saver.save(make_frame(), [], [])
        # Expected: tmp_output_dir/driveway/YYYY/MM/DD/timestamp.jpg
        relative = path.relative_to(tmp_output_dir)
        parts = relative.parts
        assert parts[0] == "driveway"
        assert len(parts[1]) == 4   # year
        assert len(parts[2]) == 2   # month zero-padded
        assert len(parts[3]) == 2   # day zero-padded
        assert parts[4].endswith(".jpg")

    def test_startup_prefix_on_filename(self, tmp_output_dir):
        saver = ImageSaver(tmp_output_dir, "driveway", cooldown_seconds=30)
        path = saver.save(make_frame(), [], [], prefix="startup_")
        assert path is not None
        assert path.name.startswith("startup_")

    def test_date_directories_created_automatically(self, tmp_output_dir):
        saver = ImageSaver(tmp_output_dir, "driveway", cooldown_seconds=0)
        path = saver.save(make_frame(), [], [])
        assert path.parent.exists()


class TestImageSaverCooldown:
    def test_first_save_succeeds(self, tmp_output_dir):
        saver = ImageSaver(tmp_output_dir, "driveway", cooldown_seconds=30)
        path = saver.save(make_frame(), [], [])
        assert path is not None

    def test_second_save_within_cooldown_returns_none(self, tmp_output_dir):
        saver = ImageSaver(tmp_output_dir, "driveway", cooldown_seconds=30)
        saver.save(make_frame(), [], [])
        result = saver.save(make_frame(), [], [])
        assert result is None

    def test_startup_save_bypasses_cooldown(self, tmp_output_dir):
        saver = ImageSaver(tmp_output_dir, "driveway", cooldown_seconds=30)
        saver.save(make_frame(), [], [])  # set last_save_time
        path = saver.save(make_frame(), [], [], prefix="startup_")
        assert path is not None

    def test_save_allowed_after_cooldown_expires(self, tmp_output_dir):
        saver = ImageSaver(tmp_output_dir, "driveway", cooldown_seconds=0)
        saver.save(make_frame(), [], [])
        time.sleep(0.01)
        second = saver.save(make_frame(), [], [])
        assert second is not None


class TestImageSaverAnnotation:
    def test_vehicle_bounding_box_drawn_red(self, tmp_output_dir):
        saver = ImageSaver(tmp_output_dir, "driveway", cooldown_seconds=0)
        vehicle = make_vehicle(x1=50, y1=50, x2=200, y2=200)
        frame = make_frame()
        annotated = saver._annotate(frame.copy(), [vehicle], [])
        # Red in BGR is (B=0, G=0, R=255). Rectangle drawn at (col=50, row=50).
        assert annotated[50, 50, 2] == 255  # R channel high
        assert annotated[50, 50, 1] == 0    # G channel zero
        assert annotated[50, 50, 0] == 0    # B channel zero

    def test_scan_region_drawn_green(self, tmp_output_dir):
        saver = ImageSaver(tmp_output_dir, "driveway", cooldown_seconds=0)
        region = ScanRegion(x=50, y=50, width=100, height=100)
        frame = make_frame()
        annotated = saver._annotate(frame.copy(), [], [region])
        # Green in BGR is (B=0, G=255, R=0). Rectangle drawn at (col=50, row=50).
        assert annotated[50, 50, 1] == 255  # G channel high
        assert annotated[50, 50, 2] == 0    # R channel zero

    def test_annotate_does_not_modify_original_frame(self, tmp_output_dir):
        saver = ImageSaver(tmp_output_dir, "driveway", cooldown_seconds=0)
        frame = make_frame()
        original = frame.copy()
        saver._annotate(frame, [], [ScanRegion(x=10, y=10, width=50, height=50)])
        np.testing.assert_array_equal(frame, original)

    def test_save_jpeg_at_85_quality(self, tmp_output_dir):
        saver = ImageSaver(tmp_output_dir, "driveway", cooldown_seconds=0)
        path = saver.save(make_frame(), [], [])
        assert path is not None
        # Verify it is a valid JPEG
        img = cv2.imread(str(path))
        assert img is not None
        assert img.shape[2] == 3
