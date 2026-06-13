import math
import numpy as np
import cv2
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config import ScanRegion, IgnoreRegion
from src.detector import Detector, Detection, TrackedVehicle


TEST_CASES = yaml.safe_load(
    (Path(__file__).parent / 'data' / 'test_cases.yaml').read_text()
)


def test_detector_constructs_without_threshold_params():
    with patch('src.detector.YOLO'):
        d = Detector(
            model_path='yolov8n.pt',
            vehicle_classes=['car'],
            detection_confidence=0.4,
            iou_threshold=0.5,
            stationary_seconds=3,
            target_fps=1,
            night_enhancement=True,
            scan_regions=[],
            ignore_regions=[],
        )
    assert d is not None


def make_bgr_from_hsv(saturation, brightness, height=480, width=640):
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[:, :] = [0, saturation, brightness]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def make_detector(**kwargs):
    defaults = dict(
        model_path='yolov8n.pt',
        vehicle_classes=['car', 'truck', 'bus'],
        detection_confidence=0.4,
        iou_threshold=0.5,
        stationary_seconds=3,
        target_fps=1,
        night_enhancement=True,
        scan_regions=[],
        ignore_regions=[],
    )
    defaults.update(kwargs)
    with patch('src.detector.YOLO'):
        return Detector(**defaults)


class TestEnhancement:
    def test_enhance_frame_calls_clahe_for_dark_color_frame(self, dim_color_frame):
        d = make_detector()
        with patch.object(d, '_apply_clahe', return_value=dim_color_frame) as clahe_mock:
            with patch.object(d, '_apply_ir_enhancement', return_value=dim_color_frame, create=True):
                d._enhance_frame(dim_color_frame)
        clahe_mock.assert_called_once()

    def test_enhance_frame_calls_ir_enhancement_for_dark_ir_frame(self):
        d = make_detector()
        ir_frame = make_bgr_from_hsv(saturation=0, brightness=50)
        with patch.object(d, '_apply_ir_enhancement', return_value=ir_frame, create=True) as ir_mock:
            with patch.object(d, '_apply_clahe', return_value=ir_frame):
                d._enhance_frame(ir_frame)
        ir_mock.assert_called_once()

    def test_enhance_frame_returns_frame_unchanged_when_bright(self):
        d = make_detector()
        bright_frame = make_bgr_from_hsv(saturation=100, brightness=200)
        with patch.object(d, '_apply_clahe') as clahe_mock:
            with patch.object(d, '_apply_ir_enhancement', create=True) as ir_mock:
                result = d._enhance_frame(bright_frame)
        clahe_mock.assert_not_called()
        ir_mock.assert_not_called()

    def test_apply_clahe_returns_same_shape(self, dim_color_frame):
        d = make_detector()
        result = d._apply_clahe(dim_color_frame)
        assert result.shape == dim_color_frame.shape
        assert result.dtype == np.uint8

    def test_apply_clahe_brightens_dark_frame(self, dim_color_frame):
        d = make_detector()
        result = d._apply_clahe(dim_color_frame)
        assert np.mean(result) > np.mean(dim_color_frame)

    def test_apply_clahe_does_not_modify_original(self, dim_color_frame):
        d = make_detector()
        original_copy = dim_color_frame.copy()
        d._apply_clahe(dim_color_frame)
        np.testing.assert_array_equal(dim_color_frame, original_copy)

    def test_night_enhancement_disabled_skips_enhance_frame(self, dim_color_frame):
        detector = make_detector(night_enhancement=False)
        with patch.object(detector, '_enhance_frame') as mock:
            with patch.object(detector, '_run_inference', return_value=[]):
                detector.process_frame(dim_color_frame)
        mock.assert_not_called()

    def test_night_enhancement_enabled_calls_enhance_frame(self, dim_color_frame):
        detector = make_detector(night_enhancement=True)
        with patch.object(detector, '_enhance_frame', return_value=dim_color_frame) as mock:
            with patch.object(detector, '_run_inference', return_value=[]):
                detector.process_frame(dim_color_frame)
        mock.assert_called_once()

    def test_apply_ir_enhancement_returns_same_shape(self):
        d = make_detector()
        ir_frame = make_bgr_from_hsv(saturation=0, brightness=50)
        result = d._apply_ir_enhancement(ir_frame)
        assert result.shape == ir_frame.shape
        assert result.dtype == np.uint8

    def test_apply_ir_enhancement_brightens_dark_ir_frame(self):
        d = make_detector()
        ir_frame = make_bgr_from_hsv(saturation=0, brightness=50)
        result = d._apply_ir_enhancement(ir_frame)
        assert np.mean(result) > np.mean(ir_frame)

    def test_apply_ir_enhancement_does_not_modify_original(self):
        d = make_detector()
        ir_frame = make_bgr_from_hsv(saturation=0, brightness=50)
        original_copy = ir_frame.copy()
        d._apply_ir_enhancement(ir_frame)
        np.testing.assert_array_equal(ir_frame, original_copy)


class TestComputeIou:
    def test_identical_boxes_return_1(self):
        d = make_detector()
        assert d._compute_iou((0, 0, 100, 100), (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_non_overlapping_boxes_return_0(self):
        d = make_detector()
        assert d._compute_iou((0, 0, 50, 50), (100, 100, 200, 200)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        d = make_detector()
        # box1: (0,0,100,100), box2: (50,0,150,100) → overlap = 50*100=5000
        # area1=10000, area2=10000, union=15000, iou=1/3
        result = d._compute_iou((0, 0, 100, 100), (50, 0, 150, 100))
        assert result == pytest.approx(1 / 3, rel=1e-3)

    def test_zero_area_box_returns_0(self):
        d = make_detector()
        assert d._compute_iou((0, 0, 0, 0), (0, 0, 100, 100)) == pytest.approx(0.0)


class TestUpdateTracker:
    def test_new_detection_starts_with_frames_1(self):
        d = make_detector()
        d._update_tracker([Detection(box=(0, 0, 100, 100), class_name='car', confidence=0.9)])
        assert len(d._tracked) == 1
        assert d._tracked[0].frames == 1

    def test_matched_vehicle_increments_frames(self):
        d = make_detector(iou_threshold=0.5)
        d._tracked = [TrackedVehicle(box=(0, 0, 100, 100), frames=5)]
        d._update_tracker([Detection(box=(0, 0, 100, 100), class_name='car', confidence=0.9)])
        assert len(d._tracked) == 1
        assert d._tracked[0].frames == 6

    def test_unmatched_vehicle_removed_from_tracker(self):
        d = make_detector()
        d._tracked = [TrackedVehicle(box=(0, 0, 100, 100), frames=5)]
        d._update_tracker([])
        assert len(d._tracked) == 0

    def test_low_iou_vehicle_not_matched(self):
        d = make_detector(iou_threshold=0.5)
        d._tracked = [TrackedVehicle(box=(0, 0, 100, 100), frames=3)]
        # Barely overlapping box → IoU well below 0.5
        d._update_tracker([Detection(box=(90, 90, 200, 200), class_name='car', confidence=0.9)])
        # New vehicle starts at frames=1, old vehicle dropped
        assert len(d._tracked) == 1
        assert d._tracked[0].frames == 1

    def test_two_detections_matched_to_two_tracked(self):
        d = make_detector()
        d._tracked = [
            TrackedVehicle(box=(0, 0, 100, 100), frames=2),
            TrackedVehicle(box=(200, 200, 300, 300), frames=4),
        ]
        d._update_tracker([
            Detection(box=(0, 0, 100, 100), class_name='car', confidence=0.9),
            Detection(box=(200, 200, 300, 300), class_name='truck', confidence=0.8),
        ])
        frames = sorted(v.frames for v in d._tracked)
        assert frames == [3, 5]


class TestStationaryCount:
    def test_vehicle_counted_after_required_frames(self):
        d = make_detector(stationary_seconds=3, target_fps=1)
        # required_frames = ceil(3 * 1) = 3
        detection = Detection(box=(0, 0, 100, 100), class_name='car', confidence=0.9)
        for _ in range(3):
            d._update_tracker([detection])
        stationary = [v for v in d._tracked if v.frames >= 3]
        assert len(stationary) == 1

    def test_vehicle_not_counted_before_required_frames(self):
        d = make_detector(stationary_seconds=3, target_fps=1)
        detection = Detection(box=(0, 0, 100, 100), class_name='car', confidence=0.9)
        for _ in range(2):
            d._update_tracker([detection])
        stationary = [v for v in d._tracked if v.frames >= 3]
        assert len(stationary) == 0

    def test_fractional_fps_rounds_up(self):
        d = make_detector(stationary_seconds=3, target_fps=2)
        # required_frames = ceil(3 * 2) = 6
        assert d._required_frames == 6

    def test_process_frame_returns_correct_count(self):
        d = make_detector(stationary_seconds=2, target_fps=1, night_enhancement=False)
        box = (0, 0, 100, 100)
        with patch.object(d, '_run_inference', return_value=[
            Detection(box=box, class_name='car', confidence=0.9)
        ]):
            count, vehicles = d.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
            assert count == 0  # frames=1, need 2

            count, vehicles = d.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
            assert count == 1
            assert len(vehicles) == 1
            assert vehicles[0].box == box


class TestScanRegions:
    def test_no_regions_includes_all_detections(self):
        d = make_detector(scan_regions=[])
        assert d._is_in_scan_regions((0, 0, 1920, 1080)) is True
        assert d._is_in_scan_regions((500, 500, 600, 600)) is True

    def test_box_overlapping_region_is_included(self):
        region = ScanRegion(x=100, y=100, width=200, height=200)
        d = make_detector(scan_regions=[region])
        # Box fully inside region
        assert d._is_in_scan_regions((110, 110, 150, 150)) is True

    def test_box_partially_overlapping_region_is_included(self):
        region = ScanRegion(x=100, y=100, width=200, height=200)
        d = make_detector(scan_regions=[region])
        # Box partially overlaps region (straddles left edge)
        assert d._is_in_scan_regions((50, 110, 150, 150)) is True

    def test_box_outside_all_regions_excluded(self):
        region = ScanRegion(x=100, y=100, width=200, height=200)
        d = make_detector(scan_regions=[region])
        # Completely outside
        assert d._is_in_scan_regions((400, 400, 500, 500)) is False

    def test_box_in_any_of_multiple_regions_is_included(self):
        regions = [
            ScanRegion(x=0, y=0, width=100, height=100),
            ScanRegion(x=500, y=500, width=100, height=100),
        ]
        d = make_detector(scan_regions=regions)
        assert d._is_in_scan_regions((510, 510, 560, 560)) is True


class TestIgnoreRegions:
    def test_no_ignore_regions_does_not_suppress(self):
        d = make_detector(ignore_regions=[])
        assert d._is_in_ignore_regions((0, 0, 100, 100)) is False

    def test_vehicle_fully_inside_ignore_region_is_suppressed(self):
        region = IgnoreRegion(x=0, y=0, width=200, height=200)
        d = make_detector(ignore_regions=[region])
        # box entirely inside: 100% overlap
        assert d._is_in_ignore_regions((10, 10, 190, 190)) is True

    def test_vehicle_exactly_95_percent_inside_is_suppressed(self):
        # region: x=0..200, y=0..200 (area 40000)
        # box: x=0..200, y=0..200 (area 40000)
        # intersection: 40000 => coverage 100% >= 0.95 → suppressed
        region = IgnoreRegion(x=0, y=0, width=200, height=200)
        d = make_detector(ignore_regions=[region])
        assert d._is_in_ignore_regions((0, 0, 200, 200)) is True

    def test_vehicle_94_percent_inside_is_not_suppressed(self):
        # region: x=0..100, y=0..100 (area 10000)
        # box: x=0..100, y=0..106 (area 10600)
        # intersection: x=0..100, y=0..100 => 10000
        # coverage: 10000 / 10600 = 0.943... < 0.95 → not suppressed
        region = IgnoreRegion(x=0, y=0, width=100, height=100)
        d = make_detector(ignore_regions=[region])
        assert d._is_in_ignore_regions((0, 0, 100, 106)) is False

    def test_vehicle_95_percent_inside_one_of_multiple_regions_is_suppressed(self):
        regions = [
            IgnoreRegion(x=500, y=500, width=100, height=100),
            IgnoreRegion(x=0, y=0, width=200, height=200),
        ]
        d = make_detector(ignore_regions=regions)
        assert d._is_in_ignore_regions((10, 10, 190, 190)) is True

    def test_vehicle_outside_all_ignore_regions_is_not_suppressed(self):
        region = IgnoreRegion(x=500, y=500, width=100, height=100)
        d = make_detector(ignore_regions=[region])
        assert d._is_in_ignore_regions((0, 0, 100, 100)) is False


class TestRealImageDetection:
    @pytest.mark.parametrize("case", TEST_CASES['detection_cases'], ids=lambda c: c['name'])
    def test_detects_expected_vehicle_count(self, case):
        image_path = Path(__file__).parent / 'data' / case['image']
        frame = cv2.imread(str(image_path))
        assert frame is not None, f"Could not load test image: {image_path}"

        scan_regions = [
            ScanRegion(x=r['x'], y=r['y'], width=r['width'], height=r['height'])
            for r in case.get('scan_regions', [])
        ]
        detector = Detector(
            model_path='yolov8l.pt',
            vehicle_classes=case.get('vehicle_classes', ['car', 'truck', 'bus']),
            detection_confidence=case.get('detection_confidence', 0.4),
            iou_threshold=0.5,
            stationary_seconds=1,
            target_fps=1,
            night_enhancement=case.get('night_enhancement', False),
            scan_regions=scan_regions,
            ignore_regions=[],
        )

        count, vehicles = detector.process_frame(frame)
        assert count == case['expected_count'], (
            f"Expected {case['expected_count']} vehicles, got {count}: "
            + ", ".join(f"box=({v.box[0]:.0f},{v.box[1]:.0f},{v.box[2]:.0f},{v.box[3]:.0f}) frames={v.frames}" for v in vehicles)
        )


class TestRunInference:
    def test_filters_by_vehicle_class(self):
        d = make_detector(vehicle_classes=['car'], detection_confidence=0.4)

        mock_box_car = MagicMock()
        mock_box_car.xyxy = [np.array([0, 0, 100, 100], dtype=np.float32)]
        mock_box_car.conf = [np.float32(0.9)]
        mock_box_car.cls = [np.float32(2)]  # class_id=2

        mock_box_person = MagicMock()
        mock_box_person.xyxy = [np.array([200, 200, 300, 300], dtype=np.float32)]
        mock_box_person.conf = [np.float32(0.9)]
        mock_box_person.cls = [np.float32(0)]  # class_id=0

        mock_result = MagicMock()
        mock_result.boxes = [mock_box_car, mock_box_person]
        d._model.return_value = [mock_result]
        d._model.names = {0: 'person', 2: 'car'}

        detections = d._run_inference(np.zeros((480, 640, 3), dtype=np.uint8))

        assert len(detections) == 1
        assert detections[0].class_name == 'car'

    def test_suppresses_overlapping_cross_class_detections(self):
        # Same vehicle detected as both car and truck — should count as one
        d = make_detector(vehicle_classes=['car', 'truck'], detection_confidence=0.4)

        def make_box(x1, y1, x2, y2, conf, cls_id):
            b = MagicMock()
            b.xyxy = [np.array([x1, y1, x2, y2], dtype=np.float32)]
            b.conf = [np.float32(conf)]
            b.cls = [np.float32(cls_id)]
            return b

        mock_result = MagicMock()
        mock_result.boxes = [
            make_box(0, 0, 100, 100, 0.9, 2),   # car
            make_box(1, 0, 101, 100, 0.8, 7),   # truck, nearly same box
        ]
        d._model.return_value = [mock_result]
        d._model.names = {2: 'car', 7: 'truck'}

        detections = d._run_inference(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(detections) == 1
        assert detections[0].class_name == 'car'  # higher confidence wins

    def test_filters_by_confidence(self):
        d = make_detector(vehicle_classes=['car'], detection_confidence=0.7)

        mock_box = MagicMock()
        mock_box.xyxy = [np.array([0, 0, 100, 100], dtype=np.float32)]
        mock_box.conf = [np.float32(0.5)]  # below threshold
        mock_box.cls = [np.float32(2)]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        d._model.return_value = [mock_result]
        d._model.names = {2: 'car'}

        detections = d._run_inference(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(detections) == 0
