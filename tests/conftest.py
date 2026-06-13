"""Shared pytest fixtures: synthetic frames (bright color, dim color, IR) for detector tests."""
import numpy as np
import cv2
import pytest


def _make_bgr_from_hsv(hue, saturation, value, height=480, width=640):
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[:, :] = [hue, saturation, value]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@pytest.fixture
def bright_color_frame():
    """H=0, S=120, V=200 — brightness 200, saturation 120. CLAHE should NOT apply."""
    return _make_bgr_from_hsv(0, 120, 200)


@pytest.fixture
def dim_color_frame():
    """H=0, S=80, V=50 — brightness 50 (<80), saturation 80 (>=30). CLAHE SHOULD apply."""
    return _make_bgr_from_hsv(0, 80, 50)


@pytest.fixture
def ir_frame():
    """H=0, S=10, V=50 — brightness 50 (<80), saturation 10 (<30). IR mode, CLAHE should NOT apply."""
    return _make_bgr_from_hsv(0, 10, 50)


@pytest.fixture
def tmp_output_dir(tmp_path):
    d = tmp_path / "output"
    d.mkdir()
    return d
