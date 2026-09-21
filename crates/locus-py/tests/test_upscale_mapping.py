"""Regression: ``quad.upscale_factor > 1`` must return corners in ORIGINAL pixels.

Previously the corners were left in the upscaled frame (e.g. ~2x the true
coordinates at ``upscale_factor=2``). Rendering uses OpenCV's ArUco generator,
which is part of the ``bench`` dependency group, so the test is skipped when
``cv2`` is unavailable.
"""

from __future__ import annotations

import locus
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

_CANVAS = 640
_TAG_PX = 100
_ORIGIN = (_CANVAS - _TAG_PX) // 2  # tag outer border spans [270, 370)


def _scene() -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 0, _TAG_PX)
    img = np.full((_CANVAS, _CANVAS), 255, dtype=np.uint8)
    img[_ORIGIN : _ORIGIN + _TAG_PX, _ORIGIN : _ORIGIN + _TAG_PX] = marker
    return img


def _detect_corners(img: np.ndarray, upscale_factor: int) -> np.ndarray:
    cfg = locus.DetectorConfig.from_profile("standard")
    cfg.quad.upscale_factor = upscale_factor
    det = locus.Detector(config=cfg)
    batch = det.detect(img)
    assert len(batch.ids) == 1
    return np.asarray(batch.corners[0], dtype=np.float64)


@pytest.mark.parametrize("upscale_factor", [2, 3])
def test_upscale_corners_in_original_frame(upscale_factor: int) -> None:
    img = _scene()
    base = _detect_corners(img, 1)
    up = _detect_corners(img, upscale_factor)
    # A frame bug shifts corners by hundreds of pixels; the residual here is
    # the sub-pixel refinement difference between processing grids.
    assert np.abs(up - base).max() < 0.25
    # And they sit on the rendered tag, not at ~upscale_factor x its position.
    assert up.min() > _ORIGIN - 2
    assert up.max() < _ORIGIN + _TAG_PX + 2
