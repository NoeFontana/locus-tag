import cv2
import locus
import numpy as np
import pytest


def test_zero_copy_ingestion():
    """Verify that the detector handles various numpy layouts correctly."""
    detector = locus.Detector()

    # 1. Standard C-contiguous array
    img = np.zeros((100, 100), dtype=np.uint8)
    # Draw a simple white box
    img[20:80, 20:80] = 255
    img[30:70, 30:70] = 0

    result = detector.detect(img)
    assert isinstance(result, locus.Result)

    # 2. Strided array (padding)
    full_img = np.zeros((100, 120), dtype=np.uint8)
    img_strided = full_img[:, :100]
    assert img_strided.strides[0] == 120
    assert img_strided.strides[1] == 1

    result = detector.detect(img_strided)
    assert isinstance(result, locus.Result)

    # 3. Non-contiguous slice (step > 1)
    img_sliced = img[:, ::2]
    assert not img_sliced.flags["C_CONTIGUOUS"]

    with pytest.raises(ValueError, match="Array must be C-contiguous"):
        detector.detect(img_sliced)


def test_detector_api():
    """Test the high-level Detector class and vectorized results."""
    # Create an ArUco 4x4_50 tag image
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag_img = cv2.aruco.generateImageMarker(dictionary, 0, 100)

    canvas = np.ones((400, 400), dtype=np.uint8) * 128
    canvas[150:250, 150:250] = tag_img

    # 1. Default config (should ignore ArUco by default)
    detector = locus.Detector()
    result = detector.detect(canvas)
    assert len(result) == 0

    # 2. Specific family selection via __init__
    detector_aruco = locus.Detector(families=[locus.TagFamily.ArUco4x4_50])
    result = detector_aruco.detect(canvas)
    assert len(result) == 1
    assert result.ids[0] == 0
    assert result.centers.shape == (1, 2)
    assert result.corners.shape == (1, 4, 2)

    # 3. Test to_list conversion
    dets = result.to_list()
    assert len(dets) == 1
    assert dets[0]["id"] == 0


def test_config_object():
    """Test that DetectorConfig correctly validates parameters."""
    from locus import DetectorConfig

    cfg = DetectorConfig(threshold_tile_size=16, quad_min_area=500)
    assert cfg.threshold_tile_size == 16
    assert cfg.quad_min_area == 500


def test_soft_decoding():
    """Verify that Detector can be instantiated and run with DecodeMode.Soft."""
    detector = locus.Detector(decode_mode=locus.DecodeMode.Soft)
    assert detector is not None

    # Create a dummy image
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:80, 20:80] = 255
    img[30:70, 30:70] = 0

    # Run detection
    result = detector.detect(img)
    assert isinstance(result, locus.Result)
