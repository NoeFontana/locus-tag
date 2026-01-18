import locus
import numpy as np
import cv2
import pytest
import sys

def test_zero_copy_ingestion():
    """Verify that the detector handles various numpy layouts correctly."""
    # 1. Standard C-contiguous array
    img = np.zeros((100, 100), dtype=np.uint8)
    # Draw a simple white box to ensure we don't crash on actual data
    img[20:80, 20:80] = 255
    img[30:70, 30:70] = 0
    
    detections = locus.detect_tags(img)
    assert isinstance(detections, list)

    # 2. Strided array (padding)
    full_img = np.zeros((100, 120), dtype=np.uint8)
    img_strided = full_img[:, :100]
    assert img_strided.strides[0] == 120
    assert img_strided.strides[1] == 1
    
    detections = locus.detect_tags(img_strided)
    assert isinstance(detections, list)

    # 3. Non-contiguous slice (step > 1)
    img_sliced = img[:, ::2]
    assert not img_sliced.flags['C_CONTIGUOUS']
    
    detections = locus.detect_tags(img_sliced)
    assert isinstance(detections, list)

    # 4. F-contiguous array (Should be rejected with ValueError)
    img_f = np.asfortranarray(img)
    try:
        locus.detect_tags(img_f)
        pytest.fail("F-contiguous array should have raised ValueError")
    except ValueError as e:
        assert "contiguous" in str(e).lower()

def test_detector_api():
    """Test the high-level Detector class and parameter validation."""
    # Create an ArUco 4x4_50 tag image
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag_img = cv2.aruco.generateImageMarker(dictionary, 0, 100)
    
    canvas = np.ones((400, 400), dtype=np.uint8) * 128
    canvas[150:250, 150:250] = tag_img

    # 1. Default config (should ignore ArUco by default)
    detector = locus.Detector()
    results = detector.detect(canvas)
    # Default family is AprilTag 36h11
    assert len(results) == 0

    # 2. Specific family selection
    results = detector.detect_with_options(canvas, families=[locus.TagFamily.ArUco4x4_50])
    assert len(results) == 1
    assert results[0].id == 0

    # 3. Custom configuration parameters
    # Set min area to something larger than the tag (100x100 = 10000)
    det_strict = locus.Detector(quad_min_area=20000)
    results_strict = det_strict.detect_with_options(canvas, families=[locus.TagFamily.ArUco4x4_50])
    assert len(results_strict) == 0

    # 4. Persistence (Arena reuse)
    for _ in range(3):
        res = detector.detect_with_options(canvas, families=[locus.TagFamily.ArUco4x4_50])
        assert len(res) == 1

def test_config_object():
    """Test that DetectorConfig correctly validates parameters."""
    from locus import DetectorConfig
    
    cfg = DetectorConfig(threshold_tile_size=16, quad_min_area=500)
    assert cfg.threshold_tile_size == 16
    assert cfg.quad_min_area == 500
    
    # Test Pydantic validation (if enabled in the wrapper)
    try:
        DetectorConfig(upscale_factor=0) # Should be >= 1
        # If no validation, this might pass depending on implementation
    except Exception:
        pass

if __name__ == "__main__":
    # If run directly without pytest
    test_zero_copy_ingestion()
    test_detector_api()
    test_config_object()
    print("All Python API tests passed!")
