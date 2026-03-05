import cv2
import locus
import numpy as np
import pytest

def test_return_type():
    """Verify that the detector returns a DetectionBatch dataclass."""
    detector = locus.Detector()
    img = np.zeros((100, 100), dtype=np.uint8)
    
    batch = detector.detect(img)
    assert isinstance(batch, locus.DetectionBatch)

def test_array_shapes():
    """Test that returned arrays have the correct dimensions and types."""
    # Create an ArUco tag
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag_img = cv2.aruco.generateImageMarker(dictionary, 0, 100)
    canvas = np.ones((400, 400), dtype=np.uint8) * 128
    canvas[150:250, 150:250] = tag_img

    detector = locus.Detector(families=[locus.TagFamily.ArUco4x4_50])
    batch = detector.detect(canvas)

    assert len(batch) == 1
    assert batch.ids.shape == (1,)
    assert batch.ids.dtype == np.int32
    
    assert batch.corners.shape == (1, 4, 2)
    assert batch.corners.dtype == np.float32
    
    assert batch.error_rates.shape == (1,)
    assert batch.error_rates.dtype == np.float32

def test_optional_poses():
    """Verify that poses are None when not requested."""
    detector = locus.Detector()
    img = np.zeros((100, 100), dtype=np.uint8)
    batch = detector.detect(img)
    
    assert batch.poses is None

def test_invalid_input_contiguity():
    """Check that non-C-contiguous arrays raise ValueError."""
    detector = locus.Detector()
    img = np.zeros((100, 100), dtype=np.uint8)
    img_sliced = img[:, ::2] # Non-contiguous
    
    with pytest.raises(ValueError, match="C-contiguous"):
        detector.detect(img_sliced)

def test_invalid_input_dtype():
    """Check that non-uint8 arrays raise ValueError."""
    detector = locus.Detector()
    img = np.zeros((100, 100), dtype=np.float32)
    
    with pytest.raises(ValueError, match="uint8"):
        detector.detect(img)
