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


def test_vectorized_poses():
    """Verify that 3D poses are returned in the compact (N, 7) format."""
    # Create an ArUco tag
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag_img = cv2.aruco.generateImageMarker(dictionary, 0, 100)
    canvas = np.ones((400, 400), dtype=np.uint8) * 128
    canvas[150:250, 150:250] = tag_img

    detector = locus.Detector(families=[locus.TagFamily.ArUco4x4_50])

    # Request pose estimation
    intrinsics = locus.CameraIntrinsics(fx=800.0, fy=800.0, cx=200.0, cy=200.0)
    batch = detector.detect(canvas, intrinsics=intrinsics, tag_size=0.10)

    assert len(batch) == 1
    assert batch.poses is not None
    assert batch.poses.shape == (1, 7)
    assert batch.poses.dtype == np.float32

    # [tx, ty, tz, qx, qy, qz, qw]. The 100 px tag is centered exactly on the
    # principal point (200, 200), facing the camera front-on, so this is a fully
    # determined pose — assert the actual values, not just a sign:
    #   * tx, ty ~ 0        (tag center == principal point)
    #   * tz = fx * tag_size / tag_px = 800 * 0.10 / 100 = 0.80 m
    #   * quaternion ~ identity (frontal, unrotated tag)
    tx, ty, tz, qx, qy, qz, qw = (float(v) for v in batch.poses[0])
    assert abs(tx) < 1e-2, f"tx {tx} should be ~0 (tag on principal point)"
    assert abs(ty) < 1e-2, f"ty {ty} should be ~0 (tag on principal point)"
    assert abs(tz - 0.80) < 1e-2, f"tz {tz} should be ~0.80 m (fx*size/px)"
    # Frontal tag: rotation is near-identity, so |qw| ~ 1 and the vector part ~ 0.
    assert abs(qw) > 0.99, f"qw {qw} should be ~1 for a frontal tag"
    assert abs(qx) < 0.05 and abs(qy) < 0.05 and abs(qz) < 0.05, (
        f"quaternion vector part ({qx}, {qy}, {qz}) should be ~0 for a frontal tag"
    )


def test_invalid_input_contiguity():
    """Check that non-C-contiguous arrays raise ValueError."""
    detector = locus.Detector()
    img = np.zeros((100, 100), dtype=np.uint8)
    img_sliced = img[:, ::2]  # Non-contiguous

    with pytest.raises(ValueError, match="C-contiguous"):
        detector.detect(img_sliced)


def test_invalid_input_dtype():
    """Check that non-uint8 arrays raise ValueError."""
    detector = locus.Detector()
    img = np.zeros((100, 100), dtype=np.float32)

    with pytest.raises(ValueError, match="uint8"):
        detector.detect(img)
