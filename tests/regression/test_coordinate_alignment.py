import cv2
import locus
import numpy as np
import pytest


def generate_synthetic_tag(tag_size=100, canvas_size=320, start=110):
    img = np.zeros((canvas_size, canvas_size), dtype=np.uint8) + 255
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag_img = cv2.aruco.generateImageMarker(dictionary, 0, tag_size)

    # Place tag: pixel 110 is the first black pixel.
    # Physical boundary is at 110.0 in the OpenCV/Locus + 0.5 convention.
    img[start : start + tag_size, start : start + tag_size] = tag_img

    ts = tag_size
    # In OpenCV convention, if the first black pixel is at index 110,
    # the top-left boundary of that pixel is at 110.0.
    corners = np.array(
        [[start, start], [start + ts, start], [start + ts, start + ts], [start, start + ts]],
        dtype=np.float32,
    )

    return img, corners


def test_decimation_mapping():
    """Verify that decimation=2 doesn't introduce a coordinate shift."""
    tag_size = 100
    canvas_size = 400
    start = 140
    img, gt_corners = generate_synthetic_tag(tag_size, canvas_size, start)

    # Use decimation=2
    cfg = locus.DetectorConfig.from_profile("standard")
    cfg_dict = cfg.model_dump()
    cfg_dict["decoder"]["refinement_mode"] = "Edge"
    detector = locus.Detector(
        config=locus.DetectorConfig.model_validate(cfg_dict),
        families=[locus.TagFamily.AprilTag36h11],
        decimation=2,
    )

    batch = detector.detect(img)
    assert len(batch) == 1

    det_corners = batch.corners[0]
    # Error should be very small (< 0.1px)
    rmse = np.sqrt(np.mean(np.sum((det_corners - gt_corners) ** 2, axis=1)))
    print(f"Decimation=2 RMSE: {rmse:.4f}px")
    assert rmse < 0.1, f"Decimation mapping bug: RMSE {rmse:.4f}px is too high"


def test_no_refine_expansion():
    """Verify that expansion=0.5px correctly aligns the geometric quad with boundaries."""
    tag_size = 100
    canvas_size = 400
    start = 140
    img, gt_corners = generate_synthetic_tag(tag_size, canvas_size, start)

    # Use no refinement
    cfg = locus.DetectorConfig.from_profile("standard")
    cfg_dict = cfg.model_dump()
    cfg_dict["decoder"]["refinement_mode"] = "None"
    detector = locus.Detector(
        config=locus.DetectorConfig.model_validate(cfg_dict),
        families=[locus.TagFamily.AprilTag36h11],
    )

    batch = detector.detect(img)
    assert len(batch) == 1

    det_corners = batch.corners[0]
    # Without expansion, the error would be ~0.5px (RMSE ~0.7px).
    # With 0.5px expansion, it should be very close to the integer boundary.
    rmse = np.sqrt(np.mean(np.sum((det_corners - gt_corners) ** 2, axis=1)))
    print(f"No-Refine RMSE: {rmse:.4f}px")
    # We expect integer alignment now (RMSE < 0.2 if the fit is good)
    assert rmse < 0.25, (
        f"Geometric quad bias: RMSE {rmse:.4f}px is too high for unrefined detection"
    )


if __name__ == "__main__":
    pytest.main([__file__])
