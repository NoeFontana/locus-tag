import locus
import numpy as np
import pytest


def test_non_contiguous_ingestion():
    """Verify that the detector raises ValueError for non-contiguous arrays (Zero-Copy Enforcement)."""
    # 1. Standard C-contiguous array (should work)
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:80, 20:80] = 255

    detector = locus.Detector()
    detections = detector.detect(img)
    assert isinstance(detections, list)

    # 2. Non-contiguous slice (step > 1) - Should raise ValueError
    img_sliced = img[:, ::2]
    assert not img_sliced.flags["C_CONTIGUOUS"]

    print("\nTesting non-contiguous slice (should raise ValueError):")
    with pytest.raises(ValueError, match="Array must be C-contiguous"):
        detector.detect(img_sliced)

    # 3. F-contiguous array - Should raise ValueError
    img_f = np.asfortranarray(img)
    assert img_f.flags["F_CONTIGUOUS"]
    assert not img_f.flags["C_CONTIGUOUS"]

    print("\nTesting F-contiguous array (should raise ValueError):")
    with pytest.raises(ValueError, match="Array must be C-contiguous"):
        detector.detect(img_f)


def test_legacy_detect_non_contiguous():
    """Verify legacy functions also raise ValueError."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img_nc = img[:, ::2]

    with pytest.raises(ValueError, match="Array must be C-contiguous"):
        locus.detect_tags(img_nc)

    with pytest.raises(ValueError, match="Array must be C-contiguous"):
        locus.detect_tags_with_stats(img_nc)


if __name__ == "__main__":
    # If run as a script, use pytest to run itself
    pytest.main([__file__])
