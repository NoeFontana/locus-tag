import cv2
import locus
import numpy as np
import pytest


def test_non_contiguous_ingestion():
    """Verify that the detector raises ValueError for non-contiguous arrays (Zero-Copy Enforcement)."""
    # 1. Standard C-contiguous array (should work)
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:80, 20:80] = 255

    detector = locus.Detector()
    batch = detector.detect(img)
    assert isinstance(batch, locus.DetectionBatch)

    # 2. Sliced array (non-contiguous)
    img_non_contiguous = img[:, ::2]
    assert not img_non_contiguous.flags["C_CONTIGUOUS"]

    with pytest.raises(ValueError, match="C-contiguous"):
        detector.detect(img_non_contiguous)

    # 3. Fortran-layout array (non-contiguous)
    img_fortran = np.asfortranarray(img)
    assert not img_fortran.flags["C_CONTIGUOUS"]

    with pytest.raises(ValueError, match="C-contiguous"):
        detector.detect(img_fortran)
