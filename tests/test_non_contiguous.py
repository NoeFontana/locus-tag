import locus
import numpy as np
import pytest


def test_non_contiguous_ingestion():
    """Verify that the detector raises ValueError for non-contiguous arrays (Zero-Copy Enforcement)."""
    # 1. Padded (H, W) view from a (H, W+3) parent — passes the FFI SIMD-padding gate.
    parent = np.zeros((100, 103), dtype=np.uint8)
    img = parent[:, :100]
    img[20:80, 20:80] = 255

    detector = locus.Detector()
    batch = detector.detect(img)
    assert isinstance(batch, locus.DetectionBatch)

    # 2. Sliced array (non-contiguous, stride_x != 1)
    img_non_contiguous = img[:, ::2]
    assert not img_non_contiguous.flags["C_CONTIGUOUS"]

    with pytest.raises(ValueError, match="C-contiguous"):
        detector.detect(img_non_contiguous)

    # 3. Fortran-layout array (non-contiguous, stride_x != 1)
    img_fortran = np.asfortranarray(img)
    assert not img_fortran.flags["C_CONTIGUOUS"]

    with pytest.raises(ValueError, match="C-contiguous"):
        detector.detect(img_fortran)
