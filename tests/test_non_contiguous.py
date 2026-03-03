import locus
import numpy as np


def test_non_contiguous_ingestion():
    """Verify that the detector handles non-contiguous arrays with auto-conversion."""
    # 1. Standard C-contiguous array
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:80, 20:80] = 255

    detector = locus.Detector()
    detections = detector.detect(img)
    assert isinstance(detections, list)

    # 2. Non-contiguous slice (step > 1)
    img_sliced = img[:, ::2]
    assert not img_sliced.flags["C_CONTIGUOUS"]

    # This used to raise ValueError. Now it should work (with a warning to stderr).
    print("\nTesting non-contiguous slice (should see warning):")
    detections = detector.detect(img_sliced)
    assert isinstance(detections, list)

    # 3. F-contiguous array
    img_f = np.asfortranarray(img)
    assert img_f.flags["F_CONTIGUOUS"]
    assert not img_f.flags["C_CONTIGUOUS"]

    print("\nTesting F-contiguous array (should see warning):")
    detections = detector.detect(img_f)
    assert isinstance(detections, list)


if __name__ == "__main__":
    test_non_contiguous_ingestion()
    print("\nNon-contiguous ingestion test passed!")
