import locus
import numpy as np


def test_soft_decoding_instantiation():
    """Verify that Detector can be instantiated with DecodeMode.Soft."""
    # This ensures the Enum is exported and accepted
    detector = locus.Detector(decode_mode=locus.DecodeMode.Soft)
    assert detector is not None


def test_soft_decoding_e2e():
    """Verify soft decoding detects nothing on empty/noisy image (smoke test)."""
    detector = locus.Detector(decode_mode=locus.DecodeMode.Soft)

    # Create a dummy image
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:80, 20:80] = 200  # White
    img[30:70, 30:70] = 50  # Grey/Black

    # Run detection
    dets = detector.detect(img)
    assert isinstance(dets, list)

    # Verify stats works
    _, stats = detector.detect_with_stats(img)
    assert stats.total_ms >= 0


if __name__ == "__main__":
    test_soft_decoding_instantiation()
    test_soft_decoding_e2e()
    print("Soft decoding tests passed")
