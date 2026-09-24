"""EuRoC MAV calibration dataset regression tests (Python API).

Exercises detection of AprilTag 36h11 markers on real grayscale frames
from the EuRoC cam_april calibration sequence (752×480, MT9V034 sensor).

Requirements:
  - EuRoC cam_april dataset: ``bash scripts/fetch_euroc_calibration.sh``
  - Env var ``LOCUS_EUROC_DATASET_DIR`` or auto-discovered ``tests/data/euroc``
"""

import os
from pathlib import Path

import numpy as np
import pytest

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # noqa: N806

# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

_EUROC_ENV = os.environ.get("LOCUS_EUROC_DATASET_DIR", "")
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "tests" / "data" / "euroc"
EUROC_ROOT = Path(_EUROC_ENV) if _EUROC_ENV else _DEFAULT_ROOT
CAM0_DIR = EUROC_ROOT / "cam_april" / "mav0" / "cam0" / "data"

_DATASET_AVAILABLE = CAM0_DIR.is_dir()
_PILLOW_AVAILABLE = Image is not None

pytestmark = [
    pytest.mark.skipif(not _DATASET_AVAILABLE, reason="EuRoC dataset not available"),
    pytest.mark.skipif(not _PILLOW_AVAILABLE, reason="Pillow not installed"),
]

# ---------------------------------------------------------------------------
# Constants (EuRoC cam_april AprilGrid — 6×6 tag36h11)
# ---------------------------------------------------------------------------

EUROC_GRID_TAGS = 36  # 6×6 grid, IDs 0–35
SAMPLE_STRIDE = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_images():
    """Return sorted list of cam0 PNG paths."""
    return sorted(CAM0_DIR.glob("*.png"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_euroc_detection_recall():
    """At least 50 % of sampled frames should detect ≥ 20 of the 36 grid tags."""
    import locus  # noqa: PLC0415 — deferred import so skip works without wheel

    detector = locus.Detector()
    images = _load_images()
    sampled = images[::SAMPLE_STRIDE]
    assert len(sampled) > 0, "No images found in cam0/data/"

    good_frames = 0
    for img_path in sampled:
        img = np.asarray(Image.open(img_path), dtype=np.uint8)  # type: ignore[union-attr]
        # Ensure C-contiguous grayscale
        if img.ndim == 3:
            img = img[:, :, 0].copy()
        img = np.ascontiguousarray(img)
        results = detector.detect(img)
        if len(results.ids) >= 20:
            good_frames += 1

    recall = good_frames / len(sampled)
    assert recall >= 0.5, (
        f"Detection recall too low: {recall:.2%} "
        f"({good_frames}/{len(sampled)} frames with ≥20 tags)"
    )


def test_euroc_no_false_positives():
    """Verify that detected tag IDs fall within the expected 0–35 range."""
    import locus  # noqa: PLC0415

    detector = locus.Detector()
    images = _load_images()
    sampled = images[::SAMPLE_STRIDE]

    for img_path in sampled:
        img = np.asarray(Image.open(img_path), dtype=np.uint8)  # type: ignore[union-attr]
        if img.ndim == 3:
            img = img[:, :, 0].copy()
        img = np.ascontiguousarray(img)
        results = detector.detect(img)
        for tag_id in results.ids:
            assert 0 <= tag_id < EUROC_GRID_TAGS, (
                f"{img_path.name}: unexpected tag ID {tag_id} (expected 0–35)"
            )
