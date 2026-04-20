"""End-to-end JSON → Pydantic → Rust → Python roundtrip.

Complements ``test_profile_values.py`` (pure-Pydantic schema tests) by
exercising the full stack — each shipped profile is parsed, shipped through
the PyO3 ``_create_detector_from_config`` entrypoint, and re-read via
``Detector.config()``. Drift between any two layers fails this suite.
"""

from __future__ import annotations

import pytest

import locus

_SHIPPED = ["standard", "grid", "high_accuracy"]


@pytest.mark.parametrize("profile", _SHIPPED)
def test_profile_builds_detector(profile: str) -> None:
    det = locus.Detector(profile=profile)
    # Smoke-detect a blank frame: the detector is constructible and runnable.
    import numpy as np

    batch = det.detect(np.zeros((64, 64), dtype=np.uint8))
    assert len(batch) == 0


@pytest.mark.parametrize("profile", _SHIPPED)
def test_config_roundtrip_matches_profile(profile: str) -> None:
    """``Detector(profile).config()`` must match the JSON-loaded config.

    A handful of fields are f32 on the Rust side and widen to f64 at the FFI
    boundary, so numeric fields are compared with a relative tolerance that
    absorbs a single-precision rounding trip without hiding real drift.
    """
    expected_flat = locus.DetectorConfig.from_profile(profile)._to_flat_ffi_dict()
    actual_flat = locus.Detector(profile=profile).config()._to_flat_ffi_dict()

    assert set(expected_flat) == set(actual_flat)
    for key, exp in expected_flat.items():
        got = actual_flat[key]
        if isinstance(exp, float):
            assert got == pytest.approx(exp, rel=1e-6, abs=1e-7), f"{profile}.{key}"
        else:
            assert got == exp, f"{profile}.{key}: {got!r} != {exp!r}"


def test_profile_and_config_mutually_exclusive() -> None:
    cfg = locus.DetectorConfig.from_profile("standard")
    with pytest.raises(ValueError, match="Pass either"):
        locus.Detector(profile="standard", config=cfg)


def test_profile_default_is_standard() -> None:
    left = locus.Detector().config()._to_flat_ffi_dict()
    right = locus.Detector(profile="standard").config()._to_flat_ffi_dict()
    assert set(left) == set(right)
    for key, exp in right.items():
        got = left[key]
        if isinstance(exp, float):
            assert got == pytest.approx(exp, rel=1e-6, abs=1e-7), key
        else:
            assert got == exp, key
