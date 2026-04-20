"""End-to-end JSON → Pydantic → Rust → Python roundtrip.

Complements ``test_profile_values.py`` (pure-Pydantic schema tests) by
exercising the full stack — each shipped profile is parsed, shipped through
the PyO3 ``_create_detector_from_config`` entrypoint, and re-read via
``Detector.config()``. Drift between any two layers fails this suite.
"""

from __future__ import annotations

import pytest

import locus
from locus.locus import PyDetectorConfig

_SHIPPED = ["standard", "grid", "high_accuracy"]


def _flat(cfg: PyDetectorConfig) -> dict:
    return {name: getattr(cfg, name) for name in dir(cfg) if not name.startswith("_")}


def _assert_configs_equal(actual: PyDetectorConfig, expected: PyDetectorConfig, label: str) -> None:
    # f32 fields round-trip through f64 at the FFI boundary; rel/abs tolerance
    # absorbs single-precision rounding without hiding real drift.
    a, e = _flat(actual), _flat(expected)
    assert set(a) == set(e)
    for key, exp in e.items():
        got = a[key]
        if isinstance(exp, float):
            assert got == pytest.approx(exp, rel=1e-6, abs=1e-7), f"{label}.{key}"
        else:
            assert got == exp, f"{label}.{key}: {got!r} != {exp!r}"


@pytest.mark.parametrize("profile", _SHIPPED)
def test_profile_builds_detector(profile: str) -> None:
    import numpy as np

    det = locus.Detector(profile=profile)
    batch = det.detect(np.zeros((64, 64), dtype=np.uint8))
    assert len(batch) == 0


@pytest.mark.parametrize("profile", _SHIPPED)
def test_config_roundtrip_matches_profile(profile: str) -> None:
    expected = locus.DetectorConfig.from_profile(profile)._to_ffi_config()
    actual = locus.Detector(profile=profile)._inner.config()
    _assert_configs_equal(actual, expected, profile)


def test_profile_and_config_mutually_exclusive() -> None:
    cfg = locus.DetectorConfig.from_profile("standard")
    with pytest.raises(ValueError, match="Pass either"):
        locus.Detector(profile="standard", config=cfg)


def test_profile_default_is_standard() -> None:
    left = locus.Detector()._inner.config()
    right = locus.Detector(profile="standard")._inner.config()
    _assert_configs_equal(left, right, "default_vs_standard")
