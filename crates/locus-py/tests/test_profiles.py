"""End-to-end JSON → Pydantic → Rust → Python roundtrip.

Complements ``test_profile_values.py`` (pure-Pydantic schema tests) by
exercising the full stack — each shipped profile is parsed into the Pydantic
``DetectorConfig``, shipped across the FFI as its ``model_dump_json()`` string
through ``_create_detector_from_config``, and re-read via ``Detector.config()``
(which reparses Rust's serialized effective config). Drift between any two
layers fails this suite.

Because the config now crosses the boundary as JSON — the same profile format
Rust already reads — the readback is *total* over every field. The former
field-by-field FFI struct copy silently dropped the ``pose_consistency_*`` /
``outlier_drop_*`` knobs and the adaptive policy on readback; this suite would
now catch that.
"""

from __future__ import annotations

import locus
import pytest
from locus._config import SHIPPED_PROFILES, DetectorConfig, ProfileName


def _assert_close(actual: object, expected: object, path: str) -> None:
    """Deep-compare two ``model_dump`` values, tolerating f32 rounding.

    ``f32`` config fields (e.g. ``quad.max_fill_ratio``) round-trip through
    Rust at single precision, so an ``f64`` source value comes back rounded;
    ``rel``/``abs`` tolerance absorbs that without hiding real drift.
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: {actual!r} is not a dict"
        assert set(actual) == set(expected), f"{path}: keys {set(actual)} != {set(expected)}"
        for key in expected:
            _assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7), (
            f"{path}: {actual} != {expected}"
        )
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def _assert_configs_equal(actual: DetectorConfig, expected: DetectorConfig, label: str) -> None:
    # `name` / `extends` are load-time profile metadata, not detector settings;
    # the effective config Rust reports back carries neither. Compare the
    # detector settings only.
    exclude = {"name", "extends"}
    _assert_close(
        actual.model_dump(mode="python", exclude=exclude),
        expected.model_dump(mode="python", exclude=exclude),
        label,
    )


@pytest.mark.parametrize("profile", SHIPPED_PROFILES)
def test_profile_builds_detector(profile: ProfileName) -> None:
    import numpy as np

    det = locus.Detector(profile=profile)
    batch = det.detect(np.zeros((64, 64), dtype=np.uint8))
    assert len(batch) == 0


@pytest.mark.parametrize("profile", SHIPPED_PROFILES)
def test_config_roundtrip_matches_profile(profile: ProfileName) -> None:
    # The Pydantic source of truth must survive the full JSON round-trip through
    # Rust unchanged (modulo f32 precision) — over *every* field, including the
    # pose-consistency knobs and adaptive policy the old struct copy dropped.
    expected = locus.DetectorConfig.from_profile(profile)
    actual = locus.Detector(profile=profile).config()
    _assert_configs_equal(actual, expected, profile)


def test_profile_and_config_mutually_exclusive() -> None:
    cfg = locus.DetectorConfig.from_profile("standard")
    with pytest.raises(ValueError, match="Pass either"):
        locus.Detector(profile="standard", config=cfg)


def test_profile_default_is_standard() -> None:
    left = locus.Detector().config()
    right = locus.Detector(profile="standard").config()
    _assert_configs_equal(left, right, "default_vs_standard")
