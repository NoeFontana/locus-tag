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
    # `name` is load-time profile metadata, not a detector setting; the
    # effective config Rust reports back does not carry it. Compare the
    # detector settings only.
    exclude = {"name"}
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


@pytest.mark.parametrize(
    "document",
    [
        "{}",
        '{"name": "sparse"}',
        '{"threshold": {}, "quad": {}, "decoder": {}, "pose": {}, "segmentation": {}}',
        '{"decoder": {"min_contrast": 12.0}, "quad": {"min_area": 100}}',
        '{"threshold": {"mode": "LocalMean"}, "pose": {"pose_consistency_fpr": 0.001}}',
    ],
)
def test_omitted_keys_parse_identically_in_rust_and_python(document: str) -> None:
    # Every profile key is optional. Rust's serde shim and the Pydantic model must fill an
    # omitted key with the same default; Rust parses the raw document here (not Python's
    # re-serialization of it), so a default that differs between the two shows up.
    from locus import _create_detector_from_config

    python_side = locus.DetectorConfig.from_profile_json(document)
    rust_side = locus.DetectorConfig.model_validate_json(
        _create_detector_from_config(config_json=document).config()
    )
    _assert_configs_equal(rust_side, python_side, document)
    # The keys whose omitted value used to differ, or that profiles commonly omit.
    assert rust_side.decoder.corner_subpix is python_side.decoder.corner_subpix
    assert rust_side.decoder.max_border_error_rate == python_side.decoder.max_border_error_rate
    assert rust_side.segmentation.connectivity == python_side.segmentation.connectivity
    for gate in ("min_fill_ratio", "max_elongation", "min_density"):
        assert getattr(rust_side.quad, gate) == getattr(python_side.quad, gate)
