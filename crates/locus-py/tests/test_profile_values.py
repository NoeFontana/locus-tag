"""Pure-Pydantic profile loading, validation, and serialization."""

from __future__ import annotations

import json
import warnings

import pytest
from locus._config import SHIPPED_PROFILES, DetectorConfig, ProfileName
from locus.locus import (
    CornerRefinementMode,
    EdLinesImbalanceGatePolicy,
    QuadExtractionMode,
    SegmentationConnectivity,
)
from pydantic import ValidationError

EXPECTED = {
    "standard": {
        "threshold.enable_sharpening": True,
        "threshold.tile_size": 8,
        "quad.min_fill_ratio": 0.0,
        "quad.max_elongation": 0.0,
        "quad.min_density": 0.0,
        "quad.extraction_mode": QuadExtractionMode.ContourRdp,
        "decoder.refinement_mode": CornerRefinementMode.Erf,
        "decoder.min_contrast": 20.0,
        "segmentation.connectivity": SegmentationConnectivity.Four,
    },
    "grid": {
        "threshold.enable_sharpening": False,
        "threshold.tile_size": 8,
        "quad.max_elongation": 20.0,
        "quad.min_density": 0.15,
        "quad.min_edge_score": 2.0,
        "quad.extraction_mode": QuadExtractionMode.ContourRdp,
        "decoder.min_contrast": 10.0,
        "decoder.refinement_mode": CornerRefinementMode.Erf,
        "segmentation.connectivity": SegmentationConnectivity.Four,
    },
    "high_accuracy": {
        "threshold.enable_sharpening": False,
        "threshold.tile_size": 8,
        "quad.max_elongation": 20.0,
        "quad.min_density": 0.15,
        # Under AdaptivePpb, `quad.extraction_mode` is ignored at runtime but
        # the field still round-trips through JSON.
        "quad.extraction_mode": QuadExtractionMode.EdLines,
        "quad.edlines_imbalance_gate": EdLinesImbalanceGatePolicy.Enabled,
        # `None` is a Python keyword — reach the variant via getattr.
        "decoder.refinement_mode": getattr(CornerRefinementMode, "None"),
        "segmentation.connectivity": SegmentationConnectivity.Eight,
        # Model-edge pose refinement shipped on for high_accuracy (v0.7.0).
        "pose.pose_edge_refinement_enabled": True,
    },
}


def _dotget(cfg: DetectorConfig, path: str):
    obj = cfg
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize("profile_name", sorted(SHIPPED_PROFILES))
def test_shipped_profile_loads(profile_name: ProfileName) -> None:
    cfg = DetectorConfig.from_profile(profile_name)
    assert cfg.name == profile_name


@pytest.mark.parametrize("profile_name", sorted(SHIPPED_PROFILES))
def test_shipped_profile_values(profile_name: ProfileName) -> None:
    cfg = DetectorConfig.from_profile(profile_name)
    for path, expected in EXPECTED[profile_name].items():
        actual = _dotget(cfg, path)
        assert actual == expected, (
            f"profile={profile_name} path={path}: expected {expected!r}, got {actual!r}"
        )


def test_bare_default_matches_standard_profile() -> None:
    """A bare ``DetectorConfig()`` must equal ``from_profile("standard")``.

    Rust-side counterpart: `config::schema_parity_tests::default_matches_standard_profile`.
    `standard` is documented as the implicit default (`Detector()` /
    `Detector::new()`), so the two Pydantic field defaults drifting apart is a
    silent behavior change for anyone constructing `DetectorConfig()` directly
    (`Detector()` itself is unaffected — it always resolves through
    `from_profile`, never the bare model — but the exported `DetectorConfig`
    class is still expected to match). Caught drifted on four fields
    (`threshold.enable_sharpening`, `quad.min_area`, `quad.max_elongation`,
    `quad.min_density`) before this test existed.
    """
    default = DetectorConfig()
    standard = DetectorConfig.from_profile("standard")
    exclude = {"name"}
    assert default.model_dump(mode="python", exclude=exclude) == standard.model_dump(
        mode="python", exclude=exclude
    )


def test_unknown_shipped_profile_name_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown shipped profile"):
        DetectorConfig.from_profile("does_not_exist")  # pyright: ignore[reportArgumentType]


def test_unknown_json_field_rejected() -> None:
    bad = json.dumps({"name": "x", "threshold": {"tile_size": 8, "bogus": 1}})
    with pytest.raises(ValidationError, match="bogus"):
        DetectorConfig.from_profile_json(bad)


@pytest.mark.parametrize(
    "removed",
    [
        {"extends": None},
        {"threshold": {"min_range": 10}},
        {"threshold": {"constant": 15}},
        {"quad": {"refine_before_decode": False}},
        {"decoder": {"gwlf_transversal_alpha": 0.01}},
        {"pose": {"pose_consistency_min_decisive_ratio": 5.0}},
    ],
)
def test_removed_v0_9_keys_rejected(removed: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DetectorConfig.from_profile_json(json.dumps(removed))


def test_gwlf_refinement_mode_rejected() -> None:
    bad = json.dumps({"decoder": {"refinement_mode": "Gwlf"}})
    with pytest.raises(ValidationError, match="CornerRefinementMode"):
        DetectorConfig.from_profile_json(bad)


def test_adaptive_ppb_requires_static_refinement_none() -> None:
    bad = json.dumps(
        {
            "quad": {"extraction_policy": {"AdaptivePpb": {}}},
            "decoder": {"refinement_mode": "Erf"},
        }
    )
    with pytest.raises(ValidationError, match="AdaptivePpb requires decoder.refinement_mode=None"):
        DetectorConfig.from_profile_json(bad)
    ok = json.dumps(
        {
            "quad": {"extraction_policy": {"AdaptivePpb": {}}},
            "decoder": {"refinement_mode": "None"},
        }
    )
    DetectorConfig.from_profile_json(ok)


@pytest.mark.parametrize("noise_k", [0.0, -1.0])
def test_threshold_noise_k_must_be_positive(noise_k: float) -> None:
    bad = json.dumps({"threshold": {"noise_k": noise_k}})
    with pytest.raises(ValidationError, match="noise_k"):
        DetectorConfig.from_profile_json(bad)


def test_structure_tensor_radius_must_be_positive() -> None:
    bad = json.dumps({"pose": {"structure_tensor_radius": 0}})
    with pytest.raises(ValidationError, match="structure_tensor_radius"):
        DetectorConfig.from_profile_json(bad)


def test_edlines_rejects_erf_refinement() -> None:
    bad = json.dumps(
        {
            "name": "x",
            "quad": {"extraction_mode": "EdLines"},
            "decoder": {"refinement_mode": "Erf"},
        }
    )
    with pytest.raises(ValidationError, match="EdLines"):
        DetectorConfig.from_profile_json(bad)


def test_threshold_local_mean_radius_must_be_positive() -> None:
    bad = json.dumps({"threshold": {"local_mean_radius": 0}})
    with pytest.raises(ValidationError, match="local_mean_radius"):
        DetectorConfig.from_profile_json(bad)


def test_threshold_mode_rejects_unknown_variant() -> None:
    bad = json.dumps({"threshold": {"mode": "NotAMode"}})
    with pytest.raises(ValidationError, match="ThresholdMode"):
        DetectorConfig.from_profile_json(bad)


def test_fill_ratio_ordering_enforced() -> None:
    bad = json.dumps({"quad": {"min_fill_ratio": 0.9, "max_fill_ratio": 0.5}})
    with pytest.raises(ValidationError, match="min_fill_ratio"):
        DetectorConfig.from_profile_json(bad)


@pytest.mark.parametrize("profile_name", sorted(SHIPPED_PROFILES))
def test_json_roundtrip_is_lossless(profile_name: ProfileName) -> None:
    cfg = DetectorConfig.from_profile(profile_name)
    dumped = cfg.model_dump_json()
    roundtripped = DetectorConfig.from_profile_json(dumped)
    assert roundtripped == cfg


@pytest.mark.parametrize("profile_name", sorted(SHIPPED_PROFILES))
def test_json_emits_enum_names(profile_name: ProfileName) -> None:
    cfg = DetectorConfig.from_profile(profile_name)
    parsed = json.loads(cfg.model_dump_json())
    assert isinstance(parsed["decoder"]["refinement_mode"], str)
    assert isinstance(parsed["quad"]["extraction_mode"], str)
    assert isinstance(parsed["segmentation"]["connectivity"], str)


def test_int_discriminant_accepted_for_enum_fields() -> None:
    bare = json.dumps(
        {
            "name": "x",
            "decoder": {"refinement_mode": int(CornerRefinementMode.Erf)},
            "segmentation": {"connectivity": int(SegmentationConnectivity.Four)},
        }
    )
    cfg = DetectorConfig.from_profile_json(bare)
    assert cfg.decoder.refinement_mode == CornerRefinementMode.Erf
    assert cfg.segmentation.connectivity == SegmentationConnectivity.Four


@pytest.mark.parametrize("bool_value", [True, False])
def test_imbalance_gate_bool_form_rejected(bool_value: bool) -> None:
    bare = json.dumps({"name": "x", "quad": {"edlines_imbalance_gate": bool_value}})
    with pytest.raises(ValidationError, match="EdLinesImbalanceGatePolicy"):
        DetectorConfig.from_profile_json(bare)


@pytest.mark.parametrize(
    ("string_value", "expected"),
    [
        ("Enabled", EdLinesImbalanceGatePolicy.Enabled),
        ("Disabled", EdLinesImbalanceGatePolicy.Disabled),
    ],
)
def test_imbalance_gate_string_form_no_warning(
    string_value: str, expected: EdLinesImbalanceGatePolicy
) -> None:
    bare = json.dumps({"name": "x", "quad": {"edlines_imbalance_gate": string_value}})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cfg = DetectorConfig.from_profile_json(bare)
    assert cfg.quad.edlines_imbalance_gate == expected


def test_imbalance_gate_unknown_string_rejected() -> None:
    bare = json.dumps({"name": "x", "quad": {"edlines_imbalance_gate": "AutoMagic"}})
    with pytest.raises(ValidationError, match="EdLinesImbalanceGatePolicy"):
        DetectorConfig.from_profile_json(bare)
