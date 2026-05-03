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
        "quad.max_elongation": 20.0,
        "quad.min_density": 0.15,
        "quad.extraction_mode": QuadExtractionMode.ContourRdp,
        "decoder.refinement_mode": CornerRefinementMode.Erf,
        "decoder.min_contrast": 20.0,
        "segmentation.connectivity": SegmentationConnectivity.Eight,
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
    },
    "max_recall_adaptive": {
        "threshold.enable_sharpening": True,
        "threshold.tile_size": 8,
        "quad.max_elongation": 20.0,
        "quad.min_density": 0.15,
        # Under AdaptivePpb, `quad.extraction_mode` is ignored at runtime but
        # the field still round-trips through JSON.
        "quad.extraction_mode": QuadExtractionMode.ContourRdp,
        "decoder.refinement_mode": CornerRefinementMode.Erf,
        "segmentation.connectivity": SegmentationConnectivity.Eight,
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
    assert cfg.extends is None


@pytest.mark.parametrize("profile_name", sorted(SHIPPED_PROFILES))
def test_shipped_profile_values(profile_name: ProfileName) -> None:
    cfg = DetectorConfig.from_profile(profile_name)
    for path, expected in EXPECTED[profile_name].items():
        actual = _dotget(cfg, path)
        assert actual == expected, (
            f"profile={profile_name} path={path}: expected {expected!r}, got {actual!r}"
        )


def test_unknown_shipped_profile_name_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown shipped profile"):
        DetectorConfig.from_profile("does_not_exist")  # type: ignore[arg-type]


def test_unknown_json_field_rejected() -> None:
    bad = json.dumps({"name": "x", "threshold": {"tile_size": 8, "bogus": 1}})
    with pytest.raises(ValidationError, match="bogus"):
        DetectorConfig.from_profile_json(bad)


def test_extends_non_null_rejected() -> None:
    bad = json.dumps({"name": "x", "extends": "standard"})
    with pytest.raises((ValidationError, NotImplementedError), match="extends"):
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


def test_threshold_radius_ordering_enforced() -> None:
    bad = json.dumps({"threshold": {"min_radius": 10, "max_radius": 5}})
    with pytest.raises(ValidationError, match="min_radius"):
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


@pytest.mark.parametrize(
    ("bool_value", "expected"),
    [(True, EdLinesImbalanceGatePolicy.Enabled), (False, EdLinesImbalanceGatePolicy.Disabled)],
)
def test_imbalance_gate_legacy_bool_accepted_with_warning(
    bool_value: bool, expected: EdLinesImbalanceGatePolicy
) -> None:
    bare = json.dumps({"name": "x", "quad": {"edlines_imbalance_gate": bool_value}})
    with pytest.warns(DeprecationWarning, match="edlines_imbalance_gate"):
        cfg = DetectorConfig.from_profile_json(bare)
    assert cfg.quad.edlines_imbalance_gate == expected


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
