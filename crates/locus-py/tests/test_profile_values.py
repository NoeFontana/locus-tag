"""Pure-Pydantic profile loading, validation, and serialization."""

from __future__ import annotations

import json

import pytest
from locus._config import SHIPPED_PROFILES, DetectorConfig, ProfileName
from locus.locus import (
    CornerRefinementMode,
    DecodeMode,
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
        "decoder.decode_mode": DecodeMode.Hard,
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
        "decoder.decode_mode": DecodeMode.Hard,
        "segmentation.connectivity": SegmentationConnectivity.Four,
    },
    "high_accuracy": {
        "threshold.enable_sharpening": False,
        "threshold.tile_size": 8,
        "quad.max_elongation": 20.0,
        "quad.min_density": 0.15,
        "quad.extraction_mode": QuadExtractionMode.EdLines,
        # `None` is a Python keyword — reach the variant via getattr.
        "decoder.refinement_mode": getattr(CornerRefinementMode, "None"),
        "decoder.decode_mode": DecodeMode.Hard,
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
        "decoder.decode_mode": DecodeMode.Hard,
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


def test_edlines_rejects_soft_decode() -> None:
    bad = json.dumps(
        {
            "name": "x",
            "quad": {"extraction_mode": "EdLines"},
            "decoder": {"refinement_mode": "None", "decode_mode": "Soft"},
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
    assert isinstance(parsed["decoder"]["decode_mode"], str)
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
