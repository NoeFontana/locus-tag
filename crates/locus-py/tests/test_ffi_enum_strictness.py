"""The JSON FFI boundary must reject malformed profile documents.

The Pydantic ``DetectorConfig`` crosses into Rust as its ``model_dump_json()``
string, parsed by ``DetectorConfig::from_profile_json`` (serde, with
``deny_unknown_fields``). Invalid enum variants and unknown keys must surface as
a ``ValueError`` at construction, not silently coerce or pass through.
"""

from __future__ import annotations

import json

import locus
import pytest


def _base_json() -> dict:
    return json.loads(locus.DetectorConfig.from_profile("standard").model_dump_json())


def _build(doc: dict) -> None:
    locus._create_detector_from_config(
        config_json=json.dumps(doc),
        decimation=None,
        threads=None,
        families=[int(locus.TagFamily.AprilTag36h11)],
    )


def test_valid_json_builds_detector() -> None:
    # Sanity: the unmodified base document is accepted across the boundary.
    _build(_base_json())


@pytest.mark.parametrize(
    ("group", "field"),
    [
        ("quad", "extraction_mode"),
        ("decoder", "refinement_mode"),
        ("segmentation", "connectivity"),
    ],
)
def test_invalid_enum_variant_rejected(group: str, field: str) -> None:
    doc = _base_json()
    doc[group][field] = "NotAVariant"
    with pytest.raises(ValueError):
        _build(doc)


@pytest.mark.parametrize(
    ("group", "field"),
    [
        ("quad", "extraction_mode"),
        ("decoder", "refinement_mode"),
        ("segmentation", "connectivity"),
    ],
)
def test_raw_integer_in_enum_position_rejected(group: str, field: str) -> None:
    # The enums are serialized by their variant *name* (a JSON string). A raw
    # integer must NOT silently coerce into a variant across the boundary — serde
    # rejects it. (This preserves the guarantee the pre-JSON PyDetectorConfig
    # strictness test enforced, which the boundary change would otherwise drop.)
    doc = _base_json()
    doc[group][field] = 0
    with pytest.raises(ValueError):
        _build(doc)


def test_unknown_field_rejected() -> None:
    doc = _base_json()
    doc["quad"]["bogus_knob"] = 1
    with pytest.raises(ValueError):
        _build(doc)
