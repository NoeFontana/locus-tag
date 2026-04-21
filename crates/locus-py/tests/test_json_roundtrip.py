"""JSON ↔ Pydantic roundtrip tests.

The shipped JSON profiles are the source of truth; these tests pin the
invariant that dumping a loaded config and reloading it is lossless, and
that the on-disk text form uses enum variant names (not integer
discriminants).
"""

from __future__ import annotations

import json

import pytest
from locus._config import SHIPPED_PROFILES, DetectorConfig, ProfileName


@pytest.mark.parametrize("profile", sorted(SHIPPED_PROFILES))
def test_roundtrip_is_semantically_lossless(profile: ProfileName) -> None:
    original = DetectorConfig.from_profile(profile)
    dumped = original.model_dump_json()
    reloaded = DetectorConfig.from_profile_json(dumped)
    assert reloaded == original


@pytest.mark.parametrize("profile", sorted(SHIPPED_PROFILES))
def test_json_preserves_enum_variant_names(profile: ProfileName) -> None:
    cfg = DetectorConfig.from_profile(profile)
    parsed = json.loads(cfg.model_dump_json())
    assert isinstance(parsed["decoder"]["refinement_mode"], str)
    assert isinstance(parsed["decoder"]["decode_mode"], str)
    assert isinstance(parsed["quad"]["extraction_mode"], str)
    assert isinstance(parsed["segmentation"]["connectivity"], str)


@pytest.mark.parametrize("profile", sorted(SHIPPED_PROFILES))
def test_shipped_json_matches_loaded_model(profile: ProfileName) -> None:
    """The bytes on disk deserialize to a model that re-serializes to the same
    structured content (order-independent)."""
    from pathlib import Path

    workspace_root = Path(__file__).resolve().parents[3]
    profile_path = workspace_root / "crates" / "locus-core" / "profiles" / f"{profile}.json"
    on_disk = json.loads(profile_path.read_text())
    loaded = json.loads(DetectorConfig.from_profile(profile).model_dump_json())
    # Drop the ``$schema`` pragma which is metadata-only.
    on_disk.pop("$schema", None)
    # ``extends`` is optional null at rest but required-field in the model.
    on_disk.setdefault("extends", None)
    assert on_disk == loaded
