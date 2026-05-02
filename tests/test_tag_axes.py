"""Tests for ``DatasetLoader.load_axes`` and the ``TagAxes`` dataclass.

Uses the real ``single_tag_locus_v1_tag36h11_*`` hub corpora as fixtures —
those manifests are checked into ``tests/data/hub_cache/`` and treated as
golden data by the regression suite.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tools.bench.utils import (
    HUB_CACHE_DIR,
    DatasetLoader,
    TagAxes,
    max_edge_px,
)


@pytest.fixture
def loader() -> DatasetLoader:
    # icra_dir doesn't matter — load_axes resolves hub_cache via HUB_CACHE_DIR.
    return DatasetLoader(icra_dir=HUB_CACHE_DIR)


@pytest.mark.parametrize(
    "scenario",
    [
        "single_tag_locus_v1_tag36h11_640x480",
        "single_tag_locus_v1_tag36h11_1280x720",
        "single_tag_locus_v1_tag36h11_1920x1080",
        "single_tag_locus_v1_tag36h11_3840x2160",
    ],
)
def test_load_axes_returns_one_entry_per_image(loader: DatasetLoader, scenario: str) -> None:
    if not (HUB_CACHE_DIR / scenario).exists():
        pytest.skip(f"{scenario} not present in tests/data/hub_cache/")
    axes = loader.load_axes(scenario)
    assert len(axes) > 0
    # Each image has at least one tag with finite physical axes.
    for img_axes in axes.values():
        assert len(img_axes) >= 1
        for tag_axes in img_axes.values():
            assert isinstance(tag_axes, TagAxes)
            assert math.isfinite(tag_axes.distance_m)
            assert math.isfinite(tag_axes.aoi_deg)
            assert math.isfinite(tag_axes.ppm)
            assert tag_axes.ppm > 0


def test_resolution_h_matches_scenario_name(loader: DatasetLoader) -> None:
    """The H value on every TagAxes must match the corpus name."""
    scenario = "single_tag_locus_v1_tag36h11_1920x1080"
    if not (HUB_CACHE_DIR / scenario).exists():
        pytest.skip(f"{scenario} not present")
    axes = loader.load_axes(scenario)
    for img_axes in axes.values():
        for tag_axes in img_axes.values():
            assert tag_axes.resolution_h == 1080


def test_ppm_derivation_is_consistent_with_corners(loader: DatasetLoader) -> None:
    """``TagAxes.ppm`` must equal ``max_edge_px(corners) / (tag_size_mm/1000)``.

    The hub manifest stores ``ppm: 0.0`` (sentinel for "not provided"), so the
    loader is responsible for the derivation. Re-derive from the GT corners
    and compare.
    """
    import json

    scenario = "single_tag_locus_v1_tag36h11_1920x1080"
    rich_path = HUB_CACHE_DIR / scenario / "rich_truth.json"
    if not rich_path.exists():
        pytest.skip(f"{scenario} not present")

    with rich_path.open() as f:
        data = json.load(f)
    entries = data["records"] if isinstance(data, dict) and "records" in data else data
    tag_entries = [e for e in entries if e.get("record_type", "TAG") == "TAG"]
    assert len(tag_entries) > 0

    axes = loader.load_axes(scenario)

    for entry in tag_entries[:5]:
        img = entry["image_filename"] if "image_filename" in entry else f"{entry['image_id']}.png"
        if not img.endswith(".png"):
            img = f"{img}.png"
        tid = int(entry["tag_id"])
        expected_ppm = max_edge_px(np.asarray(entry["corners"], dtype=np.float64)) / (
            float(entry["tag_size_mm"]) / 1000.0
        )
        assert axes[img][tid].ppm == pytest.approx(expected_ppm, rel=1e-6)


def test_velocity_is_none_for_static_corpora(loader: DatasetLoader) -> None:
    """All v1 hub corpora have ``velocity: null`` — the loader must surface
    that as ``None`` (not NaN) so :func:`tools.bench.strata._bucket_mot`
    classifies them as ``static``.
    """
    scenario = "single_tag_locus_v1_tag36h11_1920x1080"
    if not (HUB_CACHE_DIR / scenario).exists():
        pytest.skip(f"{scenario} not present")
    axes = loader.load_axes(scenario)
    for img_axes in axes.values():
        for tag_axes in img_axes.values():
            assert tag_axes.velocity is None


def test_unknown_scenario_returns_empty(loader: DatasetLoader, tmp_path: Path) -> None:
    """Asking for a scenario without a rich_truth.json returns an empty dict,
    not a crash — caller treats this as ICRA-style ``unk`` strata.
    """
    icra_only = DatasetLoader(icra_dir=tmp_path)
    assert icra_only.load_axes("nonexistent_scenario") == {}
