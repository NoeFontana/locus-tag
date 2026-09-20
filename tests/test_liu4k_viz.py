"""Unit tests for ``tools/bench/liu4k_viz.py`` (synthetic data; no dataset, no detector)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.bench.liu4k_viz import (
    CONFIGS,
    FUNNEL_REJECTED_CONTRAST,
    FUNNEL_REJECTED_SAMPLING,
    LOSS_CANDIDATE_OFFSET,
    LOSS_CONTRAST_REJECT,
    LOSS_DECODE_FAIL,
    LOSS_DECODED_WRONG_ID,
    LOSS_HOLLOW_BORDER,
    LOSS_LOW_CONTRAST,
    LOSS_MERGED,
    LOSS_MERGED_GIANT,
    LOSS_MERGED_OR_GATE,
    LOSS_QUAD_GATE,
    ImageScan,
    MissFeatures,
    UnsupportedConfigError,
    build_config_dump,
    classify_miss,
    laplacian_sharpen,
    load_scan,
    mid_extreme_threshold,
    ring_points,
    scan_to_json,
    select_images,
    tile_neighbourhood_stats,
)

# --- loss-class labelling ---------------------------------------------------


def _miss(**kwargs: object) -> MissFeatures:
    base: dict[str, object] = {"ring_foreground": 1.0, "low_range_frac": 0.0}
    base.update(kwargs)
    return MissFeatures(**base)  # pyright: ignore[reportArgumentType]


def test_decoder_stage_wins_over_threshold_stage() -> None:
    # A candidate quad within the match radius means the marker survived the
    # threshold/CCL stages, whatever the ring measurements say.
    f = _miss(
        ring_foreground=0.1,
        low_range_frac=1.0,
        candidate_kind="rejected",
        candidate_funnel_status=FUNNEL_REJECTED_SAMPLING,
        candidate_error=3.0,
        nearest_candidate_px=4.0,
    )
    cls, label = classify_miss(f)
    assert cls == LOSS_DECODE_FAIL
    assert "3 bits" in label

    f = _miss(
        candidate_kind="rejected",
        candidate_funnel_status=FUNNEL_REJECTED_CONTRAST,
        nearest_candidate_px=9.9,
    )
    assert classify_miss(f)[0] == LOSS_CONTRAST_REJECT

    f = _miss(candidate_kind="detection", nearest_candidate_px=2.0)
    assert classify_miss(f)[0] == LOSS_DECODED_WRONG_ID


def test_candidate_just_outside_the_radius_is_its_own_class() -> None:
    f = _miss(candidate_kind="rejected", nearest_candidate_px=18.0)
    cls, label = classify_miss(f)
    assert cls == LOSS_CANDIDATE_OFFSET
    assert "18 px" in label
    # far away: not a candidate story at all, fall through to the pixel stages
    assert classify_miss(_miss(candidate_kind="rejected", nearest_candidate_px=400.0))[0] != cls


def test_hollow_border_needs_flat_tiles_low_contrast_otherwise() -> None:
    assert classify_miss(_miss(ring_foreground=0.26, low_range_frac=0.76))[0] == LOSS_HOLLOW_BORDER
    assert classify_miss(_miss(ring_foreground=0.26, low_range_frac=0.10))[0] == LOSS_LOW_CONTRAST
    # exactly at the cut-offs: 0.70 ring foreground is "binarised", 0.50 flat is "hollow"
    assert classify_miss(_miss(ring_foreground=0.70, low_range_frac=0.9))[0] == LOSS_MERGED_OR_GATE
    assert classify_miss(_miss(ring_foreground=0.69, low_range_frac=0.50))[0] == LOSS_HOLLOW_BORDER


def test_component_measurements_split_merged_from_quad_gate() -> None:
    giant = _miss(dominant_bbox_ratio=120.0, dominant_image_frac=0.95, rdp_vertices=3)
    assert classify_miss(giant)[0] == LOSS_MERGED_GIANT
    merged = _miss(dominant_bbox_ratio=5.7, dominant_image_frac=0.30, rdp_vertices=3)
    assert classify_miss(merged)[0] == LOSS_MERGED
    gate_rdp = _miss(dominant_bbox_ratio=1.4, dominant_image_frac=0.02, rdp_vertices=14)
    cls, label = classify_miss(gate_rdp)
    assert cls == LOSS_QUAD_GATE
    assert "14 vertices" in label
    gate_other = _miss(dominant_bbox_ratio=1.4, dominant_image_frac=0.02, rdp_vertices=5)
    assert classify_miss(gate_other)[0] == LOSS_QUAD_GATE
    # without the component measurements the class stays coarse
    assert classify_miss(_miss())[0] == LOSS_MERGED_OR_GATE


# --- image selection --------------------------------------------------------


def _scans() -> dict[str, dict[str, ImageScan]]:
    standard = {
        "001.jpg": ImageScan("001.jpg", 6, 0, 0, 6, {"merged_component": 6}),
        "004.jpg": ImageScan("004.jpg", 11, 7, 0, 4, {"hollow_border": 4}),
        "010.jpg": ImageScan("010.jpg", 8, 8, 0, 0, {}),
        "011.jpg": ImageScan("011.jpg", 9, 1, 3, 8, {"low_contrast": 8}),
        "012.jpg": ImageScan("012.jpg", 7, 5, 0, 2, {"quad_gate_reject": 2}),
        "013.jpg": ImageScan("013.jpg", 7, 4, 0, 3, {"merged_component": 1}),
        "014.jpg": ImageScan("014.jpg", 6, 6, 0, 0, {}),
    }
    local_mean = dict(standard)
    local_mean["013.jpg"] = ImageScan("013.jpg", 7, 7, 0, 0, {})  # fixes 3
    local_mean["012.jpg"] = ImageScan("012.jpg", 7, 2, 0, 5, {})  # regresses 3
    return {"standard": standard, "local_mean": local_mean}


def test_selection_is_deterministic_and_documents_every_rule() -> None:
    first = select_images(_scans(), target=6)
    second = select_images(_scans(), target=6)
    assert [s.image for s in first] == [s.image for s in second]
    assert [s.reasons for s in first] == [s.reasons for s in second]
    reasons = {s.image: " | ".join(s.reasons) for s in first}
    assert "RCA case" in reasons["004.jpg"]
    assert "RCA case" in reasons["001.jpg"]
    assert any("worst per-image recall" in r for r in reasons.values())
    assert any("best per-image recall" in r for r in reasons.values())
    assert "local_mean fixes 3 marker(s) vs standard" in reasons["013.jpg"]
    assert "local_mean regresses 3 marker(s) vs standard" in reasons["012.jpg"]
    assert any("most false positives" in r for r in reasons.values())
    assert any("most 'hollow_border' misses" in r for r in reasons.values())
    assert all(s.reasons for s in first)


def test_selection_random_fill_is_seeded_and_reaches_the_target() -> None:
    scans = _scans()
    chosen = select_images(scans, target=7)
    assert len(chosen) == 7  # every image in the fixture, via the random fill
    a = select_images(scans, target=6, seed=1)
    b = select_images(scans, target=6, seed=2)
    assert [s.image for s in a] == sorted(s.image for s in a)  # sorted output
    # a different seed may pick a different filler, never a different rule set
    rules = {s.image for s in a if not any("random" in r for r in s.reasons)}
    assert rules == {s.image for s in b if not any("random" in r for r in s.reasons)}


def test_scan_round_trip(tmp_path: Path) -> None:
    scan = ImageScan("007.jpg", 5, 2, 1, 3, {"hollow_border": 3})
    path = tmp_path / "scan_standard.jsonl"
    path.write_text(scan_to_json(scan) + "\n")
    back = load_scan(path)["007.jpg"]
    assert back == scan


# --- configuration matrix ---------------------------------------------------


def test_build_config_dump_applies_overrides_and_rejects_missing_knobs() -> None:
    base = {"threshold": {"enable_sharpening": True, "tile_size": 8}, "quad": {"min_area": 36}}
    out = build_config_dump(base, CONFIGS["no_sharpen"])
    assert out["threshold"]["enable_sharpening"] is False
    assert base["threshold"]["enable_sharpening"] is True  # input untouched
    with pytest.raises(UnsupportedConfigError, match="threshold.mode"):
        build_config_dump(base, CONFIGS["local_mean"])
    with pytest.raises(UnsupportedConfigError, match="sharpening_mode"):
        build_config_dump(base, CONFIGS["shoot_limited"])
    assert build_config_dump(base, CONFIGS["standard"]) == base


# --- pixel-level reproduction ----------------------------------------------


def _reference_sharpen(img: np.ndarray, shoot_limited: bool) -> np.ndarray:
    """Straight transcription of ``filter.rs::laplacian_sharpen`` (slow, explicit)."""
    h, w = img.shape
    out = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            c = int(img[y, x])
            up = int(img[max(y - 1, 0), x])
            down = int(img[min(y + 1, h - 1), x])
            left = int(img[y, max(x - 1, 0)])
            right = int(img[y, min(x + 1, w - 1)])
            v = 5 * c - (up + down + left + right)
            if shoot_limited:
                v = min(max(v, min(c, up, down, left, right)), max(c, up, down, left, right))
            out[y, x] = min(255, max(0, v))
    return out


@pytest.mark.parametrize("shoot_limited", [False, True])
def test_sharpen_matches_the_reference_kernel(shoot_limited: bool) -> None:
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(17, 23), dtype=np.uint8)
    got = laplacian_sharpen(img, shoot_limited=shoot_limited)
    assert np.array_equal(got, _reference_sharpen(img, shoot_limited))


def test_tile_stats_use_the_3x3_tile_neighbourhood_and_drop_the_remainder() -> None:
    img = np.zeros((10, 10), dtype=np.uint8)  # tile_size 4 -> 2x2 tiles, 2 px remainder
    img[0:4, 0:4] = 10
    img[0:4, 4:8] = 200
    img[4:8, 0:4] = 30
    img[4:8, 4:8] = 40
    nmin, nmax = tile_neighbourhood_stats(img, 4)
    # every tile touches every other tile in a 2x2 grid, so all share the extremes
    assert nmin[0, 0] == 10
    assert nmax[0, 0] == 200
    thresh = mid_extreme_threshold(nmin, nmax)
    assert thresh[0, 0] == (10 + 200) >> 1
    # the uncovered remainder is "never foreground"
    assert thresh[9, 9] == 0
    assert np.all(thresh[8:, :] == 0)
    assert np.all(thresh[:, 8:] == 0)


def test_ring_points_stay_inside_the_marker_border() -> None:
    quad = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    pts = ring_points(quad)
    assert len(pts) == 4 * 100 * 3
    assert pts[:, 0].min() >= 0.0
    assert pts[:, 0].max() <= 100.0
    # the deepest inset is 9 % of the way to the centre: inside the 1/8 border
    inset = np.min([pts[:, 0].min(), pts[:, 1].min()])
    assert 0.0 <= inset <= 100.0 / 8.0
