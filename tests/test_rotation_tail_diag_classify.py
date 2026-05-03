"""Unit tests for the Phase 0 rotation-tail failure-mode classifier.

These tests pin the *corpus-relative* behaviour added in 2026-05-03:
the healthy bucket is now defined by a percentile cutoff on
``rotation_error_chosen_deg``, and three new modes
(``corner_geometry_outlier``, ``grazing_extreme``/``grazing_moderate``,
``ppm_starved``) sub-divide the residual tail.

The tests work directly against ``classify._classify_one`` and the
``run`` driver via a synthetic ``ScenesFile`` — no Hub / Rust dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.bench.rotation_tail_diag import classify
from tools.bench.rotation_tail_diag.io_models import (
    FailureModesFile,
    SceneRecord,
    ScenesFile,
)


def _make_scene(
    *,
    scene_id: str,
    rot_err: float,
    aoi: float = 30.0,
    ppm: float = 1500.0,
    distance_m: float = 1.0,
    detected: bool = True,
) -> SceneRecord:
    return SceneRecord(
        scene_id=scene_id,
        tag_id=0,
        distance_m=distance_m,
        angle_of_incidence_deg=aoi,
        pixel_area=1000.0,
        occlusion_ratio=0.0,
        gt_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        gt_translation_xyz=(0.0, 0.0, distance_m),
        detected=detected,
        detected_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0) if detected else None,
        detected_translation_xyz=(0.0, 0.0, distance_m) if detected else None,
        alternate_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0) if detected else None,
        alternate_translation_xyz=(0.0, 0.0, distance_m) if detected else None,
        rotation_error_chosen_deg=rot_err if detected else None,
        rotation_error_alternate_deg=rot_err if detected else None,
        translation_error_chosen_mm=0.0 if detected else None,
        translation_error_alternate_mm=0.0 if detected else None,
        branch_chosen_idx=0,
        aggregate_d2_chosen=1.0,
        aggregate_d2_alternate=1.5,
        branch_d2_ratio=1.5,
        max_corner_d2=2.0,
        lm_iterations=5,
        lm_convergence=0,
        image_noise_sigma=1.0,
        latency_us=10000.0,
        ppm_estimated=ppm,
    )


def _classify_scene(
    scene: SceneRecord,
    corners: list[dict[str, Any]],
    *,
    healthy_cutoff: float = 0.5,
    ppm_cutoff: float | None = 800.0,
    sigma_n_sq: float = 4.0,
) -> str:
    fm = classify._classify_one(
        scene.model_dump(),
        corners,
        sigma_n_sq,
        healthy_rot_cutoff_deg=healthy_cutoff,
        ppm_starved_cutoff=ppm_cutoff,
    )
    return fm.mode


def _empty_corners() -> list[dict[str, Any]]:
    return [
        {
            "scene_id": "s",
            "corner_idx": i,
            "gt_corner_x": 0.0,
            "gt_corner_y": 0.0,
            "detected_corner_x": 0.0,
            "detected_corner_y": 0.0,
            "residual_norm_px": 0.1,
            "final_mahalanobis_d2": 1.0,
            "final_irls_weight": 1.0,
            "structure_tensor_lambda_max": 1.0,
            "structure_tensor_lambda_min": 0.5,
            "structure_tensor_R": 0.5,
            "leave_one_out_rotation_err_drop_pct": 5.0,
        }
        for i in range(4)
    ]


# ---------------------------------------------------------------------------
# Healthy cutoff is corpus-relative.


def test_healthy_below_cutoff() -> None:
    scene = _make_scene(scene_id="s1", rot_err=0.1)
    assert _classify_scene(scene, _empty_corners(), healthy_cutoff=0.5) == "healthy"


def test_above_cutoff_not_healthy() -> None:
    scene = _make_scene(scene_id="s1", rot_err=0.6, ppm=1500.0)
    # Falls through everything else → "other".
    assert _classify_scene(scene, _empty_corners(), healthy_cutoff=0.5) == "other"


# ---------------------------------------------------------------------------
# corner_geometry_outlier strictly precedes corner_outlier and triggers on
# any corner d² > χ²(1, α=1e-4) ≈ 15.137.


def test_corner_geometry_outlier_triggers_on_extreme_d2() -> None:
    corners = _empty_corners()
    corners[2]["final_mahalanobis_d2"] = 20.0  # above 15.137
    scene = _make_scene(scene_id="s1", rot_err=2.0)
    assert _classify_scene(scene, corners, healthy_cutoff=0.5) == "corner_geometry_outlier"


def test_corner_geometry_outlier_below_cutoff_no_trigger() -> None:
    corners = _empty_corners()
    corners[2]["final_mahalanobis_d2"] = 14.9  # just below 15.137
    scene = _make_scene(scene_id="s1", rot_err=2.0)
    assert _classify_scene(scene, corners, healthy_cutoff=0.5) != "corner_geometry_outlier"


def test_corner_geometry_outlier_precedes_corner_outlier() -> None:
    corners = _empty_corners()
    # IRLS+LOO would otherwise classify as `corner_outlier`.
    corners[0]["final_irls_weight"] = 0.05
    corners[0]["leave_one_out_rotation_err_drop_pct"] = 50.0
    # …but a d² spike on a different corner wins first.
    corners[2]["final_mahalanobis_d2"] = 20.0
    scene = _make_scene(scene_id="s1", rot_err=2.0)
    assert _classify_scene(scene, corners, healthy_cutoff=0.5) == "corner_geometry_outlier"


# ---------------------------------------------------------------------------
# grazing_moderate vs grazing_extreme sub-binning.


def test_grazing_moderate() -> None:
    corners = _empty_corners()
    corners[1]["structure_tensor_R"] = 0.05
    scene = _make_scene(scene_id="s1", rot_err=2.0, aoi=70.0)
    assert _classify_scene(scene, corners, healthy_cutoff=0.5) == "grazing_moderate"


def test_grazing_extreme() -> None:
    corners = _empty_corners()
    corners[1]["structure_tensor_R"] = 0.05
    scene = _make_scene(scene_id="s1", rot_err=2.0, aoi=80.0)
    assert _classify_scene(scene, corners, healthy_cutoff=0.5) == "grazing_extreme"


def test_grazing_below_60_no_trigger() -> None:
    corners = _empty_corners()
    corners[1]["structure_tensor_R"] = 0.05
    scene = _make_scene(scene_id="s1", rot_err=2.0, aoi=55.0)
    mode = _classify_scene(scene, corners, healthy_cutoff=0.5)
    assert mode not in {"grazing_moderate", "grazing_extreme"}


def test_grazing_requires_low_R() -> None:
    corners = _empty_corners()  # all R = 0.5
    scene = _make_scene(scene_id="s1", rot_err=2.0, aoi=80.0)
    mode = _classify_scene(scene, corners, healthy_cutoff=0.5)
    assert mode not in {"grazing_moderate", "grazing_extreme"}


# ---------------------------------------------------------------------------
# ppm_starved: bottom-quartile PPM AND rot >= P85.


def test_ppm_starved_triggers() -> None:
    scene = _make_scene(scene_id="s1", rot_err=0.6, ppm=500.0)
    assert (
        _classify_scene(scene, _empty_corners(), healthy_cutoff=0.5, ppm_cutoff=800.0)
        == "ppm_starved"
    )


def test_ppm_starved_requires_tail_rot() -> None:
    scene = _make_scene(scene_id="s1", rot_err=0.1, ppm=500.0)
    # Below healthy cutoff → healthy wins.
    assert (
        _classify_scene(scene, _empty_corners(), healthy_cutoff=0.5, ppm_cutoff=800.0) == "healthy"
    )


def test_ppm_starved_requires_low_ppm() -> None:
    scene = _make_scene(scene_id="s1", rot_err=0.6, ppm=1500.0)
    assert _classify_scene(scene, _empty_corners(), healthy_cutoff=0.5, ppm_cutoff=800.0) == "other"


# ---------------------------------------------------------------------------
# End-to-end run() wiring: corpus cutoffs are derived from the population.


def test_run_uses_corpus_relative_cutoffs(tmp_path: Path) -> None:
    # Build a synthetic ScenesFile with 20 scenes ranging from 0..1°.
    scenes = ScenesFile(
        config_name="synthetic",
        profile="high_accuracy",
        pose_estimation_mode="Accurate",
        sigma_n_sq_configured=4.0,
        n_scenes=20,
        scenes=[
            _make_scene(
                scene_id=f"s{i:02d}",
                rot_err=i * 0.05,
                ppm=300.0 + i * 100.0,
            )
            for i in range(20)
        ],
    )
    diag_dir = tmp_path / "diag"
    diag_dir.mkdir()
    (diag_dir / "scenes.json").write_text(scenes.model_dump_json())
    # No corners.parquet → empty corner aggregator (handled by _classify_one).

    out_path = classify.run(diag_dir)
    fm = FailureModesFile.model_validate_json(out_path.read_text())

    # P85 of [0, 0.05, 0.10, ..., 0.95] is ~0.808°. Roughly 17 of 20 are
    # below P85 → ~17 healthy, residual partitioned into ppm_starved /
    # other depending on PPM rank. The key invariant is "not 100% healthy".
    healthy_count = sum(1 for c in fm.classifications if c.mode == "healthy")
    assert 0 < healthy_count < scenes.n_scenes
    assert fm.classifier_version == classify.CLASSIFIER_VERSION


def test_run_handles_undetected_scenes(tmp_path: Path) -> None:
    scenes = ScenesFile(
        config_name="synthetic",
        profile="high_accuracy",
        pose_estimation_mode="Accurate",
        sigma_n_sq_configured=4.0,
        n_scenes=2,
        scenes=[
            _make_scene(scene_id="s00", rot_err=0.0, detected=False),
            _make_scene(scene_id="s01", rot_err=0.1, detected=True),
        ],
    )
    diag_dir = tmp_path / "diag"
    diag_dir.mkdir()
    (diag_dir / "scenes.json").write_text(scenes.model_dump_json())

    out_path = classify.run(diag_dir)
    fm = FailureModesFile.model_validate_json(out_path.read_text())
    by_sid = {c.scene_id: c.mode for c in fm.classifications}
    assert by_sid["s00"] == "production_miss"
    # detected scene is below the (degenerate, single-sample) cutoff fallback.
    assert by_sid["s01"] in {"healthy", "other"}


# ---------------------------------------------------------------------------
# JSON output schema sanity.


def test_failure_modes_json_round_trip(tmp_path: Path) -> None:
    scenes = ScenesFile(
        config_name="synthetic",
        profile="high_accuracy",
        pose_estimation_mode="Accurate",
        sigma_n_sq_configured=4.0,
        n_scenes=10,
        scenes=[_make_scene(scene_id=f"s{i:02d}", rot_err=i * 0.1) for i in range(10)],
    )
    diag_dir = tmp_path / "diag"
    diag_dir.mkdir()
    (diag_dir / "scenes.json").write_text(scenes.model_dump_json())
    out = classify.run(diag_dir)
    parsed = json.loads(out.read_text())
    assert "classifier_version" in parsed
    assert "population" in parsed
    assert sum(parsed["population"].values()) == scenes.n_scenes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
