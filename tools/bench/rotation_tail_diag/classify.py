"""Phase 0 rotation-tail failure-mode classifier.

Reads `scenes.json` + `corners.parquet` from an extract run and assigns each
detected scene to exactly one failure mode. Rules are evaluated in order; first
match wins. The `other` bucket is the residual.

The classifier is **corpus-relative**: a scene is `healthy` only if its
rotation error is below the 85th percentile of the population. This keeps the
healthy bucket from absorbing the tail when the absolute residuals are small
(e.g. on the post-fix `high_accuracy` corpus where p99 ≈ 0.77°), so the
counterfactual table in `report.py` retains discriminating power.

Modes (in evaluation order):
  1. healthy                  — below P85 of rotation_error_chosen_deg.
  2. branch_flip              — alternate IPPE branch is rotationally better
                                AND statistically tied on d².
  3. corner_geometry_outlier  — ANY corner d² > χ²(1, α=1e-4) ≈ 15.137.
                                Stricter sub-class of `corner_outlier`.
  4. corner_outlier           — IRLS weight < 0.3 + LOO refit drops rot >30%.
  5. grazing_extreme          — AoI ≥ 75° AND min structure-tensor R < 0.10.
  6. grazing_moderate         — 60° ≤ AoI < 75° AND min R < 0.10.
  7. frame_or_winding         — corners fit (low d²) but rot vs GT is huge.
  8. sigma_miscalibration     — estimated noise floor << configured σ.
  9. ppm_starved              — bottom-quartile PPM AND rot ≥ P85.
 10. other                    — residual; manual Rerun deep-dive needed.

Output: `failure_modes.json`.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from tools.bench.rotation_tail_diag.io_models import (
    FailureMode,
    FailureModesFile,
    ScenesFile,
)

CLASSIFIER_VERSION = "rotation_tail_diag/v2"

# Thresholds (tunable). All distances/units carry their suffix in the rule name.
HEALTHY_PERCENTILE = 85.0  # scenes below this percentile of rot_err are healthy
BRANCH_FLIP_RATIO = 0.95  # alternate / chosen, where chosen is what shipped
BRANCH_FLIP_REL_D2_GAP = 0.05  # |d2_chosen - d2_alternate| / max(d2)
# χ²(1) inverse-CDF at α = 1e-4. Per-corner Mahalanobis d² above this means
# the corner residual is in the extreme tail of the noise model — a strict
# super-set of the IRLS-weight gate used by `corner_outlier`.
CORNER_GEOMETRY_OUTLIER_D2 = 15.137
CORNER_OUTLIER_WEIGHT_THRESHOLD = 0.3
CORNER_OUTLIER_LOO_DROP_PCT = 30.0
GRAZING_MODERATE_MIN_AOI_DEG = 60.0
GRAZING_EXTREME_MIN_AOI_DEG = 75.0
GRAZING_R_THRESHOLD = 0.10
SIGMA_MISCAL_RATIO = 0.5  # estimated_sigma / sqrt(configured_sigma_n_sq)
FRAME_OR_WINDING_D2_AGGREGATE_OK = 5.0  # corners fit well even though rot is bad
FRAME_OR_WINDING_ROT_DEG = 30.0  # …yet rotation against GT is large
PPM_STARVED_QUANTILE = 0.25  # bottom-quartile PPM


def _aggregate_corner_records(parquet_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Group corner records by scene_id."""
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["scene_id"], []).append(r)
    for scene_id in out:
        out[scene_id].sort(key=lambda r: r["corner_idx"])
    return out


def _classify_one(
    scene: dict[str, Any],
    corners: list[dict[str, Any]],
    sigma_n_sq_configured: float,
    *,
    healthy_rot_cutoff_deg: float,
    ppm_starved_cutoff: float | None,
) -> FailureMode:
    sid = scene["scene_id"]
    rot_chosen = scene.get("rotation_error_chosen_deg")
    rot_alt = scene.get("rotation_error_alternate_deg")
    d2_chosen = scene.get("aggregate_d2_chosen")
    d2_alt = scene.get("aggregate_d2_alternate")
    sigma_n = scene.get("image_noise_sigma")
    aoi = scene.get("angle_of_incidence_deg")
    ppm = scene.get("ppm_estimated")

    # Healthy scenes are not a failure mode — they don't classify. The cutoff
    # is corpus-relative (P85 of `rotation_error_chosen_deg`), so the bucket
    # size is bounded regardless of absolute residuals.
    if rot_chosen is not None and rot_chosen < healthy_rot_cutoff_deg:
        return FailureMode(
            scene_id=sid,
            mode="healthy",
            evidence={
                "rotation_error_chosen_deg": float(rot_chosen),
                "healthy_cutoff_deg": float(healthy_rot_cutoff_deg),
            },
        )

    # 1. branch_flip: alternate branch is rotationally better AND statistically tied.
    if rot_chosen is not None and rot_alt is not None and d2_chosen and d2_alt:
        d2_max = max(abs(d2_chosen), abs(d2_alt), 1e-12)
        rel_gap = abs(d2_chosen - d2_alt) / d2_max
        if rot_alt < rot_chosen * BRANCH_FLIP_RATIO and rel_gap < BRANCH_FLIP_REL_D2_GAP:
            return FailureMode(
                scene_id=sid,
                mode="branch_flip",
                evidence={
                    "rotation_error_chosen_deg": float(rot_chosen),
                    "rotation_error_alternate_deg": float(rot_alt),
                    "aggregate_d2_chosen": float(d2_chosen),
                    "aggregate_d2_alternate": float(d2_alt),
                    "rel_d2_gap": float(rel_gap),
                },
            )

    # 2. corner_geometry_outlier: ANY corner has Mahalanobis d² above the
    # χ²(1) cutoff at α=1e-4 (≈15.137). Stricter than the IRLS / LOO gate
    # below because it triggers on a single severely-mis-localized corner
    # even when the aggregate weights look acceptable.
    corner_d2: list[float] = [
        float(d) for c in corners if (d := c.get("final_mahalanobis_d2")) is not None
    ]
    if corner_d2 and max(corner_d2) > CORNER_GEOMETRY_OUTLIER_D2:
        return FailureMode(
            scene_id=sid,
            mode="corner_geometry_outlier",
            evidence={
                "max_corner_d2": float(max(corner_d2)),
                "chi2_cutoff": float(CORNER_GEOMETRY_OUTLIER_D2),
                "n_corners_above_cutoff": int(
                    sum(1 for d in corner_d2 if d > CORNER_GEOMETRY_OUTLIER_D2)
                ),
            },
        )

    # 3. corner_outlier: ≥1 corner with low IRLS weight AND LOO refit drops rot >30%.
    irls_weights: list[float] = [
        float(w) for c in corners if (w := c.get("final_irls_weight")) is not None
    ]
    loo_drops: list[float] = [
        float(d) for c in corners if (d := c.get("leave_one_out_rotation_err_drop_pct")) is not None
    ]
    if irls_weights and loo_drops:
        bad_weights = [w for w in irls_weights if w < CORNER_OUTLIER_WEIGHT_THRESHOLD]
        max_loo_drop = max(loo_drops) if loo_drops else 0.0
        if bad_weights and max_loo_drop > CORNER_OUTLIER_LOO_DROP_PCT:
            return FailureMode(
                scene_id=sid,
                mode="corner_outlier",
                evidence={
                    "min_irls_weight": min(irls_weights),
                    "max_loo_rotation_drop_pct": max_loo_drop,
                    "n_corners_below_weight_threshold": len(bad_weights),
                },
            )

    # 4. grazing_extreme / grazing_moderate: high AoI AND degenerate
    # structure tensor on at least one corner. Sub-binned because extreme
    # grazing (>=75°) is qualitatively a different failure mode from
    # moderate grazing (60°…75°) — the former is dominated by sub-pixel
    # foreshortening, the latter by line-fit anisotropy.
    if aoi is not None and aoi >= GRAZING_MODERATE_MIN_AOI_DEG:
        rs: list[float] = [
            float(r) for c in corners if (r := c.get("structure_tensor_R")) is not None
        ]
        if rs and min(rs) < GRAZING_R_THRESHOLD:
            mode = "grazing_extreme" if aoi >= GRAZING_EXTREME_MIN_AOI_DEG else "grazing_moderate"
            return FailureMode(
                scene_id=sid,
                mode=mode,
                evidence={
                    "angle_of_incidence_deg": float(aoi),
                    "min_structure_tensor_R": min(rs),
                },
            )

    # 5. frame_or_winding: corners fit (low d²) but rotation vs GT is huge.
    # Diagnostic value of this rule: catches the case where the pose is
    # *geometrically consistent* with the observed corners but rotationally
    # wrong against GT — i.e., upstream coord frame / corner-ordering mismatch.
    # Evaluated *before* sigma_miscalibration because a low-noise render that
    # is also rotationally wrong belongs in this bucket; sigma is a property
    # of the image, not the failure mode.
    if (
        rot_chosen is not None
        and rot_chosen > FRAME_OR_WINDING_ROT_DEG
        and d2_chosen is not None
        and abs(d2_chosen) < FRAME_OR_WINDING_D2_AGGREGATE_OK
    ):
        return FailureMode(
            scene_id=sid,
            mode="frame_or_winding",
            evidence={
                "rotation_error_chosen_deg": float(rot_chosen),
                "aggregate_d2_chosen": float(d2_chosen),
                "branch_chosen_idx": int(scene.get("branch_chosen_idx", 255)),
            },
        )

    # 6. sigma_miscalibration: estimated noise floor << configured σ.
    if sigma_n is not None and sigma_n_sq_configured > 0:
        configured_sigma = sigma_n_sq_configured**0.5
        ratio = sigma_n / configured_sigma if configured_sigma > 0 else 1.0
        if ratio < SIGMA_MISCAL_RATIO:
            return FailureMode(
                scene_id=sid,
                mode="sigma_miscalibration",
                evidence={
                    "image_noise_sigma_estimated": float(sigma_n),
                    "configured_sigma": float(configured_sigma),
                    "ratio": float(ratio),
                },
            )

    # 7. ppm_starved: bottom-quartile pixels-per-meter AND rotation error in
    # the population tail (≥ P85). Captures the "too-far-from-camera" mode
    # where the geometric IPC residual is small relative to the LM noise
    # model but the angular leverage of the corners is collapsing.
    if (
        ppm is not None
        and ppm_starved_cutoff is not None
        and rot_chosen is not None
        and ppm <= ppm_starved_cutoff
        and rot_chosen >= healthy_rot_cutoff_deg
    ):
        return FailureMode(
            scene_id=sid,
            mode="ppm_starved",
            evidence={
                "ppm_estimated": float(ppm),
                "ppm_q1_cutoff": float(ppm_starved_cutoff),
                "rotation_error_chosen_deg": float(rot_chosen),
                "healthy_cutoff_deg": float(healthy_rot_cutoff_deg),
            },
        )

    # 8. other: residual, requires manual Rerun deep-dive.
    return FailureMode(
        scene_id=sid,
        mode="other",
        evidence={
            "rotation_error_chosen_deg": (float(rot_chosen) if rot_chosen is not None else None),
            "aggregate_d2_chosen": (float(d2_chosen) if d2_chosen is not None else None),
        },
    )


def _compute_corpus_cutoffs(
    scenes: ScenesFile,
) -> tuple[float, float | None]:
    """Compute (healthy_rot_cutoff_deg, ppm_starved_cutoff).

    The rotation-error cutoff is the P85 percentile of `rotation_error_chosen_deg`
    across detected scenes; it falls back to the legacy 1.0° threshold when
    fewer than 5 scenes have rotation telemetry (population too small for a
    meaningful percentile).

    The PPM cutoff is the lower-quartile of `ppm_estimated` across detected
    scenes, used by the `ppm_starved` rule. Returns ``None`` for the PPM
    cutoff when the input lacks PPM data.
    """
    rot_errs = [
        s.rotation_error_chosen_deg
        for s in scenes.scenes
        if s.detected and s.rotation_error_chosen_deg is not None
    ]
    if len(rot_errs) >= 5:
        healthy_cutoff = float(np.percentile(np.asarray(rot_errs), HEALTHY_PERCENTILE))
    else:
        healthy_cutoff = 1.0

    ppms = [s.ppm_estimated for s in scenes.scenes if s.detected and s.ppm_estimated is not None]
    ppm_cutoff: float | None = (
        float(np.quantile(np.asarray(ppms), PPM_STARVED_QUANTILE)) if ppms else None
    )
    return healthy_cutoff, ppm_cutoff


def run(diagnostic_dir: Path) -> Path:
    """Classify every scene in `diagnostic_dir/scenes.json`. Writes failure_modes.json."""
    scenes_path = diagnostic_dir / "scenes.json"
    corners_path = diagnostic_dir / "corners.parquet"
    out_path = diagnostic_dir / "failure_modes.json"

    raw = json.loads(scenes_path.read_text())
    scenes = ScenesFile.model_validate(raw)
    corners_by_scene = _aggregate_corner_records(corners_path) if corners_path.exists() else {}

    healthy_cutoff, ppm_cutoff = _compute_corpus_cutoffs(scenes)

    classifications = []
    for scene in scenes.scenes:
        scene_dict = scene.model_dump()
        if not scene.detected:
            classifications.append(
                FailureMode(
                    scene_id=scene.scene_id,
                    mode="production_miss",
                    evidence={"reason": "detector returned no matching tag"},
                )
            )
            continue
        corners = corners_by_scene.get(scene.scene_id, [])
        classifications.append(
            _classify_one(
                scene_dict,
                corners,
                scenes.sigma_n_sq_configured,
                healthy_rot_cutoff_deg=healthy_cutoff,
                ppm_starved_cutoff=ppm_cutoff,
            )
        )

    population = Counter(c.mode for c in classifications)
    out = FailureModesFile(
        classifier_version=CLASSIFIER_VERSION,
        sigma_n_sq_configured=scenes.sigma_n_sq_configured,
        population=dict(population),
        classifications=classifications,
    )
    out_path.write_text(out.model_dump_json(indent=2))

    print()
    print(f"  total scenes:           {len(classifications)}")
    print(f"  healthy cutoff (P{HEALTHY_PERCENTILE:g}):  {healthy_cutoff:.3f}°")
    if ppm_cutoff is not None:
        print(f"  ppm bottom-quartile cutoff: {ppm_cutoff:.1f} px/m")
    print("  failure-mode population:")
    for mode, count in population.most_common():
        pct = 100.0 * count / max(len(classifications), 1)
        print(f"    {mode:<26} {count:>3}  ({pct:5.1f}%)")
    return out_path
