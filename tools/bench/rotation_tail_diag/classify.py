"""Phase 0 rotation-tail failure-mode classifier.

Reads `scenes.json` + `corners.parquet` from an extract run and assigns each
detected scene to exactly one failure mode. Rules are evaluated in order; first
match wins. The `other` bucket is the residual.

Original five rules from the plan, augmented with a `frame_or_winding` mode
that surfaces "corners fit perfectly (low d²) but rotation vs GT is huge" —
the signature of a coordinate-frame or corner-ordering mismatch upstream of
the pose solver. This was added after the first extract run revealed that
both IPPE branches frequently converge to corner-consistent poses with
catastrophic GT rotation error.

Output: `failure_modes.json`.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tools.bench.rotation_tail_diag.io_models import (
    FailureMode,
    FailureModesFile,
    ScenesFile,
)

CLASSIFIER_VERSION = "rotation_tail_diag/v1"

# Thresholds (tunable). All distances/units carry their suffix in the rule name.
ROT_HEALTHY_DEG = 1.0  # below this, scene is healthy and unclassified
BRANCH_FLIP_RATIO = 0.95  # alternate / chosen, where chosen is what shipped
BRANCH_FLIP_REL_D2_GAP = 0.05  # |d2_chosen - d2_alternate| / max(d2)
CORNER_OUTLIER_WEIGHT_THRESHOLD = 0.3
CORNER_OUTLIER_LOO_DROP_PCT = 30.0
GRAZING_AOI_DEG = 65.0
GRAZING_R_THRESHOLD = 0.10
SIGMA_MISCAL_RATIO = 0.5  # estimated_sigma / sqrt(configured_sigma_n_sq)
FRAME_OR_WINDING_D2_AGGREGATE_OK = 5.0  # corners fit well even though rot is bad
FRAME_OR_WINDING_ROT_DEG = 30.0  # …yet rotation against GT is large


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
) -> FailureMode:
    sid = scene["scene_id"]
    rot_chosen = scene.get("rotation_error_chosen_deg")
    rot_alt = scene.get("rotation_error_alternate_deg")
    d2_chosen = scene.get("aggregate_d2_chosen")
    d2_alt = scene.get("aggregate_d2_alternate")
    sigma_n = scene.get("image_noise_sigma")
    aoi = scene.get("angle_of_incidence_deg")

    # Healthy scenes are not a failure mode — they don't classify.
    if rot_chosen is not None and rot_chosen < ROT_HEALTHY_DEG:
        return FailureMode(
            scene_id=sid,
            mode="healthy",
            evidence={"rotation_error_chosen_deg": float(rot_chosen)},
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

    # 2. corner_outlier: ≥1 corner with low IRLS weight AND LOO refit drops rot >30%.
    irls_weights = [
        c.get("final_irls_weight")
        for c in corners
        if c.get("final_irls_weight") is not None
    ]
    loo_drops = [
        c.get("leave_one_out_rotation_err_drop_pct")
        for c in corners
        if c.get("leave_one_out_rotation_err_drop_pct") is not None
    ]
    if irls_weights and loo_drops:
        bad_weights = [w for w in irls_weights if w < CORNER_OUTLIER_WEIGHT_THRESHOLD]
        max_loo_drop = max(loo_drops) if loo_drops else 0.0
        if bad_weights and max_loo_drop > CORNER_OUTLIER_LOO_DROP_PCT:
            return FailureMode(
                scene_id=sid,
                mode="corner_outlier",
                evidence={
                    "min_irls_weight": float(min(irls_weights)),
                    "max_loo_rotation_drop_pct": float(max_loo_drop),
                    "n_corners_below_weight_threshold": len(bad_weights),
                },
            )

    # 3. grazing_angle: high AoI AND degenerate structure tensor.
    if aoi is not None and aoi > GRAZING_AOI_DEG:
        Rs = [
            c.get("structure_tensor_R")
            for c in corners
            if c.get("structure_tensor_R") is not None
        ]
        if Rs and min(Rs) < GRAZING_R_THRESHOLD:
            return FailureMode(
                scene_id=sid,
                mode="grazing_angle",
                evidence={
                    "angle_of_incidence_deg": float(aoi),
                    "min_structure_tensor_R": float(min(Rs)),
                },
            )

    # 4. frame_or_winding: corners fit (low d²) but rotation vs GT is huge.
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

    # 5. sigma_miscalibration: estimated noise floor << configured σ.
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

    # 6. other: residual, requires manual Rerun deep-dive.
    return FailureMode(
        scene_id=sid,
        mode="other",
        evidence={
            "rotation_error_chosen_deg": (
                float(rot_chosen) if rot_chosen is not None else None
            ),
            "aggregate_d2_chosen": (
                float(d2_chosen) if d2_chosen is not None else None
            ),
        },
    )


def run(diagnostic_dir: Path) -> Path:
    """Classify every scene in `diagnostic_dir/scenes.json`. Writes failure_modes.json."""
    scenes_path = diagnostic_dir / "scenes.json"
    corners_path = diagnostic_dir / "corners.parquet"
    out_path = diagnostic_dir / "failure_modes.json"

    raw = json.loads(scenes_path.read_text())
    scenes = ScenesFile.model_validate(raw)
    corners_by_scene = (
        _aggregate_corner_records(corners_path) if corners_path.exists() else {}
    )

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
            _classify_one(scene_dict, corners, scenes.sigma_n_sq_configured)
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
    print("  failure-mode population:")
    for mode, count in population.most_common():
        pct = 100.0 * count / max(len(classifications), 1)
        print(f"    {mode:<22} {count:>3}  ({pct:5.1f}%)")
    return out_path
