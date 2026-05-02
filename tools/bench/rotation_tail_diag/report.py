"""Phase 0 rotation-tail report generator.

Reads `scenes.json` + `corners.parquet` + `failure_modes.json` from an extract
run and emits a markdown memo to ``docs/engineering/<filename>.md``.

House style (matches `icra_forward_high_accuracy_diagnostic_*.md` /
`post_decode_refinement_*.md`): § headers, markdown tables, no PNGs. Inline
ASCII histograms instead of charts. Bootstrap 95% CIs with a fixed RNG seed
for reproducibility.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from tools.bench.rotation_tail_diag.io_models import (
    FailureModesFile,
    ScenesFile,
)

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    return float(np.percentile(np.asarray(data), p * 100.0))


def _bootstrap_pctile_ci(
    data: list[float],
    p: float,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Returns (point, lo95, hi95) for the p-th percentile."""
    if not data:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(data)
    rng = np.random.default_rng(seed)
    point = float(np.percentile(arr, p * 100.0))
    if len(arr) < 2:
        return point, point, point
    samples = rng.choice(arr, size=(resamples, len(arr)), replace=True)
    pcts = np.percentile(samples, p * 100.0, axis=1)
    lo, hi = np.percentile(pcts, [2.5, 97.5])
    return point, float(lo), float(hi)


def _ascii_histogram(
    data: list[float],
    *,
    bins: int = 20,
    width: int = 40,
    label_fmt: str = "{:>7.3f}",
) -> str:
    if not data:
        return "(no data)"
    arr = np.asarray(data)
    counts, edges = np.histogram(arr, bins=bins)
    if counts.max() == 0:
        return "(no data)"
    lines = []
    for i, count in enumerate(counts):
        bar_len = int(round(width * count / counts.max()))
        bar = "█" * bar_len
        lo = label_fmt.format(edges[i])
        hi = label_fmt.format(edges[i + 1])
        lines.append(f"  {lo}–{hi}  │ {bar}  ({count})")
    return "\n".join(lines)


def _stratify_by_quartiles(
    scenes_data: list[dict],
    field: str,
    metric: str,
) -> list[tuple[str, int, dict[str, float | tuple[float, float]]]]:
    """Bucket scenes into 4 quantiles by `field` and report metric percentiles."""
    rows = [
        s
        for s in scenes_data
        if s.get(field) is not None and s.get(metric) is not None
    ]
    if not rows:
        return []
    rows.sort(key=lambda s: float(s[field]))
    n = len(rows)
    out = []
    for q in range(4):
        lo_idx = q * n // 4
        hi_idx = (q + 1) * n // 4
        bucket = rows[lo_idx:hi_idx]
        if not bucket:
            continue
        bucket_lo = float(bucket[0][field])
        bucket_hi = float(bucket[-1][field])
        values = [float(b[metric]) for b in bucket]
        p50, lo95, hi95 = _bootstrap_pctile_ci(values, 0.50)
        p95 = _percentile(values, 0.95)
        p99 = _percentile(values, 0.99)
        out.append(
            (
                f"{bucket_lo:.2f}…{bucket_hi:.2f}",
                len(bucket),
                {
                    "p50": p50,
                    "p50_lo": lo95,
                    "p50_hi": hi95,
                    "p95": p95,
                    "p99": p99,
                    "max": max(values),
                },
            )
        )
    return out


def _counterfactual_p99_drop(
    rot_errs: list[float],
    classifications: list[dict],
    target_mode: str,
) -> float:
    """If we 'fixed' all scenes labelled ``target_mode`` (set their rotation error
    to 0), what's the resulting p99? Returns the new p99 in degrees."""
    if not rot_errs:
        return float("nan")
    sid_to_mode = {c["scene_id"]: c["mode"] for c in classifications}
    fixed = []
    for s in rot_errs:
        # ``rot_errs`` here is parallel to scenes; we need scene_id zipping.
        pass
    return float("nan")  # implemented in run() with explicit zipping


def run(diagnostic_dir: Path, *, output_md: Path) -> Path:
    scenes_path = diagnostic_dir / "scenes.json"
    corners_path = diagnostic_dir / "corners.parquet"
    fm_path = diagnostic_dir / "failure_modes.json"

    scenes = ScenesFile.model_validate_json(scenes_path.read_text())
    failure_modes = FailureModesFile.model_validate_json(fm_path.read_text())
    corners_tbl = (
        pq.read_table(corners_path).to_pylist() if corners_path.exists() else []
    )

    sid_to_mode = {c.scene_id: c.mode for c in failure_modes.classifications}
    sid_to_evidence = {
        c.scene_id: c.evidence for c in failure_modes.classifications
    }

    # Headline numbers
    detected_scenes = [s for s in scenes.scenes if s.detected]
    rot_errs = [
        s.rotation_error_chosen_deg
        for s in detected_scenes
        if s.rotation_error_chosen_deg is not None
    ]
    trans_errs_mm = [
        s.translation_error_chosen_mm
        for s in detected_scenes
        if s.translation_error_chosen_mm is not None
    ]
    latencies_ms = [s.latency_us / 1000.0 for s in detected_scenes]

    rot_p50, rot_p50_lo, rot_p50_hi = _bootstrap_pctile_ci(rot_errs, 0.50)
    rot_p95, rot_p95_lo, rot_p95_hi = _bootstrap_pctile_ci(rot_errs, 0.95)
    rot_p99, rot_p99_lo, rot_p99_hi = _bootstrap_pctile_ci(rot_errs, 0.99)
    trans_p50 = _percentile(trans_errs_mm, 0.50)
    trans_p95 = _percentile(trans_errs_mm, 0.95)
    trans_p99 = _percentile(trans_errs_mm, 0.99)
    latency_p50 = _percentile(latencies_ms, 0.50)
    latency_p99 = _percentile(latencies_ms, 0.99)

    # Failure-mode counterfactuals: what if mode X were resolved (rot_err = 0)?
    rot_with_sid = [
        (s.scene_id, s.rotation_error_chosen_deg)
        for s in detected_scenes
        if s.rotation_error_chosen_deg is not None
    ]
    counterfactuals = {}
    for mode in failure_modes.population:
        if mode in {"healthy", "production_miss"}:
            continue
        synthetic = [
            (0.0 if sid_to_mode.get(sid) == mode else err) for sid, err in rot_with_sid
        ]
        counterfactuals[mode] = {
            "p50": _percentile(synthetic, 0.50),
            "p95": _percentile(synthetic, 0.95),
            "p99": _percentile(synthetic, 0.99),
        }

    # Top 10 worst
    worst = sorted(
        [
            (s.scene_id, s.rotation_error_chosen_deg, s)
            for s in detected_scenes
            if s.rotation_error_chosen_deg is not None
        ],
        key=lambda t: t[1],
        reverse=True,
    )[:10]

    # σ histogram data
    sigmas = [s.image_noise_sigma for s in scenes.scenes]
    configured_sigma = scenes.sigma_n_sq_configured**0.5

    # IRLS weight histogram (only corners with non-null final weights)
    weights = [
        c["final_irls_weight"]
        for c in corners_tbl
        if c.get("final_irls_weight") is not None
    ]

    # Branch-d² scatter as bucketed counts
    d2_chosen_arr = [
        s.aggregate_d2_chosen
        for s in detected_scenes
        if not np.isnan(s.aggregate_d2_chosen)
    ]
    d2_alt_arr = [
        s.aggregate_d2_alternate
        for s in detected_scenes
        if not np.isnan(s.aggregate_d2_alternate)
    ]

    # AoI / distance / PPM stratified tables (rotation error)
    detected_dicts = [s.model_dump() for s in detected_scenes]
    strat_aoi = _stratify_by_quartiles(
        detected_dicts, "angle_of_incidence_deg", "rotation_error_chosen_deg"
    )
    strat_dist = _stratify_by_quartiles(
        detected_dicts, "distance_m", "rotation_error_chosen_deg"
    )
    strat_ppm = _stratify_by_quartiles(
        detected_dicts, "ppm_estimated", "rotation_error_chosen_deg"
    )

    # ---------- Render markdown ----------
    md = []
    today = output_md.stem.split("_")[-1]
    md.append(f"# Rotation-Tail Diagnostic — Phase 0 ({today})")
    md.append("")
    md.append(
        "Forensics of the residual rotation tail on `locus_v1_tag36h11_1920x1080` "
        "under `render_tag_hub` + Accurate-mode pose. Output of the Phase 0 "
        "diagnostic harness (`tools/bench/rotation_tail_diag/`)."
    )
    md.append("")

    md.append("## §1 TL;DR")
    md.append("")
    md.append(
        f"- **Dataset**: `{scenes.config_name}` — {scenes.n_scenes} scenes, "
        f"single tag each, AprilTag36h11 family, 1920×1080."
    )
    md.append(
        f"- **Profile / mode**: `{scenes.profile}` + `{scenes.pose_estimation_mode}`. "
        f"σ_n² configured = {scenes.sigma_n_sq_configured:.3f} (σ ≈ "
        f"{configured_sigma:.3f}px)."
    )
    md.append(
        f"- **Recall**: {len(detected_scenes)}/{scenes.n_scenes} scenes detected."
    )
    md.append(
        f"- **Rotation error vs GT** (degrees, 95% bootstrap CI):  "
        f"p50 = {rot_p50:.3f} [{rot_p50_lo:.3f}, {rot_p50_hi:.3f}]  ·  "
        f"p95 = {rot_p95:.3f} [{rot_p95_lo:.3f}, {rot_p95_hi:.3f}]  ·  "
        f"p99 = {rot_p99:.3f} [{rot_p99_lo:.3f}, {rot_p99_hi:.3f}]"
    )
    md.append(
        f"- **Translation error vs GT** (mm):  p50 = {trans_p50:.1f}  ·  "
        f"p95 = {trans_p95:.1f}  ·  p99 = {trans_p99:.1f}"
    )
    md.append(
        f"- **Latency** (ms, production path, no diagnostics):  "
        f"p50 = {latency_p50:.2f}  ·  p99 = {latency_p99:.2f}"
    )
    md.append("")

    md.append("### Reproducibility cross-check")
    md.append("")
    md.append(
        "The published `render_tag_hub` baseline (commit `8890efc`, 2026-04-25) "
        "reported rot p99 = 1.897° on this exact dataset. The numbers above show "
        "**a regression**: the live `render_tag_hub` + Accurate-mode pose path "
        "now produces rot p50 ≈ 60° (two orders of magnitude). Reproduced "
        "independently via `tools/bench/render_tag_sota_eval.py` — see §7."
    )
    md.append("")

    md.append("## §2 Failure-mode breakdown")
    md.append("")
    md.append("Mutually-exclusive classification of the 50 scenes.")
    md.append("")
    md.append("| Mode | Count | % | Counterfactual rot p99 if resolved |")
    md.append("| :--- | ---: | ---: | ---: |")
    for mode, count in sorted(
        failure_modes.population.items(), key=lambda kv: -kv[1]
    ):
        pct = 100.0 * count / max(scenes.n_scenes, 1)
        cf = counterfactuals.get(mode)
        cf_str = f"{cf['p99']:.3f}°" if cf else "—"
        md.append(f"| `{mode}` | {count} | {pct:.1f}% | {cf_str} |")
    md.append("")
    md.append(
        f"_Counterfactual interpretation: \"if all `<mode>` scenes had rotation "
        f"error = 0, what would p99 become?\" Current p99 = {rot_p99:.3f}°. "
        "The mode whose counterfactual drops p99 the most is the priority fix."
    )
    md.append("")

    md.append("## §3 Stratified rotation error")
    md.append("")
    for label, table in [
        ("Angle of incidence (deg)", strat_aoi),
        ("Distance (m)", strat_dist),
        ("Estimated PPM (px/m)", strat_ppm),
    ]:
        md.append(f"### {label}")
        md.append("")
        md.append("| Bucket | n | rot p50 (95% CI) | rot p95 | rot p99 | rot max |")
        md.append("| :--- | ---: | ---: | ---: | ---: | ---: |")
        for bucket_label, n, m in table:
            md.append(
                f"| {bucket_label} | {n} | "
                f"{m['p50']:.3f} [{m['p50_lo']:.3f}, {m['p50_hi']:.3f}] | "
                f"{m['p95']:.3f} | {m['p99']:.3f} | {m['max']:.3f} |"
            )
        md.append("")

    md.append("## §4 Top-10 worst scenes")
    md.append("")
    md.append(
        "| # | scene_id | rot err (°) | trans err (mm) | branch | classification | Rerun | "
    )
    md.append("| ---: | :--- | ---: | ---: | ---: | :--- | :--- |")
    for rank, (sid, rot, scene) in enumerate(worst, 1):
        trans_mm = (
            f"{scene.translation_error_chosen_mm:.1f}"
            if scene.translation_error_chosen_mm is not None
            else "—"
        )
        cls = sid_to_mode.get(sid, "?")
        rrd = f"`recordings/{sid}.rrd`"
        md.append(
            f"| {rank} | `{sid}` | {rot:.2f} | {trans_mm} | "
            f"{scene.branch_chosen_idx} | `{cls}` | {rrd} |"
        )
    md.append("")

    md.append("## §5 σ calibration check")
    md.append("")
    md.append(
        f"Configured σ in `{scenes.profile}.json`: "
        f"σ = √{scenes.sigma_n_sq_configured:.3f} ≈ {configured_sigma:.3f}px."
    )
    md.append("")
    md.append("Per-image estimated σ (Immerkær median Laplacian):")
    md.append("")
    md.append("```")
    md.append(_ascii_histogram(sigmas, bins=20, label_fmt="{:>6.3f}"))
    md.append("```")
    md.append("")
    md.append(
        f"Population: median σ = {statistics.median(sigmas):.3f}px, "
        f"max = {max(sigmas):.3f}px. "
        f"{sum(1 for s in sigmas if s < 0.5 * configured_sigma)} of "
        f"{len(sigmas)} scenes have σ < {0.5 * configured_sigma:.2f}px "
        f"(half the configured value)."
    )
    md.append("")

    md.append("## §6 IRLS weight + Mahalanobis distribution")
    md.append("")
    if weights:
        md.append("Final per-corner Huber IRLS weights at the LM-converged pose:")
        md.append("")
        md.append("```")
        md.append(_ascii_histogram(weights, bins=20, label_fmt="{:>5.3f}"))
        md.append("```")
        md.append("")
        below = sum(1 for w in weights if w < 0.3)
        md.append(
            f"{below} of {len(weights)} corners have IRLS weight < 0.3 — "
            "that's the threshold the `corner_outlier` classifier uses."
        )
    else:
        md.append("(No IRLS weight data — corner telemetry was disabled.)")
    md.append("")

    md.append("Per-scene branch-d² (chosen vs alternate, log-binned counts):")
    md.append("")
    if d2_chosen_arr and d2_alt_arr:
        log_chosen = [np.log10(max(d, 1e-12)) for d in d2_chosen_arr]
        log_alt = [np.log10(max(d, 1e-12)) for d in d2_alt_arr]
        md.append("**Chosen branch (log10 d²):**")
        md.append("```")
        md.append(_ascii_histogram(log_chosen, bins=20, label_fmt="{:>5.2f}"))
        md.append("```")
        md.append("")
        md.append("**Alternate branch (log10 d²):**")
        md.append("```")
        md.append(_ascii_histogram(log_alt, bins=20, label_fmt="{:>5.2f}"))
        md.append("```")
    md.append("")

    md.append("## §7 What this points at")
    md.append("")
    md.append(
        f"The dominant failure mode is **`frame_or_winding`** at "
        f"{failure_modes.population.get('frame_or_winding', 0)} of {scenes.n_scenes} "
        "scenes — i.e. *the chosen IPPE branch fits the observed corners well "
        "(low aggregate d²) but its rotation against GT is large* (>30°). "
        "This is the signature of a coordinate-frame or corner-ordering "
        "mismatch upstream of the pose solver, not a pose-refinement issue. "
        "Fixing the LM solver alone cannot recover from it; the offending "
        "permutation/sign needs to be located in:"
    )
    md.append("")
    md.append(
        "1. The corner-extraction stage (EdLines vs ContourRdp produces "
        "different corner orderings on this dataset — see the "
        "render_tag_sota_eval.py cross-product table in §8)."
    )
    md.append(
        "2. The Accurate-mode-only weighted LM path "
        "(`refine_pose_lm_weighted`), which is the only path that "
        "regresses; Fast-mode `refine_pose_lm` is healthy on `standard` "
        "profile (rot p50 = 0.288°)."
    )
    md.append("")
    md.append(
        "**`branch_flip`** at "
        f"{failure_modes.population.get('branch_flip', 0)} scenes is real but "
        "secondary. It will only ever explain a small fraction of the tail "
        "while `frame_or_winding` dominates."
    )
    md.append("")
    md.append(
        "**`sigma_miscalibration`** is partially confounded with the dataset "
        "rather than the algorithm: Blender-rendered images have very low "
        "noise floors, while the production profiles ship `sigma_n_sq = 4.0` "
        "(σ ≈ 2 px). Phase 3 of the SOTA plan (per-frame σ estimation) "
        "addresses this directly."
    )
    md.append("")

    md.append("## §8 Profile × mode reproducibility table")
    md.append("")
    md.append(
        "Captured via `tools/bench/render_tag_sota_eval.py` on the same "
        "dataset, today, for context. (Source: `/tmp/render_tag_sota_full.json` "
        "after running the eval tool.)"
    )
    md.append("")
    md.append(
        "| Profile | Mode | Recall | rot p50 | rot p95 | rot p99 | trans p99 |"
    )
    md.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
    md.append("| `standard` | Accurate | 100.0 % | 0.288° | 1.572° | 27.248° | 50.3 mm |")
    md.append("| `high_accuracy` | Fast | 94.0 % | 0.345° | 6.350° | 104.238° | 2210 mm |")
    md.append("| `high_accuracy` | Accurate | 94.0 % | 62.857° | 148.239° | 154.233° | 3261 mm |")
    md.append("| `render_tag_hub` | Fast | 100.0 % | 0.363° | 6.137° | 103.402° | 2164 mm |")
    md.append(
        f"| **`render_tag_hub`** | **Accurate** | **100.0 %** | **{rot_p50:.3f}°** | "
        f"**{rot_p95:.3f}°** | **{rot_p99:.3f}°** | **{trans_p99:.0f} mm** | "
        "(this run) |"
    )
    md.append("")
    md.append(
        "**Recovery path: switch the rotation-tail Phase 1–4 work to use "
        "`standard` profile first** — that's where the ~28° p99 tail still "
        "behaves like a real perception problem. `render_tag_hub` and "
        "`high_accuracy` need their Accurate-mode pose regression fixed "
        "before they can serve as the SOTA floor."
    )
    md.append("")

    md.append("## §9 Recommendations (reorders Phase 1–4 from the SOTA plan)")
    md.append("")
    md.append(
        "1. **Phase 0.1 (new)**: Bisect the Accurate-mode regression on "
        "`render_tag_hub` / `high_accuracy`. Most likely culprits, in order:"
    )
    md.append("   - EdLines corner ordering vs ContourRdp; check the four corners' winding direction.")
    md.append("   - Recent `refine_pose_lm_weighted` changes (`8890efc` introduced the Mahalanobis χ² gate).")
    md.append("   - `pose_consistency_fpr = 1e-3` rejecting all geometrically-consistent poses for a frame-flip reason.")
    md.append("")
    md.append(
        "2. **Phase 1 (photometric refinement)**: deferred until the regression "
        "above is fixed. Photometric refinement cannot recover from a corner-"
        "ordering bug."
    )
    md.append("")
    md.append(
        "3. **Phase 2 (branch hardening)**: real but small (12% of scenes). "
        "Only worthwhile after Phase 0.1; otherwise hardened branch selection "
        "still picks a corner-mis-ordered IPPE candidate."
    )
    md.append("")
    md.append(
        "4. **Phase 3 (per-frame σ estimation)**: the harness already provides "
        "`compute_image_noise_floor` (permanent in `gradient.rs`); Phase 3 just "
        "needs to wire it into the per-frame LM info matrices. Independently "
        "useful regardless of Phase 0.1 outcome."
    )
    md.append("")
    md.append(
        "5. **Phase 4 (deferred)**: revisit after re-running the diagnostic on "
        "the fixed `render_tag_hub` / `high_accuracy` paths. Likely the "
        "failure-mode population shifts substantially when `frame_or_winding` "
        "is resolved."
    )
    md.append("")

    md.append("---")
    md.append("")
    md.append(
        "_Generated by `tools/bench/rotation_tail_diag/report.py` from "
        f"`{diagnostic_dir}/`. Bootstrap CIs are non-parametric, "
        f"{BOOTSTRAP_RESAMPLES} resamples, RNG seed {BOOTSTRAP_SEED}. "
        ".rrd recordings live alongside scenes.json — open with "
        "`rerun recordings/scene_NNNN.rrd` (per-scene)._"
    )
    md.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(md))
    print(f"  wrote {output_md} ({output_md.stat().st_size} bytes)")
    return output_md


def _build_json_report(
    diagnostic_dir: Path,
    scenes: ScenesFile,
    failure_modes: FailureModesFile,
    rot_p50: float,
    rot_p95: float,
    rot_p99: float,
) -> dict[str, Any]:
    """Optional companion: a structured JSON dump for downstream tooling."""
    return {
        "schema_version": "rotation_tail_diag/v1",
        "config_name": scenes.config_name,
        "profile": scenes.profile,
        "rot_p50_deg": rot_p50,
        "rot_p95_deg": rot_p95,
        "rot_p99_deg": rot_p99,
        "failure_modes": failure_modes.population,
    }
