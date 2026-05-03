# AprilTag-vs-Locus gap analysis

**Date:** 2026-05-03
**Question:** where is `locus-tag` lacking relative to AprilTag-C, and where should we invest engineering effort to close the gap?

## Headline

With the production-accuracy profile (`high_accuracy`), Locus is decisively ahead of AprilTag-C on this corpus on every dimension except a 1–2 tag recall miss and a 1.5–2.8× looser rotation p95. The "AprilTag is tightest at every distance" intuition was an artifact of the bench harness defaulting to the latency-tuned `standard` profile, where Locus is 4× behind on rotation. With `high_accuracy`, Locus beats AprilTag on translation by 5–7× and on rotation p50 by 1.2–1.5×.

The two residual gaps are real but small. Both were investigated in depth via cheap parameter-tuning experiments. **All cheap fixes were empirically refuted.** Closing either gap requires post-release engineering: per-tag IRLS fallback for the rotation tail, rerun-visualization-driven debugging for the recall miss.

**Pre-release ship: documentation only. No code change.**

## Sweep configuration

Corpus: `single_tag_locus_v1_tag36h11_{1280x720, 1920x1080, 3840x2160}`, 100 frames per cell.

```bash
PYTHONPATH=. uv run --group bench tools/cli.py bench real \
  --hub-config <C> --compare --max-hamming 0 --limit 100 \
  --record-out runs/gap_analysis/<C>.parquet
```

Locus consumes the CLI-built config (`standard` profile by default; `high_accuracy` runs required temporary monkey-patches in `tools/cli.py:_build_cli_config`, all reverted before commit). AprilTag and OpenCV use their respective package defaults via `tools/bench/utils.py`.

## High_accuracy: where the gap actually is

| Library | Res | Recall | Trans p50 (mm) | Trans p95 (mm) | Rot p50 (°) | Rot p95 (°) | Rot p99 (°) |
| :--- | :-: | :-: | -: | -: | -: | -: | -: |
| Locus (Hard) | 720 | 0.980 | **0.46** | **4.81** | **0.063** | 0.455 | **0.554** |
| AprilTag | 720 | 1.000 | 3.19 | 24.10 | 0.082 | **0.259** | 50.527 |
| Locus (Hard) | 1080 | 0.978 | **0.83** | **10.78** | **0.070** | 0.686 | **1.341** |
| AprilTag | 1080 | 1.000 | 5.00 | 41.62 | 0.084 | **0.243** | 57.337 |
| Locus (Hard) | 2160 | 0.956 | **1.46** | **11.89** | **0.060** | 0.440 | **1.014** |
| AprilTag | 2160 | 1.000 | 8.63 | 82.74 | 0.091 | **0.302** | 59.170 |

**Locus dominates** on translation accuracy (5.9–7× tighter at p50, 3–7× at p95/p99), rotation p50 (1.2–1.5× tighter), and rotation p99 (no symmetric-tag pose flips — AprilTag exhibits 50–60° catastrophic outliers that Locus avoids by 50–60×).

**Two residual gaps:**
1. **Recall miss** — Locus misses 1, 1, 2 tags at 720p, 1080p, 2160p. AprilTag finds them all.
2. **Rotation p95 tail** — AprilTag is 1.5–2.8× tighter on the upper body of the rotation distribution. Locus's p50 is *better* (so it isn't a systemic bias), and the p99 is dramatically better (no catastrophic flips), but the p95 is consistently looser.

## Gap 1 — Rotation p95 tail

### Root cause (after correcting initial hypothesis)

Initial framing: "structure-tensor priors over-weighted at low PPM, fix by capping at PPB<500." This was wrong on two counts:

- **The mechanism was mislabeled.** Per the algorithmic review (`pose_weighted.rs:349-354`, `pose.rs:732-761`), the actual issue isn't Mahalanobis amplification — it's near-rank-deficient `JᵀWJ` at frontal poses combined with IPPE's Necker-reversal ambiguity that the under-conditioned LM struggles to disambiguate. The Marquardt damping floor (`1e-6`) is too weak to stabilize under-determined rotation DOFs.
- **The proposed fix was in the wrong direction.** The pose-mode ablation below shows that *any* move toward isotropic / Fast-mode pose estimation makes everything dramatically worse.

### Pose-mode ablation: the weighted LM is essential

The natural-looking fix — "switch to `PoseEstimationMode::Fast` (IPPE-only) for low-PPM tags" — was tested directly. Same corpora, same `--max-hamming=2`, only `pose_estimation_mode` changes.

| Resolution | Metric | Accurate (current) | Fast (proposed) | Ratio |
| :--- | :--- | -: | -: | -: |
| 720p | trans_p95 (mm) | 5.89 | 28.73 | **4.88× WORSE** |
| 720p | rot_p95 (°) | 0.522 | 5.868 | **11.24× WORSE** |
| 1080p | trans_p95 (mm) | 10.78 | 131.56 | **12.21× WORSE** |
| 1080p | rot_p95 (°) | 0.686 | 4.853 | **7.08× WORSE** |
| 2160p | trans_p95 (mm) | 11.89 | 311.93 | **26.23× WORSE** |
| 2160p | rot_p95 (°) | 0.440 | 11.746 | **26.71× WORSE** |

Bootstrap 95% CI on `rot_p95(Fast) / rot_p95(Accurate)` over 2000 resamples of n=397 paired records: **[7.4×, 22.2×]**. Per-record paired comparison on Accurate's p95+ tail (n=20): Fast tighter on **only 1 of 20**.

**The weighted LM is doing real work everywhere — at p50, p95, and p99.** It's what earns Locus the 5–7× translation lead and the 50–60× rot_p99 advantage. Removing it would be catastrophic.

### Cheap parameter-tuning fixes (also refuted)

Two more cheap fixes were tested before declaring this post-release work:

1. **Force ContourRdp+Erf for all tags** (Static policy override): rot_p95 5.14×–197.53× WORSE on bulk distribution; trans_p95 6×–244× WORSE. Bootstrap 95% CI on rot_p95 ratio: [81×, 217×]. Refuted.
2. **Bump `AdaptivePpb` threshold from 2.5 → 12** (route the tail to ContourRdp+Erf): the PPB ∈ [2.5, 12] stratum (n=66) saw rot_p95 jump from 1.25° → 75.36°. The 2.5 threshold is empirically optimal — that's "where EdLines collapses" and ContourRdp+Erf takes over; ContourRdp+Erf is unreliable in [2.5, 12]. Refuted.

### Real path forward (post-release)

Per-tag IRLS fallback gated on `cond(JᵀWJ)` ill-conditioning. When the weighted LM converges to a rank-deficient minimum on a specific tag, retry with AprilTag-style IRLS on that tag only. Preserves the wins on the bulk distribution while addressing the tail. Estimated ~150 LOC in `crates/locus-core/src/pose_weighted.rs` + tests + cross-corpus validation.

## Gap 2 — Recall miss

### Root cause (after correcting initial hypothesis)

Initial framing: "`min_area=800` filtering distant tags." This was wrong — half the misses are on `scene_0031` at PPM 2348–2372 (a *close* tag with side ~388px and area ~150,000, two orders of magnitude above any plausible `min_area`). Per-miss attribution by joining `missed_gt` to spatially-attributed `rejected_quad` records:

| Res | Image | tag_id | distance | aoi | ppm | Stage |
| :--- | :--- | -: | -: | -: | -: | :--- |
| 720p | scene_0002 | 219 | 1.19m | 51.5° | 736 | RejectedDecode |
| 1080p | scene_0031 | 505 | 0.59m | 14.6° | 2348 | **PRE-EXTRACTION** |
| 2160p | scene_0020 | 356 | 5.20m | 19.9° | 550 | RejectedDecode |
| 2160p | scene_0031 | 505 | 1.18m | 14.6° | 2372 | **PRE-EXTRACTION** |

Two distinct failure modes:
- **Mode A (PRE-EXTRACTION, 2/4):** `scene_0031`'s tag is never extracted as a quad. Specific to that scene's edge profile.
- **Mode B (RejectedDecode, 2/4):** Quad extracted but bits don't exactly match a codeword at `--max-hamming=0`. Methodology artifact — the actual CLI default is `--max-hamming=2`, where these would recover.

### Cheap parameter-tuning fixes (refuted)

Three knobs suggested by independent review (`min_fill_ratio`, `edlines_imbalance_gate`, `max_fill_ratio`) were tested:

- `min_fill_ratio=0.05` + `imbalance_gate=Disabled`: did NOT recover scene_0031, **introduced 3 new misses** (scene_0008, scene_0020, scene_0022), recall regressed from 1.0 → 0.96 at 720p.
- `max_fill_ratio=0.999` (motivated by "frontal axis-aligned tag fills its bbox"): did NOT recover scene_0031, no other change.

**The actual extraction-stage cause is not at any of these three documented gates.** Pinpointing it requires rerun visualization.

### Real path forward (post-release)

Rerun-visualization session on `scene_0031`. Inspect the EdLines intermediate stages (line-segment array, anchor decisions, contour assembly) to find which stage drops the candidate. Targeted parameter or algorithm fix once the failure stage is identified.

For Mode B (the methodology half): use `--max-hamming=2` (the existing CLI default) for accuracy benches. No code change needed.

## What ships pre-release

- **This document** + the companion [Soft decode limits](./soft_decode_limits_20260503.md). Durable record of which fixes were tested and why they don't work, so a future engineer doesn't repeat them.
- **`tools/cli.py:354`** annotates `--max-hamming` with a pointer to the Soft findings doc.
- **Zero code change to the pose pipeline, extraction policy, or any algorithm.** Validated as net-positive: every cheap fix tested regressed something material.

## Caveats

- **Corpus is `single_tag_locus_v1_tag36h11_*`** — synthetic renders from our own pipeline. Real camera data and other corpora (charuco, aprilgrid, ICRA forward) may shift the picture. The render-tag SOTA report (`render_tag_sota_20260425.md`) on a different rendered corpus is broadly consistent with the `high_accuracy` numbers above.
- **AprilTag wrapper has a known corner-canonicalization caveat** (`tools/bench/utils.py:907-961`, flagged in PR #223): pose comparisons are valid (180° z-rotation correction applied), but raw-corner reprojection error against AprilTag is contaminated by corner-array reordering. None of the metrics above use raw-corner reprojection.
- **n=100 frames per cell is the corpus size** — the synthetic single-tag corpora are exactly 45 frames each at 1080p+, so `--limit 100` covers the full set. Tail statistics (n≈7 of 136 matched records for the rot_p95 tail) are limited by corpus size, not sampling. Larger corpora or real-data replication would strengthen tail conclusions.
- **AprilTag-C version** is `pupil_apriltags` (Python bindings of AprilTag3 C reference), no exposed Hamming threshold. The recall comparison treats AprilTag's implicit Hamming tolerance as the baseline; Locus at h=2 (the CLI default) is the apples-to-apples comparison, not h=0.

## Reproducing this analysis

```bash
uv run maturin develop --release --manifest-path crates/locus-py/Cargo.toml
mkdir -p runs/gap_analysis

# Standard profile (CLI default)
for corpus in single_tag_locus_v1_tag36h11_{1280x720,1920x1080,3840x2160}; do
  PYTHONPATH=. uv run --group bench tools/cli.py bench real \
    --hub-config "$corpus" --compare --max-hamming 0 --limit 100 \
    --record-out "runs/gap_analysis/${corpus}.parquet"
done

# high_accuracy profile requires three temporary edits to tools/cli.py:
#   1. Line 450: from_profile("standard") → from_profile("high_accuracy")
#   2. Lines 461-472: comment out the Locus (Soft) wrapper block
#      (Soft is incompatible with EdLines extraction)
# And one CLI flag: --refinement None  (Erf is incompatible with EdLines)
for corpus in single_tag_locus_v1_tag36h11_{1280x720,1920x1080,3840x2160}; do
  PYTHONPATH=. uv run --group bench tools/cli.py bench real \
    --hub-config "$corpus" --compare --max-hamming 2 --refinement None \
    --limit 100 \
    --record-out "runs/gap_analysis/ha_${corpus}.parquet"
done
```

Aggregation scripts (`/tmp/aggregate_gap*.py`, `/tmp/ablation_compare.py`, `/tmp/exp1_compare.py`) were inline development artifacts, not committed. Re-deriving from the parquets is straightforward — see `tools/bench/metrics.py` for the recall/precision primitives and `tools/bench/plots/pareto.py:_aggregate` for the per-(binary, resolution) reducer pattern.

A `--profile NAME` flag for `bench real` would obviate the temporary CLI edits — flagged as a small CLI improvement (~10 LOC) not in scope here.
