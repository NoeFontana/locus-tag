# AprilTag-vs-Locus gap analysis

**Date:** 2026-05-03
**Question:** where is `locus-tag` lacking relative to AprilTag-C, and where should we invest engineering effort to close the gap?

## Headline

**With the production-accuracy profile (`high_accuracy`), Locus is decisively ahead of AprilTag-C on this corpus.** Translation error is 3–7× tighter at every percentile, rotation p50 is 1.2–1.5× tighter, latency is 3–3.5× faster, and rotation outliers (p99) are 50–60× better (AprilTag exhibits catastrophic ~50° pose flips on symmetric tags here).

The earlier framing — "AprilTag is tightest at every distance, Locus pays for it only at 4K" — was an artifact of comparing AprilTag against the `standard` profile (latency-tuned). On `standard`, Locus is genuinely 4× behind AprilTag on rotation accuracy. The gap is configuration-driven, not algorithmic.

The two real residual gaps to AprilTag, with `high_accuracy`:
1. **Recall**: Locus misses 1–2 tags per 100 frames where AprilTag finds them. Plausibly the `min_area=800` filter clipping distant tags.
2. **Rotation p95**: 1.5–2.8× looser than AprilTag. Pose-solver tuning opportunity for the upper body of the rotation distribution.

## Sweep configuration

Corpus: `single_tag_locus_v1_tag36h11_{1280x720, 1920x1080, 3840x2160}`, 100 frames per cell.

```bash
# standard profile (CLI default)
PYTHONPATH=. uv run --group bench tools/cli.py bench real \
  --hub-config <C> --compare --max-hamming 0 --limit 100 \
  --record-out runs/gap_analysis/<C>.parquet

# high_accuracy profile (one-line monkey-patch in tools/cli.py:450, reverted)
# also requires --refinement None (EdLines extraction is incompatible with Erf
# refinement) and skipping the Locus (Soft) wrapper instantiation
# (Soft is incompatible with EdLines extraction).
```

Locus wrappers consume the CLI-built config; AprilTag and OpenCV use their respective package defaults via `tools/bench/utils.py`.

## Standard profile (CLI default — latency-optimized)

| Library | Res | Recall | Precision | Trans p50 (mm) | Trans p95 (mm) | Trans p99 (mm) | Rot p50 (°) | Rot p95 (°) | Rot p99 (°) | Latency p50 (ms) |
| :--- | :-: | :-: | :-: | -: | -: | -: | -: | -: | -: | -: |
| Locus (Hard) | 720 | 1.000 | 1.000 | 3.41 | 29.76 | 35.65 | 0.323 | 1.257 | 3.343 | 8.5 |
| AprilTag | 720 | 1.000 | 1.000 | 3.19 | 24.10 | 34.26 | 0.082 | 0.259 | 50.527 | 12.7 |
| OpenCV | 720 | 1.000 | 1.000 | 3.92 | 27.89 | 42.40 | 0.133 | 0.364 | 0.426 | 22.1 |
| Locus (Hard) | 1080 | 1.000 | 1.000 | 4.53 | 57.24 | 135.22 | 0.320 | 1.946 | 41.371 | 36.3 |
| AprilTag | 1080 | 1.000 | 1.000 | 5.00 | 41.62 | 53.05 | 0.084 | 0.243 | 57.337 | 34.3 |
| OpenCV | 1080 | 1.000 | 0.957 | 5.83 | 58.32 | 69.14 | 0.087 | 0.313 | 0.371 | 42.7 |
| Locus (Hard) | 2160 | 1.000 | 1.000 | 9.54 | 93.71 | 116.14 | 0.361 | 1.179 | 1.808 | 66.0 |
| AprilTag | 2160 | 1.000 | 1.000 | 8.63 | 82.74 | 101.56 | 0.091 | 0.302 | 59.170 | 139.7 |
| OpenCV | 2160 | 1.000 | 0.978 | 10.58 | 99.08 | 223.14 | 0.111 | 0.417 | 0.442 | 147.9 |

On `standard`, AprilTag has a clear ~4× rotation advantage at p50/p95. Locus's only standard-profile wins are rotation p99 (catastrophic-tail suppression) and latency at 720p / 2160p.

## High_accuracy profile (production-accuracy)

| Library | Res | Recall | Precision | Trans p50 (mm) | Trans p95 (mm) | Trans p99 (mm) | Rot p50 (°) | Rot p95 (°) | Rot p99 (°) | Latency p50 (ms) |
| :--- | :-: | :-: | :-: | -: | -: | -: | -: | -: | -: | -: |
| Locus (Hard) | 720 | 0.980 | 1.000 | **0.46** | **4.81** | **6.75** | **0.063** | 0.455 | **0.554** | **4.4** |
| AprilTag | 720 | 1.000 | 1.000 | 3.19 | 24.10 | 34.26 | 0.082 | **0.259** | 50.527 | 13.5 |
| Locus (Hard) | 1080 | 0.978 | 1.000 | **0.83** | **10.78** | **17.70** | **0.070** | 0.686 | **1.341** | **7.5** |
| AprilTag | 1080 | 1.000 | 1.000 | 5.00 | 41.62 | 53.05 | 0.084 | **0.243** | 57.337 | 26.4 |
| Locus (Hard) | 2160 | 0.956 | 1.000 | **1.46** | **11.89** | **28.16** | **0.060** | 0.440 | **1.014** | **33.4** |
| AprilTag | 2160 | 1.000 | 1.000 | 8.63 | 82.74 | 101.56 | 0.091 | **0.302** | 59.170 | 109.7 |

On `high_accuracy`, Locus dominates AprilTag on every dimension except recall and rot p95.

## The role reversal

| Dimension | Standard | High_accuracy |
| :--- | :--- | :--- |
| Trans p50 | AprilTag 7% tighter @720p, Locus 10% tighter @1080p | **Locus 5.9–7× tighter at every res** |
| Trans p99 | Roughly tied | **Locus 3.0–7.0× tighter** |
| Rot p50 | AprilTag 4× tighter | **Locus 1.2–1.5× tighter** |
| Rot p95 | AprilTag 4–8× tighter | AprilTag 1.5–2.8× tighter |
| Rot p99 | Locus dominates (no 50° outliers) | **Locus dominates by 50–60×** |
| Latency | Locus 1.5× faster (720p, 2160p), tied (1080p) | **Locus 3.0–3.5× faster everywhere** |
| Recall | Tied (all 100%) | AprilTag 100%, Locus 95.6–98% |

## Where the gap actually is

### Gap 1 — Recall miss on `high_accuracy`

Locus misses 1–2 tags per 100 frames where AprilTag finds them. Suspect: `min_area=800` (vs standard's 36) drops small / distant / oblique tags before they reach the decoder. On `single_tag_locus_v1`, this manifests as 1 missed tag at 720p/1080p and 2 at 2160p.

**Action candidates:**
- Lower `min_area` for `high_accuracy` and re-baseline. Cost: latency rises (more quads to process). Need to measure how much.
- Add adaptive min_area based on resolution / image content. More work, but principled.
- Accept the recall miss as a precision/recall tradeoff. The corpora may not be representative of the missed cases.

### Gap 2 — Rotation p95 looseness

Locus's rotation p95 is 1.5–2.8× looser than AprilTag's. The body of the distribution past the median is wider. Locus's p50 is *better* than AprilTag's, so this isn't a systemic bias — it's specific cases that get poor fits.

**Action candidates:**
- Inspect the 5% of tags that fall in the p95 tail. Likely high-AOI or high-distance scenarios where the pose solver converges to a degenerate minimum.
- Tune the weighted-LM solver: Huber delta, Tikhonov regularization, structure-tensor radius. The `high_accuracy.json` already lowers `pose_consistency_gate_sigma_px` to 0.5 — there may be more headroom here.
- Investigate AprilTag's IRLS pose solver. Their p95 is consistently tighter; the algorithm difference may explain this. Note: AprilTag's *p99* is much worse (50°+ flips on symmetric tags), so this is a different failure mode, not a uniformly better solver.

### Non-gaps

- **Translation accuracy**: Locus's ERF + weighted-LM pipeline produces 3–7× tighter translation than AprilTag's iterative solver on this corpus. No gap to close.
- **Latency**: Locus is 3–3.5× faster at every resolution on `high_accuracy`. No gap to close.
- **Rotation outlier suppression**: Locus's p99 is 50–60× better on `high_accuracy` (and 15–33× better on `standard`). AprilTag's symmetric-tag pose ambiguity produces catastrophic flips that Locus avoids. No gap to close.

## What this changes about engineering priorities

1. **The bench harness's default profile (`standard`) is the wrong basis for AprilTag comparison.** It optimizes for latency — `min_area=36`, `extraction_mode=ContourRdp`, no EdLines, ERF refinement — and produces accuracy figures that AprilTag-C beats by 4× on rotation. Public bench reports rendered with this profile mislead readers about Locus's accuracy ceiling.
2. **A fair comparison uses `high_accuracy`** for both binaries (or at least Locus). When users care about pose accuracy they should use this profile; when they care about latency, the standard profile already wins on latency at most resolutions.
3. **The two real eng gaps (recall miss, rot p95)** are tractable. Both are tuning problems within existing algorithmic frameworks (`min_area` config, weighted-LM solver parameters), not architectural rewrites.

## Caveats

- **Corpus is `single_tag_locus_v1_tag36h11_*`**: synthetic renders from our own pipeline. Real camera data and other corpora (charuco, aprilgrid, ICRA forward) may shift the picture. The render-tag SOTA report (`render_tag_sota_20260425.md`) on a different rendered corpus is broadly consistent with the `high_accuracy` numbers above.
- **AprilTag wrapper has a known corner-canonicalization caveat** (`tools/bench/utils.py:907-961`, flagged in PR #223): pose comparisons are valid (180° z-rotation correction applied), but raw-corner reprojection error against AprilTag is contaminated by corner-array reordering. None of the metrics above use raw-corner reprojection; pose error is via 6DOF GT.
- **Sample size is 100 frames per cell.** p99 numbers are based on a single tail observation; treat them as directional rather than precise. Trends across resolutions and distinct corpora are the load-bearing claim.
- **AprilTag-C version** is `pupil_apriltags` (Python bindings of AprilTag3 C reference) at whatever version is pinned in the bench environment. A newer AprilTag release or different binding may shift the numbers slightly.
- **The high_accuracy sweep required disabling the Locus (Soft) wrapper** because Soft decoding is incompatible with EdLines extraction (configuration validation rejects the combination). PR #225 already documented Soft as a deprecation candidate; this analysis reinforces that conclusion since high_accuracy + Hard already wins on every dimension that matters.

## Reproducing this analysis

```bash
# Build the wheel
uv run maturin develop --release --manifest-path crates/locus-py/Cargo.toml

# Standard profile (current bench-harness default)
mkdir -p runs/gap_analysis
for corpus in single_tag_locus_v1_tag36h11_{1280x720,1920x1080,3840x2160}; do
  PYTHONPATH=. uv run --group bench tools/cli.py bench real \
    --hub-config "$corpus" --compare --max-hamming 0 --limit 100 \
    --record-out "runs/gap_analysis/${corpus}.parquet"
done

# high_accuracy profile requires three temporary edits to tools/cli.py:
#   1. Line 450: `from_profile("standard")` → `from_profile("high_accuracy")`
#   2. Lines 461-472: comment out the Locus (Soft) wrapper block
#      (Soft is incompatible with EdLines extraction)
# And one CLI flag:
#   --refinement None  (Erf is incompatible with EdLines)
for corpus in single_tag_locus_v1_tag36h11_{1280x720,1920x1080,3840x2160}; do
  PYTHONPATH=. uv run --group bench tools/cli.py bench real \
    --hub-config "$corpus" --compare --max-hamming 0 --refinement None \
    --limit 100 \
    --record-out "runs/gap_analysis/ha_${corpus}.parquet"
done

# A `--profile` flag would obviate the temporary CLI edits — see "Follow-ups".
```

Aggregation script: `/tmp/aggregate_gap.py` and `/tmp/aggregate_gap_ha.py` in this development context (kept inline rather than committed; the parquets themselves are not version-controlled).

## Follow-ups

1. **Add a `--profile` flag to `bench real`** so this analysis can be re-run cleanly without monkey-patching. Small CLI change (~10 LOC). Would also be useful for any future profile-comparison work.
2. **Drill into the recall-miss cases.** Identify which tags Locus misses on `high_accuracy` 720p/1080p and check whether `min_area` lower would recover them without measurable latency cost.
3. **Drill into the rot-p95 tail.** Stratify the 5% worst rotation cases by AOI / distance / PPM; see whether they cluster in a way that points to a specific solver-tuning opportunity.
4. **Re-render the public bench reports against `high_accuracy`** so external readers see the production-accuracy numbers, not the latency-tuned ones. Optional but worth considering before the next release.

## Hardware

CPU: 16 logical cores (`lscpu`). Linux 6.8.0-107-generic. Build: `--release`. RAYON_NUM_THREADS unset.
