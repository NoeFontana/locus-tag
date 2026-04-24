# Hub Regression Performance Report (2026-04-23)

Accompanies PR #198 (hub dataset rename + robustness suite). Captures
accuracy, pose metrology, and latency for every hub-backed regression test
after migration to the renamed `locus_v1_*` subsets and inclusion of four new
single-tag robustness variants (`tag16h5`, `high_iso`, `low_key`,
`raw_pipeline`).

## Environment

| Component | Value |
|---|---|
| CPU | AMD EPYC-Milan (4 cores / 8 threads) |
| L3 cache | 32 MiB |
| RAM | 32 GiB |
| Arch | x86_64 |
| Build profile | `--release --all-features --features bench-internals` |
| Threads | `cargo test -- --test-threads=1` (sequential) |

Latencies are `mean_total_ms` per frame as reported by the harness stdout
(the same value that is redacted as `[DURATION]` in the YAML snapshot for
stability). Accuracy figures come directly from the accepted `.snap` files.

## 1. Accuracy Baseline — `tag36h11` across resolutions (`high_accuracy` profile)

| Resolution | Images | Recall | Precision | RMSE (px) | Repro RMSE (px) | Trans P50 (m) | Rot P50 (°) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| 640×480 | 50 | 86.00% | 100.00% | 0.2449 | 0.1993 | 0.0005 | 0.119 | 2.02 |
| 1280×720 | 50 | 90.00% | 100.00% | 0.1873 | 0.1696 | 0.0004 | 0.064 | 4.86 |
| 1920×1080 | 50 | 94.00% | 100.00% | 0.2029 | 0.1943 | 0.0004 | 0.054 | 11.41 |
| 3840×2160 | 50 | 94.00% | 100.00% | 0.1676 | 0.1506 | 0.0008 | 0.049 | 43.50 |

`high_accuracy` is used as the baseline because its 4-6× tighter pose
bounds catch pose-solver regressions that `standard`'s wider thresholds
mask. The 6-14 points of recall ceded (vs `standard`'s 100%) is the cost
of `EdLines` + no-sharpen; small-tag recall signal is instead picked up
by the robustness suite (§ 5), which stays on `standard`.

Latency scales roughly linearly with pixel count (11.4 ms × 4 → 43.5 ms
at 4K) as expected for a preprocessing-dominated pipeline. `high_accuracy`
is also ~2× faster than `standard` because EdLines is cheaper than
ContourRdp.

## 2. Pose-Mode Variant — `tag36h11` 1080p, `standard` profile

| Mode | Recall | RMSE (px) | Repro RMSE (px) | Trans P50 (m) | Rot P50 (°) | Latency (ms) |
|---|---|---|---|---|---|---|
| Fast | 100.00% | 1.3403 | 4.8908 | 0.0073 | 2.002 | 18.70 |

The Fast-mode test stays on `standard` so that it contrasts cleanly against
the `standard`-profile Accurate metrics derivable from § 4 and the
robustness baseline. Fast mode preserves recall and pixel RMSE but its
pose accuracy is substantially worse than `standard`-Accurate (5×
reprojection error, 6× rotation error) at only ~13% latency saving —
consistent with the LM solver not dominating a 50-tag 1080p frame.

## 3. Refinement Variant — `tag36h11` 1080p, GWLF on `standard`

| Refinement | Recall | RMSE (px) | Repro RMSE (px) | Trans P50 (m) | Rot P50 (°) | Latency (ms) |
|---|---|---|---|---|---|---|
| GWLF | 100.00% | 0.9891 | 0.8502 | 0.0032 | 0.067 | 15.08 |

GWLF-on-`standard` is retained as a refinement variant since it's the only
`standard`-profile configuration with tight rotation P50 — useful for
catching regressions in the covariance-consuming weighted LM path without
paying the `high_accuracy` recall tax. Counter-intuitively it's *faster*
than `standard` default (15.1 ms vs 21.4 ms) because the weighted solver
converges in fewer iterations when fed GWLF priors.

The `high_accuracy`-profile 1080p test was removed from this module — it
is now the baseline at § 1.

## 4. Quad-Extraction Variants

`tag36h11` 720p:

| Config | Recall | RMSE (px) | Repro RMSE (px) | Latency (ms) |
|---|---|---|---|---|
| EdLines + no refinement | 90.00% | 0.1714 | 0.1599 | 7.46 |
| EdLines + GWLF | 90.00% | 0.5794 | 0.5804 | 9.21 |

`tag36h11` 1080p:

| Config | Recall | RMSE (px) | Repro RMSE (px) | Latency (ms) |
|---|---|---|---|---|
| Default + moments culling | 100.00% | 1.3403 | 0.9294 | 18.20 |
| EdLines | 96.00% | 0.5861 | 0.5809 | 24.14 |
| EdLines + moments culling | 96.00% | 0.5861 | 0.5809 | 23.76 |

Moments culling shaves ~3 ms off the default 1080p run (18.2 vs 21.4 ms)
without changing accuracy. Paired with EdLines it has no material effect —
EdLines' own filtering already rejects the same candidates.

## 5. Robustness Suite — NEW

Single-tag subsets stressing the detector under conditions the golden
`tag36h11` rendering does not exercise. All at 1920×1080, Accurate mode,
`standard` profile — the robustness suite stays on `standard` precisely
because it depends on the recall path that `high_accuracy` trades away.

| Subset | Family | Images | Recall | Precision | RMSE (px) | Trans P50 (m) | Rot P50 (°) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| `tag16h5` | AprilTag16h5 | 100 | 100.00% | **27.64%** | 1.0799 | 0.0057 | 0.463 | 16.96 |
| `high_iso` | AprilTag36h11 | 50 | 100.00% | 100.00% | 1.3361 | 0.0035 | 0.309 | 21.95 |
| `low_key` | AprilTag36h11 | 50 | **10.00%** | 100.00% | 0.2578 | 0.0029 | 0.459 | 15.24 |
| `raw_pipeline` | AprilTag36h11 | 50 | **58.00%** | 100.00% | 0.7594 | 0.0013 | 0.389 | 17.40 |

Observations (baselining new behaviour — each row is a forward-looking
watchlist, not a regression):

- **`tag16h5` — precision cliff.** 100% recall but only 27.6% precision.
  The 16h5 family has a dense codebook (30 codes in 16 bits) and rampant
  false-positive decodes are the dominant failure mode. This is a known
  property of the family rather than a detector bug; any future detector
  change that moves 16h5 precision should be treated as a *regression* if
  it drops, and a *win* if it climbs.
- **`low_key` — 10% recall.** Intentional: the dataset simulates extreme
  low-dynamic-range captures. Current adaptive thresholding leaves most
  tags buried in the black level. Surfacing this at 10% gives us a clear
  KPI for future contrast-robust threshold work.
- **`raw_pipeline` — 58% recall.** Raw-like pipeline variant exposes the
  detector to less-processed sensor output. Useful middle-ground target.
- **`high_iso` — no regression.** The detector's demosaic/denoise chain
  appears robust to synthetic high-noise captures (100% recall matches
  clean `tag36h11_1920x1080`).

### 5.1 Configuration-only tuned variants (follow-up)

Each subset above was re-run with a bespoke JSON profile (embedded at
`crates/locus-core/tests/fixtures/robustness/<subset>_tuned.json`) to
quantify how far the existing `DetectorConfig` knob set can move each KPI
without algorithmic changes. The baselines are retained as the KPI
watchlist; these rows are additive.

| Subset | Baseline | Tuned | Δ | Knobs that moved the number |
|---|---|---|---|---|
| `tag16h5` | R 100% / P **27.64%** / 16.96 ms | R 100% / P **68.85%** / 16.78 ms | **Precision ×2.5** at zero recall cost | `decoder.max_hamming_error: 2→1`, `decoder.min_contrast: 20→30`, `quad.min_edge_score: 4→8`, `quad.min_fill_ratio: 0.10→0.18` |
| `low_key` | R **10.00%** / P 100% / 15.24 ms | R **22.00%** / P 100% / 28.08 ms | **Recall ×2.2**, latency ×1.8 | `threshold.tile_size: 8→2`, `threshold.min_range: 10→0`, `threshold.constant: 0→-20` |
| `raw_pipeline` | R **58.00%** / P 100% / 17.40 ms | R **60.00%** / P 100% / 19.05 ms | Recall +2 pts (marginal) | `quad.min_area: 16→8`, `quad.min_edge_length: 4→2`, `quad.min_edge_score: 4→1` |

Findings:

- **`tag16h5` is the clear configuration win.** Halving the allowed
  Hamming distance is the dominant lever; raising the decoder's contrast
  gate and the upstream quad edge-score/fill-ratio gates trim the
  remaining false-positive pool without sacrificing a single true positive.
  A `+41.2` precision-point lift at **zero** recall cost is the ceiling we
  could find; pushing further (`min_edge_score: 10`, `min_fill_ratio:
  0.22`) recovers ~3 more precision points at the cost of 1-2 recall
  points — not a trade we took. The remaining ~31% of false positives are
  dense structural ambiguities in the 16h5 codebook itself and are no
  longer configuration-bound.
- **`low_key` has a configuration ceiling at 22% recall.** A finer
  adaptive-threshold grid (`tile_size=2`) with zero `min_range` and a
  strongly negative `constant` unlocks the tags buried in the low-DR
  black level. A full sweep (`tile_size ∈ {2,4,8,16}`, `min_range ∈
  {0,2,5,10}`, `constant ∈ {0,-3,-5,-10,-15,-20,-30,-50}`, plus decoder
  and quad-gate relaxation, plus structure-tensor radius and adaptive
  window) converges to the same 22% ceiling. Confirms the report's
  framing: this subset is *the* KPI for future contrast-robust threshold
  work.
- **`raw_pipeline` is largely config-bound at its baseline.** Threshold
  and decoder relaxation do not move the needle; only quad-gate
  relaxation adds a marginal +2 recall points. `EdLines` extraction gives
  the same recall at materially better RMSE (0.33 vs 0.76 px) but we did
  not swap the default in this batch because the suite's baseline is
  `ContourRdp` and switching extractors is a separate regression axis.
  The remaining 40% are likely noise-fragmented contours that would
  require a pre-filtering pass to recover.

## 6. Distortion — AprilGrid

All 1920×1080, 50 images per subset, AprilTag36h11, Accurate pose mode.
Pose-mode coverage lives in § 2; this suite isolates the undistortion path.

| Model | Recall | Precision | RMSE (px) | Repro RMSE (px) | Trans P50 (m) | Rot P50 (°) | Latency (ms) |
|---|---|---|---|---|---|---|---|
| Brown–Conrady | 92.93% | 99.43% | 1.0947 | 1.1336 | 0.0634 | 2.151 | 26.52 |
| Kannala–Brandt | 81.30% | 98.97% | 1.5204 | 3.0482 | 0.0926 | 7.313 | 24.76 |

Kannala–Brandt (fisheye) recall sits ~11 points below Brown–Conrady at
equivalent pixel RMSE, confirming that the wider field of view pushes a
larger fraction of tags into the distortion-heavy periphery where the
detector's undistorted assumptions cost yield.

## 7. Board-Level (reference — dataset unchanged)

Included for completeness. These snapshots pre-date PR #198 and were not
regenerated; the board datasets themselves were not re-rendered.

| Board | Frames | Frames with board | Mean Trans Err (m) | Mean Rot Err (°) | Mean Coverage |
|---|---|---|---|---|---|
| `charuco_golden_v1_1920x1080` | 150 | 149 | 0.0081 | 1.929 | 99.0% |
| `aprilgrid_golden_v1_1920x1080` | 145 | 145 | 0.0108 | 2.035 | 95.0% |

## Reproduction

```bash
TRACY_NO_INVARIANT_CHECK=1 \
LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
cargo test --release \
  --test regression_render_tag \
  --test regression_render_tag_robustness \
  --test regression_distortion_hub \
  --test regression_board_hub \
  --features bench-internals \
  -- --test-threads=1 --nocapture
```

The `regression_render_tag_robustness` suite includes both the baseline
watchlist runs (`regression_hub_<subset>_1080p`) and the configuration-
tuned follow-up runs (`regression_hub_<subset>_1080p_tuned`). The tuned
variants load their JSON profiles from
`crates/locus-core/tests/fixtures/robustness/*_tuned.json`.
