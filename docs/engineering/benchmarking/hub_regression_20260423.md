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

## 1. Accuracy Baseline — `tag36h11` across resolutions (Accurate mode)

| Resolution | Images | Recall | Precision | RMSE (px) | Repro RMSE (px) | Trans P50 (m) | Rot P50 (°) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| 640×480 | 50 | 100.00% | 100.00% | 1.0795 | 0.9677 | 0.0033 | 0.549 | 3.65 |
| 1280×720 | 50 | 100.00% | 100.00% | 1.0161 | 0.8780 | 0.0024 | 0.254 | 9.28 |
| 1920×1080 | 50 | 100.00% | 100.00% | 1.3403 | 0.9294 | 0.0036 | 0.311 | 21.45 |
| 3840×2160 | 50 | 98.00% | 100.00% | 1.3296 | 1.0613 | 0.0068 | 0.297 | 74.08 |

Latency scales roughly linearly with pixel count (21.4 ms × 4 → 74 ms at 4K)
as expected for a preprocessing-dominated pipeline.

## 2. Pose-Mode Variant — `tag36h11` 1080p, Fast

| Mode | Recall | RMSE (px) | Repro RMSE (px) | Trans P50 (m) | Rot P50 (°) | Latency (ms) |
|---|---|---|---|---|---|---|
| Accurate | 100.00% | 1.3403 | 0.9294 | 0.0036 | 0.311 | 21.45 |
| Fast | 100.00% | 1.3403 | 4.8908 | 0.0073 | 2.002 | 18.70 |

Fast mode preserves recall and pixel RMSE but its pose accuracy is
substantially worse (5× reprojection error, 6× rotation error). The latency
saving is modest (~13%) — consistent with the LM solver not dominating a
50-tag 1080p frame. Fast mode is the right choice when downstream pose
consumers tolerate coarse rotation; otherwise prefer Accurate.

## 3. Refinement Variants — `tag36h11` 1080p

| Refinement | Recall | RMSE (px) | Repro RMSE (px) | Trans P50 (m) | Rot P50 (°) | Latency (ms) |
|---|---|---|---|---|---|---|
| None (default) | 100.00% | 1.3403 | 0.9294 | 0.0036 | 0.311 | 21.45 |
| GWLF | 100.00% | 0.9891 | 0.8502 | 0.0032 | 0.067 | 15.08 |
| HighAccuracy profile | 94.00% | 0.2029 | 0.1943 | 0.0004 | 0.054 | 10.62 |

GWLF improves pose quality (4.7× better rotation P50) and is, counter-
intuitively, *faster* than the default here because the weighted LM solver
consumes the GWLF covariances as priors and converges in fewer iterations.
HighAccuracy is even sharper (6× better translation, 5× better rotation)
but sheds 6 points of recall on the small-tag long tail — expected, since
the HighAccuracy profile enables `EdLines` which is known to collapse on
tags with < 1.5 pixels-per-bit.

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
default profile.

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
