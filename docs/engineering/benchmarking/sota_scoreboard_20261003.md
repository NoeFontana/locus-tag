# SOTA scoreboard checkpoint (2026-10-03)

Live snapshot, one day after the [2026-10-02 baseline](sota_scoreboard_20261002.md). Since then
`main` has gained decode-first ordering (#416), a recovery window sized by false-trigger
probability with a dispatched codebook search (#417), and exact latency work (#418, #420–#424).
This page scores two configurations of `main` `a6199d4` against the **best published operating
point of each reference**. Reproduce any cell with `cargo xtask sota` (`xtask/README.md`).

!!! note "EuRoC corner cells carry a known measurement bias (fixed 2026-10-07)"

    The EuRoC `LOO px` figures on this page were scored with an **inexact**
    undistortion inverse (`cv2.undistortPoints`, which stops short with no convergence
    test); see PR #455.

    Across the fourteen EuRoC runs whose detections are preserved, the correction is a
    near-fixed **absolute** shift of -0.005 to +0.011 px. It is therefore largest in
    *relative* terms for the most accurate detector -- +4.0 % at 0.27 px, +2.7 % at 0.32 px,
    +0.75 % at 0.48 px, +0.3 % at 0.98 px, and indistinguishable from zero above 2 px --
    which is exactly why it flattered the best corners.

    At the 0.53-0.91 px levels in the EuRoC rows the shift is about +0.004 px on both Locus
    and the reference, so **no win/loss verdict changes**.

    The runs behind this page were not archived, so these cells are **not** re-derived here:
    a figure rescaled by a factor measured on *other* runs would not be a measurement, and the
    common-tag set each cell used depended on which runs shared the directory at the time,
    which is not recoverable. Re-running `cargo xtask sota` today produces corrected values.
    The one page whose detections survive is
    [EuRoC SOTA (2026-10-04)](euroc_sota_20261004.md), re-derived exactly.

    What the metric measures -- and why it must not be compared against a render-tag RMSE --
    is in the [EuRoC error budget](euroc_error_budget.md).

- **Candidate** is the opt-in robust pipeline: `standard` plus these overrides.

  ```json
  {"threshold": {"mode": "LocalMean", "local_mean_radius": 7, "noise_k": 4.0, "enable_sharpening": false},
   "quad": {"min_fill_ratio": 0.0, "min_density": 0.0, "max_elongation": 0.0, "refine_before_decode": false},
   "segmentation": {"connectivity": "Four"},
   "decoder": {"max_border_error_rate": 0.0}}
  ```

  It runs through `sota run --detectors "locus:<name>=standard+<file>.json"`.
- **`standard`** is the shipped profile, unchanged.

**Result:**

| Configuration | Judged cells won (of 88) |
| :-- | --: |
| Candidate | 37 |
| `standard` | 32 |
| `standard` at the baseline | 31 |

At 8 threads the candidate is faster than aruco_nano on Liu4K, render-tag 1080p and render-tag
4K, and within 0.3 ms of it on ICRA forward.

Pose metrics are not reported: the table is corner-level, so the Fast/Accurate pose-mode rule
does not apply.

## Setup

| Item | Value |
| :--- | :--- |
| CPU | AMD EPYC-Milan, 8 vCPU (4 cores × 2 threads), AVX2, KVM guest (`lscpu`, measurement session) |
| OS / toolchain | Linux 6.8.0-139-generic, rustc 1.92.0 |
| Locus | `maturin develop --release` at `main` `a6199d4` |
| References | OpenCV 4.10.0, minimal build: `CORNER_REFINE_NONE` / `SUBPIX` / `APRILTAG`, `errorCorrectionRate = 0`. aruco_nano `961b18b`. Both unpatched |
| Accuracy | Full datasets, `--jobs 4` (detectors run in parallel, so latency from these runs is not judged) |
| Latency | Separate serial runs `<benchmark>@t1` and `@t8`: `--jobs 1`, `--threads N` (`RAYON_NUM_THREADS=N` / `cv::setNumThreads(N)`), best of 2 per image, image decode excluded. Strides: Liu4K 8, EuRoC 15, ICRA forward 2, circle 4, random 8, render-tag 2, tag16h5 4, boards 6 |
| Load | 1.6–2.6 load average during the timing runs, from another session's builds |

## Latency at 1 and 8 threads (ms/img)

| Benchmark | Candidate 1T | Candidate 8T | `standard` 1T | `standard` 8T | aruco_nano 1T | aruco_nano 8T | Best OpenCV 1T | Best OpenCV 8T |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| liu4k | 89.3 | **42.2** | 143.2 | 76.6 | 53.9 | 53.4 | 318.7 | 137.7 |
| icra-forward | 31.8 | 10.6 | 51.4 | 20.4 | 10.5 | 10.3 | 32.6 | 16.4 |
| hub-1080p | 15.5 | **6.5** | 21.6 | 12.1 | 7.6 | 8.1 | 55.3 | 27.3 |
| hub-4k | 52.8 | **21.6** | 87.8 | 46.0 | 31.3 | 31.3 | 200.4 | 96.4 |

- aruco_nano is single-threaded, so its 8T column only re-measures 1T.
- **Since the baseline**, `standard` at 1T went from 194.6 → 143.2 ms on Liu4K, 98.0 → 51.4 ms
  on ICRA forward and 153.7 → 87.8 ms on 4K. The gain comes from the exact latency PRs, which
  leave output unchanged.
- **The candidate** saves a further 38–40 % on these four benchmarks by refining only decoded
  candidates.
- **1T is still lost** on every benchmark except EuRoC. The remaining Liu4K 1T time is spread
  evenly over:
  - segmentation, ~28 ms: runs, union, resolve and stats, group;
  - quad extraction, ~28 ms: trace, chain, vertex selection, gate;
  - the threshold, ~12 ms;
  - decoding, ~8 ms.

  No single exact change is left that would close the 1.7× gap. See
  [what is left](#what-is-left) below.

## Win table

Locus value against the best reference value per metric; a tie is a win.

- Corner error is order-preserving, in the OpenCV pixel convention.
- It is measured on the tags that every reference with ≥ 20 % recall matched
  (`tools/bench/sota/score.py`).
- Latency cells are the 1T runs.

| Benchmark | Metric | Candidate | | `standard` | | Best reference | Value |
| :-- | :-- | --: | :-: | --: | :-: | :-- | --: |
| liu4k | Recall % | 66.56 | ✅ | 29.12 | ❌ | aruco_nano | 66.27 |
| liu4k | Precision % | 99.97 | ❌ | 99.70 | ❌ | aruco_nano | 100.00 |
| liu4k | F1 | 79.91 | ✅ | 45.07 | ❌ | aruco_nano | 79.71 |
| liu4k | Corner px, common tags (median) | 0.716 | ❌ | 0.686 | ❌ | aruco_nano | 0.473 |
| liu4k | ms/img | 89.3 | ❌ | 143.2 | ❌ | aruco_nano | 53.9 |
| euroc | Recall % | 87.06 | ✅ | 20.43 | ❌ | opencv_apriltag | 44.55 |
| euroc | Precision % | 98.13 | ❌ | 97.99 | ❌ | opencv_apriltag | 99.65 |
| euroc | LOO px, common tags (median) | 0.703 | ❌ | 0.911 | ❌ | opencv_apriltag | 0.528 |
| euroc | LOO px, common tags (p90) | 1.285 | ❌ | 1.476 | ❌ | opencv_apriltag | 1.005 |
| euroc | ms/img | 6.9 | ✅ | 4.2 | ✅ | opencv | 8.2 |
| icra-forward | Recall % | 73.43 | ✅ | 73.70 | ✅ | aruco_nano | 53.71 |
| icra-forward | Precision % | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| icra-forward | F1 | 84.68 | ✅ | 84.86 | ✅ | aruco_nano | 69.89 |
| icra-forward | Corner RMSE px, common tags (mean) | 0.158 | ✅ | 0.139 | ✅ | aruco_nano | 0.241 |
| icra-forward | Corner RMSE px, common tags (p90) | 0.360 | ✅ | 0.359 | ✅ | aruco_nano | 0.523 |
| icra-forward | ms/img | 31.8 | ❌ | 51.4 | ❌ | aruco_nano | 10.5 |
| icra-circle | Recall % | 85.35 | ✅ | 83.29 | ✅ | aruco_nano | 80.60 |
| icra-circle | Precision % | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| icra-circle | F1 | 92.10 | ✅ | 90.88 | ✅ | aruco_nano | 89.26 |
| icra-circle | Corner RMSE px, common tags (mean) | 0.299 | ❌ | 0.252 | ❌ | aruco_nano | 0.185 |
| icra-circle | Corner RMSE px, common tags (p90) | 0.495 | ❌ | 0.432 | ❌ | aruco_nano | 0.234 |
| icra-circle | ms/img | 39.1 | ❌ | 53.2 | ❌ | aruco_nano | 12.5 |
| icra-random | Recall % | 99.91 | ❌ | 99.95 | ❌ | aruco_nano | 100.00 |
| icra-random | Precision % | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| icra-random | F1 | 99.96 | ❌ | 99.97 | ❌ | aruco_nano | 100.00 |
| icra-random | Corner RMSE px, common tags (mean) | 0.351 | ❌ | 0.323 | ❌ | aruco_nano | 0.178 |
| icra-random | Corner RMSE px, common tags (p90) | 0.504 | ❌ | 0.473 | ❌ | aruco_nano | 0.219 |
| icra-random | ms/img | 36.9 | ❌ | 51.8 | ❌ | aruco_nano | 13.3 |
| hub-640 | Recall % | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-640 | Precision % | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-640 | F1 | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-640 | Corner RMSE px, common tags (mean) | 0.877 | ❌ | 0.759 | ❌ | opencv_subpix | 0.712 |
| hub-640 | Corner RMSE px, common tags (p90) | 1.069 | ❌ | 0.968 | ❌ | opencv_subpix | 0.804 |
| hub-640 | ms/img | 3.1 | ❌ | 3.4 | ❌ | aruco_nano | 1.3 |
| hub-720p | Recall % | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-720p | Precision % | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-720p | F1 | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-720p | Corner RMSE px, common tags (mean) | 0.969 | ❌ | 0.864 | ❌ | opencv_subpix | 0.713 |
| hub-720p | Corner RMSE px, common tags (p90) | 1.066 | ❌ | 0.997 | ❌ | opencv_subpix | 0.828 |
| hub-720p | ms/img | 7.0 | ❌ | 9.6 | ❌ | aruco_nano | 3.4 |
| hub-1080p | Recall % | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-1080p | Precision % | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-1080p | F1 | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-1080p | Corner RMSE px, common tags (mean) | 0.794 | ❌ | 0.740 | ❌ | opencv_subpix | 0.706 |
| hub-1080p | Corner RMSE px, common tags (p90) | 0.959 | ❌ | 0.931 | ❌ | opencv_subpix | 0.798 |
| hub-1080p | ms/img | 15.5 | ❌ | 21.6 | ❌ | aruco_nano | 7.6 |
| hub-4k | Recall % | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-4k | Precision % | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-4k | F1 | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-4k | Corner RMSE px, common tags (mean) | 0.868 | ❌ | 0.777 | ❌ | opencv_subpix | 0.713 |
| hub-4k | Corner RMSE px, common tags (p90) | 0.957 | ❌ | 0.958 | ❌ | opencv_subpix | 0.831 |
| hub-4k | ms/img | 52.8 | ❌ | 87.8 | ❌ | aruco_nano | 31.3 |
| hub-high-iso | Recall % | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-high-iso | Precision % | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-high-iso | F1 | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-high-iso | Corner RMSE px, common tags (mean) | 0.758 | ❌ | 0.739 | ❌ | opencv_subpix | 0.708 |
| hub-high-iso | Corner RMSE px, common tags (p90) | 0.944 | ❌ | 0.933 | ❌ | opencv_subpix | 0.799 |
| hub-high-iso | ms/img | 13.9 | ❌ | 21.8 | ❌ | aruco_nano | 8.2 |
| hub-low-key | Recall % | 98.00 | ❌ | 14.00 | ❌ | aruco_nano | 100.00 |
| hub-low-key | Precision % | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-low-key | F1 | 98.99 | ❌ | 24.56 | ❌ | aruco_nano | 100.00 |
| hub-low-key | Corner RMSE px, common tags (mean) | 0.449 | ❌ | 0.921 | ❌ | opencv_subpix | 0.226 |
| hub-low-key | Corner RMSE px, common tags (p90) | 0.754 | ❌ | 1.798 | ❌ | opencv_subpix | 0.285 |
| hub-low-key | ms/img | 8.6 | ❌ | 14.0 | ❌ | aruco_nano | 6.3 |
| hub-raw-pipeline | Recall % | 100.00 | ✅ | 60.00 | ❌ | aruco_nano | 100.00 |
| hub-raw-pipeline | Precision % | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-raw-pipeline | F1 | 100.00 | ✅ | 75.00 | ❌ | aruco_nano | 100.00 |
| hub-raw-pipeline | Corner RMSE px, common tags (mean) | 0.601 | ❌ | 1.456 | ❌ | opencv_subpix | 0.389 |
| hub-raw-pipeline | Corner RMSE px, common tags (p90) | 0.899 | ❌ | 2.504 | ❌ | opencv_subpix | 0.464 |
| hub-raw-pipeline | ms/img | 11.1 | ❌ | 16.2 | ❌ | aruco_nano | 9.0 |
| hub-tag16h5 | Recall % | 100.00 | ✅ | 100.00 | ✅ | aruco_nano | 100.00 |
| hub-tag16h5 | Precision % | 92.59 | ❌ | 90.09 | ❌ | aruco_nano | 100.00 |
| hub-tag16h5 | F1 | 96.15 | ❌ | 94.79 | ❌ | aruco_nano | 100.00 |
| hub-tag16h5 | Corner RMSE px, common tags (mean) | 0.912 | ❌ | 0.753 | ❌ | aruco_nano | 0.701 |
| hub-tag16h5 | Corner RMSE px, common tags (p90) | 1.105 | ❌ | 0.878 | ❌ | opencv_subpix | 0.825 |
| hub-tag16h5 | ms/img | 13.8 | ❌ | 19.5 | ❌ | aruco_nano | 7.7 |
| hub-aprilgrid | Recall % | 99.07 | ✅ | 97.01 | ✅ | aruco_nano | 92.17 |
| hub-aprilgrid | Precision % | 99.63 | ❌ | 99.40 | ❌ | opencv | 99.91 |
| hub-aprilgrid | F1 | 99.35 | ✅ | 98.19 | ✅ | aruco_nano | 95.80 |
| hub-aprilgrid | Corner RMSE px, common tags (mean) | 0.944 | ❌ | 0.915 | ❌ | aruco_nano | 0.056 |
| hub-aprilgrid | Corner RMSE px, common tags (p90) | 1.125 | ❌ | 1.075 | ❌ | aruco_nano | 0.074 |
| hub-aprilgrid | ms/img | 24.4 | ❌ | 22.7 | ❌ | aruco_nano | 9.9 |
| hub-charuco | Recall % | 97.27 | ✅ | 99.85 | ✅ | aruco_nano | 90.36 |
| hub-charuco | Precision % | 99.57 | ❌ | 99.39 | ❌ | opencv | 99.96 |
| hub-charuco | F1 | 98.41 | ✅ | 99.62 | ✅ | aruco_nano | 94.78 |
| hub-charuco | Corner RMSE px, common tags (mean) | 0.916 | ❌ | 0.880 | ❌ | opencv_subpix | 0.819 |
| hub-charuco | Corner RMSE px, common tags (p90) | 1.106 | ❌ | 1.043 | ❌ | opencv_subpix | 0.875 |
| hub-charuco | ms/img | 20.8 | ❌ | 19.3 | ❌ | aruco_nano | 9.5 |

## Reading the table

- **Recall.** The candidate wins or ties recall on 13 of 15 benchmarks.

  | Benchmark | Candidate | `standard` | aruco_nano |
  | :-- | --: | --: | --: |
  | Liu4K | 66.56 % | 29.12 % | 66.27 % |
  | EuRoC | 87.06 % | 20.43 % | — (OpenCV 2-bit 44.55 %) |
  | low_key | 98 % | 14 % | |
  | raw_pipeline | 100 % | 60 % | |

  - **ICRA random** is the only recall loss on photographs: 99.91 % against 100 %.
  - **ChArUco is a regression** against `standard`: 97.27 % vs 99.85 %, 71 misses vs 4. It is the
    one benchmark where the candidate front end loses markers, and needs a root cause before M7.
- **Precision** is lost by a few false positives:

  | Benchmark | Candidate | `standard` | Best reference |
  | :-- | --: | --: | --: |
  | Liu4K | 2 FP (99.97 %) | 8 FP | |
  | AprilGrid | 19 FP (99.63 %) | 30 FP | OpenCV 99.91 % |
  | ChArUco | 11 FP (99.57 %) | 16 FP | OpenCV 99.96 % |
  | tag16h5 | 8 FP (92.59 %) | 11 FP (90.09 %) | |

  tag16h5 FPs are coincidental 0-error decodes on quads of about 2 px per cell. The planned M3
  bit-bimodality evidence targets them.
- **Corner accuracy is the largest block of losses** for both configurations. It is also where
  the candidate regresses: render-tag 640 mean 0.759 → 0.877 px, and the tag16h5 p99 tail
  1.07 → 3.90 px. The cause is that corner seeds inherit the LocalMean threshold's boundary
  shift. This is why M6 (photometric linearisation, EdLines bias, ERF seed decoupling) is a
  prerequisite for switching the `standard` default (M7).
  - **AprilGrid** loses by an order of magnitude (0.94 vs 0.056 px). The board's tag corners are
  X-junctions, which `cornerSubPix` models exactly.
  - **ICRA forward** is won on corner error by both configurations.
- **EuRoC.** The candidate wins recall (87 % vs 45 %) and latency. It loses:
  - precision: 542 FP, 98.13 % vs 99.65 %;
  - LOO corner error: 0.70 vs 0.53 px median.

  The Kalibr 2-bit border is still decoded only through the 0.9 scale retry; that is M5.

## What is left

| Lever | Milestone | Cells it targets |
| :-- | :-- | :-- |
| Photometric linearisation, EdLines bias, ERF seed decoupling | M6 | render-tag, ICRA circle/random, boards |
| Border bits as a layout parameter | M5 | EuRoC precision and corners |
| Bit-bimodality evidence | M3 | tag16h5 precision |
| Candidate generation that changes contour geometry; quad simplification | M8 | 1T latency |

- **1T latency.** The exact optimisations are exhausted; the measured negatives are in
  [Benchmarking Lessons §4.5](lessons.md#45-pruned-algorithms-dont-reintroduce). What remains changes contour geometry:
  - crack-traced contours restricted to the components CCL keeps;
  - a cheaper quad simplification than `select_dominant_vertices`.

  Each needs this full scoreboard to validate.
- **Default switch (M7)** stays blocked on the render-tag corner regression and the ChArUco
  recall loss.
