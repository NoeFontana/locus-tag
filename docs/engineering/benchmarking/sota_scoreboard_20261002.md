# SOTA scoreboard baseline (2026-10-02)

Live snapshot. It is the starting point of the SOTA programme (`#409` root causes, then
milestones M1–M8): shipped Locus `standard` against the **best published operating point of
each reference** on every benchmark both references can decode. Reproduce any cell with
`cargo xtask sota` (`xtask/README.md`). Mechanisms live in the
real-image root-cause snapshot ([Liu4K + EuRoC, 2026-10-01](liu4k_euroc_sota_20261001.md)) and
the lessons pages; this page carries the numbers.

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

**Result: `standard` wins 31 / 88 judged cells.** "Industrial SOTA" here means every judged
cell green. Pose metrics are not reported: the win table is corner-level, so the Fast/Accurate
pose-mode rule does not apply.

## Setup

| Item | Value |
| :--- | :--- |
| CPU | AMD EPYC-Milan, 8 vCPU (4 cores × 2 threads), AVX2, KVM guest (`lscpu`, measurement session) |
| OS / toolchain | Linux 6.8.0-139-generic, rustc 1.92.0 |
| Locus | `maturin develop --release` at `main` `0324fd0` (+ harness-only commits of this PR) |
| References | OpenCV 4.10.0 (minimal build; `CORNER_REFINE_NONE` / `SUBPIX` / `APRILTAG`, `errorCorrectionRate = 0`), aruco_nano `961b18b`; both unpatched |
| Accuracy | full datasets, `--jobs 8` (detectors in parallel; latency from these runs is not judged) |
| Latency | separate serial runs `<benchmark>@t1`: `--jobs 1`, 1 thread (`RAYON_NUM_THREADS=1` / `cv::setNumThreads(1)`), best of 2 per image, decode excluded, strided (Liu4K 8, EuRoC 15, ICRA forward 2, circle 4, random 8, render-tag 2, tag16h5 4, boards 6) |
| Load | 1.1–2.5 load average during the timing runs (another session's builds); latency ratios below are ≥ 4×, far above that noise |

## Win table

Locus value vs the best reference value per metric; a tie is a win. Corner error is
order-preserving, in the OpenCV pixel convention, on the tags every reference with ≥ 20 %
recall matched (`tools/bench/sota/score.py`).

| Benchmark | Metric | Locus | Best reference | Value | Verdict |
| :-- | :-- | --: | :-- | --: | :-: |
| liu4k | Recall % | 29.11 | aruco_nano | 66.27 | ❌ |
| liu4k | Precision % | 99.58 | aruco_nano | 100.00 | ❌ |
| liu4k | F1 | 45.05 | aruco_nano | 79.71 | ❌ |
| liu4k | Corner px, common tags (median) | 0.686 | aruco_nano | 0.473 | ❌ |
| liu4k | ms/img | 194.6 | aruco_nano | 53.7 | ❌ |
| euroc | Recall % | 20.42 | opencv_apriltag | 44.53 | ❌ |
| euroc | Precision % | 97.81 | opencv_apriltag | 99.58 | ❌ |
| euroc | LOO px, common tags (median) | 0.911 | opencv_apriltag | 0.529 | ❌ |
| euroc | LOO px, common tags (p90) | 1.474 | opencv_apriltag | 1.005 | ❌ |
| euroc | ms/img | 8.5 | opencv | 8.2 | ❌ |
| icra-forward | Recall % | 73.75 | aruco_nano | 53.71 | ✅ |
| icra-forward | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| icra-forward | F1 | 84.89 | aruco_nano | 69.89 | ✅ |
| icra-forward | Corner RMSE px, common tags (mean) | 0.139 | aruco_nano | 0.241 | ✅ |
| icra-forward | Corner RMSE px, common tags (p90) | 0.359 | aruco_nano | 0.523 | ✅ |
| icra-forward | ms/img | 98.0 | aruco_nano | 10.3 | ❌ |
| icra-circle | Recall % | 83.29 | aruco_nano | 80.60 | ✅ |
| icra-circle | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| icra-circle | F1 | 90.88 | aruco_nano | 89.26 | ✅ |
| icra-circle | Corner RMSE px, common tags (mean) | 0.252 | aruco_nano | 0.185 | ❌ |
| icra-circle | Corner RMSE px, common tags (p90) | 0.432 | aruco_nano | 0.234 | ❌ |
| icra-circle | ms/img | 98.8 | aruco_nano | 12.5 | ❌ |
| icra-random | Recall % | 99.95 | aruco_nano | 100.00 | ❌ |
| icra-random | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| icra-random | F1 | 99.97 | aruco_nano | 100.00 | ❌ |
| icra-random | Corner RMSE px, common tags (mean) | 0.323 | aruco_nano | 0.178 | ❌ |
| icra-random | Corner RMSE px, common tags (p90) | 0.473 | aruco_nano | 0.219 | ❌ |
| icra-random | ms/img | 96.8 | aruco_nano | 13.3 | ❌ |
| hub-640 | Recall % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-640 | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-640 | F1 | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-640 | Corner RMSE px, common tags (mean) | 0.759 | opencv_subpix | 0.712 | ❌ |
| hub-640 | Corner RMSE px, common tags (p90) | 0.968 | opencv_subpix | 0.804 | ❌ |
| hub-640 | ms/img | 6.7 | aruco_nano | 1.3 | ❌ |
| hub-720p | Recall % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-720p | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-720p | F1 | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-720p | Corner RMSE px, common tags (mean) | 0.864 | opencv_subpix | 0.713 | ❌ |
| hub-720p | Corner RMSE px, common tags (p90) | 0.997 | opencv_subpix | 0.828 | ❌ |
| hub-720p | ms/img | 17.8 | aruco_nano | 3.4 | ❌ |
| hub-1080p | Recall % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-1080p | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-1080p | F1 | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-1080p | Corner RMSE px, common tags (mean) | 0.740 | opencv_subpix | 0.706 | ❌ |
| hub-1080p | Corner RMSE px, common tags (p90) | 0.931 | opencv_subpix | 0.798 | ❌ |
| hub-1080p | ms/img | 39.9 | aruco_nano | 7.6 | ❌ |
| hub-4k | Recall % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-4k | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-4k | F1 | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-4k | Corner RMSE px, common tags (mean) | 0.777 | opencv_subpix | 0.713 | ❌ |
| hub-4k | Corner RMSE px, common tags (p90) | 0.958 | opencv_subpix | 0.831 | ❌ |
| hub-4k | ms/img | 153.7 | aruco_nano | 31.5 | ❌ |
| hub-high-iso | Recall % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-high-iso | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-high-iso | F1 | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-high-iso | Corner RMSE px, common tags (mean) | 0.739 | opencv_subpix | 0.708 | ❌ |
| hub-high-iso | Corner RMSE px, common tags (p90) | 0.933 | opencv_subpix | 0.799 | ❌ |
| hub-high-iso | ms/img | 38.0 | aruco_nano | 8.2 | ❌ |
| hub-low-key | Recall % | 14.00 | aruco_nano | 100.00 | ❌ |
| hub-low-key | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-low-key | F1 | 24.56 | aruco_nano | 100.00 | ❌ |
| hub-low-key | Corner RMSE px, common tags (mean) | 0.921 | opencv_subpix | 0.226 | ❌ |
| hub-low-key | Corner RMSE px, common tags (p90) | 1.798 | opencv_subpix | 0.285 | ❌ |
| hub-low-key | ms/img | 18.1 | aruco_nano | 6.3 | ❌ |
| hub-raw-pipeline | Recall % | 60.00 | aruco_nano | 100.00 | ❌ |
| hub-raw-pipeline | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-raw-pipeline | F1 | 75.00 | aruco_nano | 100.00 | ❌ |
| hub-raw-pipeline | Corner RMSE px, common tags (mean) | 1.456 | opencv_subpix | 0.389 | ❌ |
| hub-raw-pipeline | Corner RMSE px, common tags (p90) | 2.504 | opencv_subpix | 0.464 | ❌ |
| hub-raw-pipeline | ms/img | 23.2 | aruco_nano | 9.0 | ❌ |
| hub-tag16h5 | Recall % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-tag16h5 | Precision % | 83.33 | aruco_nano | 100.00 | ❌ |
| hub-tag16h5 | F1 | 90.91 | aruco_nano | 100.00 | ❌ |
| hub-tag16h5 | Corner RMSE px, common tags (mean) | 0.753 | aruco_nano | 0.701 | ❌ |
| hub-tag16h5 | Corner RMSE px, common tags (p90) | 0.878 | opencv_subpix | 0.825 | ❌ |
| hub-tag16h5 | ms/img | 26.8 | aruco_nano | 7.7 | ❌ |
| hub-aprilgrid | Recall % | 97.01 | aruco_nano | 92.17 | ✅ |
| hub-aprilgrid | Precision % | 99.40 | opencv | 99.91 | ❌ |
| hub-aprilgrid | F1 | 98.19 | aruco_nano | 95.80 | ✅ |
| hub-aprilgrid | Corner RMSE px, common tags (mean) | 0.915 | aruco_nano | 0.056 | ❌ |
| hub-aprilgrid | Corner RMSE px, common tags (p90) | 1.075 | aruco_nano | 0.074 | ❌ |
| hub-aprilgrid | ms/img | 49.9 | aruco_nano | 9.9 | ❌ |
| hub-charuco | Recall % | 99.81 | aruco_nano | 90.36 | ✅ |
| hub-charuco | Precision % | 99.39 | opencv | 99.96 | ❌ |
| hub-charuco | F1 | 99.60 | aruco_nano | 94.78 | ✅ |
| hub-charuco | Corner RMSE px, common tags (mean) | 0.880 | opencv_subpix | 0.819 | ❌ |
| hub-charuco | Corner RMSE px, common tags (p90) | 1.043 | opencv_subpix | 0.875 | ❌ |
| hub-charuco | ms/img | 31.9 | aruco_nano | 9.5 | ❌ |

## Reading the table

- **Latency is lost everywhere** (aruco_nano is 4–10× faster at 1 thread: ICRA 1080p 10 vs 98
  ms, Liu4K 54 vs 195 ms). aruco_nano thresholds with a box mean, traces contours without CCL,
  decodes from integer corners and refines only decoded markers. Locus refines every candidate
  before decoding (M4) and runs a full CCL plus a materialised threshold map (M8).
- **Recall on hard photometry** (Liu4K 29 %, render-tag low_key 14 %, raw_pipeline 60 %) is the
  tile min/max threshold (RC1). The opt-in LocalMean front end (#413) with border-ring evidence
  (#414) reaches Liu4K 66.74 % / 100 % (aruco_nano 66.27 % / 100 %) and 98–100 % on the
  render-tag sets.
- **tag16h5 precision** (83 % vs 100 %) is missing marker evidence beyond the codeword: #414
  takes it to 99 %.
- **Corner accuracy** is lost on most render-tag and ICRA sets against OpenCV SUBPIX and
  aruco_nano (`cornerSubPix`), and by an order of magnitude on the AprilGrid board, where tag
  corners are X-junctions `cornerSubPix` models exactly. `high_accuracy` already wins the
  render-tag corner cells; bringing `standard` there is M6 (photometric linearisation, EdLines
  bias, ERF seed decoupling) and M7 (profile switch).
- **EuRoC**: Locus `standard` loses recall to OpenCV with `markerBorderBits = 2`; the Kalibr
  2-bit border is decoded only by the 0.9 scale retry (RC4, M5). With the LocalMean front end
  Locus reaches 86.9 % against 44.5 %.

Detector crashes are listed per report: pupil-apriltags (AprilTag 3, not part of the win
criterion) segfaults on several 1080p sets.
