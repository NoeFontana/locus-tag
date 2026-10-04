# SOTA scoreboard checkpoint (2026-10-04)

Follows the [2026-10-03 checkpoint](sota_scoreboard_20261003.md). The champion is the `standard`
profile of the branch for #434: decode-first and the dark-ring check by default (#433),
`corner_subpix` with the junction window chosen per corner (#431, #432), plus #434's fused
junction/edge-line corners with the photometric corner calibration. `main` at `365c9ba` is the
before column. Reproduce any cell with `cargo xtask sota` (`xtask/README.md`).

**Result:** 52 of 73 judged cells won (`main`: 47). Corner metrics are debiased: each detector's
mean radial offset is removed and reported separately ([why](../lessons/rotation-tail-and-edge-refinement.md#2026-10-04--the-marker-calibrates-its-own-photometric-inset)).

| Benchmark | Corner metric (px) | `main` | This build | Best reference |
| :-- | :-- | --: | --: | --: |
| render-tag 640 | mean / p90 | 0.215 / 0.318 | 0.062 / 0.099 | 0.223 / 0.329 |
| render-tag 720p | mean / p90 | 0.347 / 0.320 | 0.210 / 0.138 | 0.226 / 0.301 |
| render-tag 1080p | mean / p90 | 0.228 / 0.326 | 0.062 / 0.109 | 0.230 / 0.323 |
| render-tag 4K | mean / p90 | 0.217 / 0.295 | 0.050 / 0.083 | 0.221 / 0.314 |
| high_iso | mean / p90 | 0.229 / 0.327 | 0.063 / 0.106 | 0.232 / 0.323 |
| low_key | mean / p90 | 0.134 / 0.147 | 0.047 / 0.058 | 0.180 / 0.241 |
| tag16h5 | mean / p90 | 0.209 / 0.299 | 0.055 / 0.089 | 0.208 / 0.297 |
| ChArUco | mean / p90 | 0.150 / 0.203 | 0.032 / 0.052 | 0.170 / 0.233 |
| AprilGrid | mean / p90 | 0.058 / 0.074 | 0.045 / 0.060 | 0.056 / 0.074 |
| ICRA forward | mean / p90 | 0.113 / 0.148 | 0.142 / 0.229 | 0.176 / 0.380 |
| ICRA circle | mean / p90 | 0.112 / 0.151 | 0.156 / 0.249 | 0.118 / 0.156 |
| ICRA random | mean / p90 | 0.113 / 0.148 | 0.188 / 0.316 | 0.112 / 0.148 |
| EuRoC | LOO median / p90 | 0.304 / 0.597 | 0.314 / 0.625 | 0.527 / 1.005 |
| Liu4K | median | 0.402 | 0.391 | 0.282 |

ICRA's rendered markers have 0.960-cell borders with an exact outline (measured against
ground-truth corners), which the calibration reads as a photometric inset: its corners move
outward by about 0.2 px. Accepted as a dataset artefact; no printed marker has it.

**Latency** (1 thread, `--jobs 1 --threads 1`, ms/img):

| Benchmark | `main` | This build |
| :-- | --: | --: |
| Liu4K (stride 8) | 126.5 | 127.8 |
| ICRA forward (stride 2) | 42.7 | 44.2 |
| ICRA random (stride 8, ≈130 markers/frame) | 48.5 (re-run 48.4) | 55.2 (re-run 55.1) |
| render-tag 1080p (stride 2) | 18.0 | 18.1 |
| render-tag 4K (stride 2) | 72.0 | 71.1 |

Pose metrics are not in this table (it is corner-level); single-tag pose results are in the
lessons page linked above, measured with the regression suites' pose path.

## Setup

| Item | Value |
| :--- | :--- |
| CPU | AMD EPYC-Milan, 8 vCPU, AVX2, KVM guest (`lscpu`, measurement session) |
| OS / toolchain | Linux 6.8.0-139-generic, rustc 1.92.0 |
| Build | `--release` wheel (`maturin develop --release`) |
| Accuracy runs | full datasets, `--jobs 4` |
| Timing runs | `--jobs 1 --threads 1`, strides as listed, serial (no concurrent load) |
| References | OpenCV 4.10 (pinned; NONE / SUBPIX / APRILTAG refinement), aruco_nano `961b18b` |

## Full win table

`locus_fin1004` against the best published reference operating point (aruco_nano, opencv, opencv_subpix, opencv_apriltag) per metric; a tie is a win. **52 / 73** judged cells won.

| Benchmark | Metric | Locus | Best reference | Value | Verdict |
| :-- | :-- | --: | :-- | --: | :-: |
| liu4k | Recall % | 28.80 | aruco_nano | 66.27 | ❌ |
| liu4k | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| liu4k | F1 | 44.72 | aruco_nano | 79.71 | ❌ |
| liu4k | Corner px, debiased, common tags (median) | 0.391 | opencv_apriltag | 0.282 | ❌ |
| liu4k | ms/img | 127.8 | — | — | n/a |
| euroc | Recall % | 20.51 | opencv_apriltag | 44.57 | ❌ |
| euroc | Precision % | 98.61 | opencv_apriltag | 99.70 | ❌ |
| euroc | LOO px, common tags (median) | 0.314 | opencv_apriltag | 0.527 | ✅ |
| euroc | LOO px, common tags (p90) | 0.625 | opencv_apriltag | 1.005 | ✅ |
| euroc | ms/img | 4.2 | opencv_subpix | 17.6 | n/a |
| icra-forward | Recall % | 73.77 | aruco_nano | 53.71 | ✅ |
| icra-forward | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| icra-forward | F1 | 84.90 | aruco_nano | 69.89 | ✅ |
| icra-forward | Corner RMSE px, debiased, common tags (mean) | 0.142 | aruco_nano | 0.176 | ✅ |
| icra-forward | Corner RMSE px, debiased, common tags (p90) | 0.229 | aruco_nano | 0.380 | ✅ |
| icra-forward | ms/img | 44.2 | — | — | n/a |
| icra-circle | Recall % | 83.40 | aruco_nano | 80.60 | ✅ |
| icra-circle | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| icra-circle | F1 | 90.95 | aruco_nano | 89.26 | ✅ |
| icra-circle | Corner RMSE px, debiased, common tags (mean) | 0.156 | aruco_nano | 0.118 | ❌ |
| icra-circle | Corner RMSE px, debiased, common tags (p90) | 0.249 | aruco_nano | 0.156 | ❌ |
| icra-circle | ms/img | 56.4 | aruco_nano | 15.8 | n/a |
| icra-random | Recall % | 99.92 | aruco_nano | 100.00 | ❌ |
| icra-random | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| icra-random | F1 | 99.96 | aruco_nano | 100.00 | ❌ |
| icra-random | Corner RMSE px, debiased, common tags (mean) | 0.188 | aruco_nano | 0.112 | ❌ |
| icra-random | Corner RMSE px, debiased, common tags (p90) | 0.316 | aruco_nano | 0.148 | ❌ |
| icra-random | ms/img | 55.2 | — | — | n/a |
| hub-640 | Recall % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-640 | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-640 | F1 | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-640 | Corner RMSE px, debiased, common tags (mean) | 0.062 | opencv_subpix | 0.223 | ✅ |
| hub-640 | Corner RMSE px, debiased, common tags (p90) | 0.099 | opencv_subpix | 0.329 | ✅ |
| hub-640 | ms/img | 3.0 | aruco_nano | 1.6 | n/a |
| hub-720p | Recall % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-720p | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-720p | F1 | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-720p | Corner RMSE px, debiased, common tags (mean) | 0.210 | opencv_subpix | 0.226 | ✅ |
| hub-720p | Corner RMSE px, debiased, common tags (p90) | 0.138 | opencv_subpix | 0.301 | ✅ |
| hub-720p | ms/img | 8.1 | aruco_nano | 4.6 | n/a |
| hub-1080p | Recall % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-1080p | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-1080p | F1 | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-1080p | Corner RMSE px, debiased, common tags (mean) | 0.062 | opencv_subpix | 0.230 | ✅ |
| hub-1080p | Corner RMSE px, debiased, common tags (p90) | 0.109 | aruco_nano | 0.323 | ✅ |
| hub-1080p | ms/img | 18.1 | — | — | n/a |
| hub-4k | Recall % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-4k | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-4k | F1 | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-4k | Corner RMSE px, debiased, common tags (mean) | 0.050 | opencv_subpix | 0.221 | ✅ |
| hub-4k | Corner RMSE px, debiased, common tags (p90) | 0.083 | opencv_subpix | 0.314 | ✅ |
| hub-4k | ms/img | 71.1 | — | — | n/a |
| hub-high-iso | Recall % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-high-iso | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-high-iso | F1 | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-high-iso | Corner RMSE px, debiased, common tags (mean) | 0.063 | opencv_subpix | 0.232 | ✅ |
| hub-high-iso | Corner RMSE px, debiased, common tags (p90) | 0.106 | opencv_subpix | 0.323 | ✅ |
| hub-high-iso | ms/img | 19.0 | aruco_nano | 11.2 | n/a |
| hub-low-key | Recall % | 14.00 | aruco_nano | 100.00 | ❌ |
| hub-low-key | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-low-key | F1 | 24.56 | aruco_nano | 100.00 | ❌ |
| hub-low-key | Corner RMSE px, debiased, common tags (mean) | 0.047 | opencv_subpix | 0.180 | ✅ |
| hub-low-key | Corner RMSE px, debiased, common tags (p90) | 0.058 | aruco_nano | 0.241 | ✅ |
| hub-low-key | ms/img | 13.7 | aruco_nano | 10.1 | n/a |
| hub-raw-pipeline | Recall % | 62.00 | aruco_nano | 100.00 | ❌ |
| hub-raw-pipeline | Precision % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-raw-pipeline | F1 | 76.54 | aruco_nano | 100.00 | ❌ |
| hub-raw-pipeline | Corner RMSE px, debiased, common tags (mean) | 1.329 | opencv_subpix | 0.194 | ❌ |
| hub-raw-pipeline | Corner RMSE px, debiased, common tags (p90) | 2.557 | aruco_nano | 0.252 | ❌ |
| hub-raw-pipeline | ms/img | 15.3 | aruco_nano | 12.4 | n/a |
| hub-tag16h5 | Recall % | 100.00 | aruco_nano | 100.00 | ✅ |
| hub-tag16h5 | Precision % | 99.01 | aruco_nano | 100.00 | ❌ |
| hub-tag16h5 | F1 | 99.50 | aruco_nano | 100.00 | ❌ |
| hub-tag16h5 | Corner RMSE px, debiased, common tags (mean) | 0.055 | aruco_nano | 0.208 | ✅ |
| hub-tag16h5 | Corner RMSE px, debiased, common tags (p90) | 0.089 | aruco_nano | 0.297 | ✅ |
| hub-tag16h5 | ms/img | 17.2 | aruco_nano | 10.2 | n/a |
| hub-aprilgrid | Recall % | 96.53 | aruco_nano | 92.17 | ✅ |
| hub-aprilgrid | Precision % | 99.48 | opencv | 99.91 | ❌ |
| hub-aprilgrid | F1 | 97.98 | aruco_nano | 95.80 | ✅ |
| hub-aprilgrid | Corner RMSE px, debiased, common tags (mean) | 0.045 | aruco_nano | 0.056 | ✅ |
| hub-aprilgrid | Corner RMSE px, debiased, common tags (p90) | 0.060 | aruco_nano | 0.074 | ✅ |
| hub-aprilgrid | ms/img | 24.4 | aruco_nano | 12.0 | n/a |
| hub-charuco | Recall % | 99.54 | aruco_nano | 90.36 | ✅ |
| hub-charuco | Precision % | 99.54 | opencv | 99.96 | ❌ |
| hub-charuco | F1 | 99.54 | aruco_nano | 94.78 | ✅ |
| hub-charuco | Corner RMSE px, debiased, common tags (mean) | 0.032 | opencv_subpix | 0.170 | ✅ |
| hub-charuco | Corner RMSE px, debiased, common tags (p90) | 0.052 | opencv_subpix | 0.233 | ✅ |
| hub-charuco | ms/img | 20.0 | aruco_nano | 13.3 | n/a |

Corner cells are judged on the RMSE left after removing each detector's mean radial
offset on the benchmark; the offsets themselves (px, + = outward) are not judged:

| Benchmark | locus_fin1004 | aruco_nano | opencv | opencv_subpix | opencv_apriltag |
| :-- | --: | --: | --: | --: | --: |
| liu4k | +0.333 | +0.189 | -0.209 | +0.228 | +0.390 |
| icra-forward | +0.184 | -0.198 | -0.459 | -0.256 | +0.101 |
| icra-circle | +0.211 | -0.145 | -0.469 | -0.222 | +0.071 |
| icra-random | +0.310 | -0.140 | -0.500 | -0.174 | +0.074 |
| hub-640 | +0.024 | -0.681 | -1.022 | -0.679 | -0.614 |
| hub-720p | -0.029 | -0.684 | -1.077 | -0.680 | -0.610 |
| hub-1080p | +0.027 | -0.673 | -1.026 | -0.670 | -0.605 |
| hub-4k | +0.023 | -0.685 | -1.073 | -0.680 | -0.605 |
| hub-high-iso | +0.019 | -0.675 | -1.039 | -0.672 | -0.607 |
| hub-low-key | +0.013 | -0.146 | -0.666 | -0.139 | +0.016 |
| hub-raw-pipeline | +0.177 | -0.358 | -0.817 | -0.343 | -0.207 |
| hub-tag16h5 | +0.020 | -0.673 | -1.141 | -0.754 | -0.607 |
| hub-aprilgrid | -0.002 | -0.003 | -1.273 | -0.158 | -0.728 |
| hub-charuco | +0.000 | -0.844 | -1.179 | -0.800 | -0.728 |
