# Real-image competitiveness: Liu4K + EuRoC (2026-10-01)

Live snapshot (supersedes the 2026-09-19 Liu4K report). Locus against OpenCV `aruco` and
the current SOTA, aruco_nano, on the two real-image datasets. Reproduce any row with
`cargo xtask sota` (`xtask/README.md`). Mechanisms and fix directions live in the lessons
pages linked under [Root causes](#root-causes); this page carries the numbers. Pose
metrics are not reported: Liu4K has no poses, and EuRoC is scored on corners only, so the
Fast/Accurate pose-mode rule does not apply.

!!! note "EuRoC corner cells carry a known measurement bias (fixed 2026-10-07)"

    The EuRoC `LOO px` figures on this page were scored with an **inexact**
    undistortion inverse (`cv2.undistortPoints`, which stops short with no convergence
    test); see PR #455.

    Across the fourteen EuRoC runs whose detections are preserved, the correction is a
    near-fixed **absolute** shift of -0.005 to +0.011 px. It is therefore largest in
    *relative* terms for the most accurate detector -- +4.0 % at 0.27 px, +2.7 % at 0.32 px,
    +0.75 % at 0.48 px, +0.3 % at 0.98 px, and indistinguishable from zero above 2 px --
    which is exactly why it flattered the best corners.

    At the 0.40-1.69 px own-set levels on this page the shift is about +0.003 to +0.004 px,
    and no conclusion here turns on it; the `cv2.cornerSubPix` comparison (0.695 -> 0.248
    px) would widen slightly, since the correction is larger at the lower value.

    The runs behind this page were not archived, so these cells are **not** re-derived here:
    a figure rescaled by a factor measured on *other* runs would not be a measurement, and the
    common-tag set each cell used depended on which runs shared the directory at the time,
    which is not recoverable. Re-running `cargo xtask sota` today produces corrected values.
    The one page whose detections survive is
    [EuRoC SOTA (2026-10-04)](euroc_sota_20261004.md), re-derived exactly.

    What the metric measures -- and why it must not be compared against a render-tag RMSE --
    is in the [EuRoC error budget](euroc_error_budget.md).

## Setup

| Item | Value |
| :--- | :--- |
| CPU | AMD EPYC-Milan, 8 vCPU (4 cores × 2 threads), AVX2, KVM guest (`lscpu`, measurement session) |
| OS / toolchain | Linux 6.8.0-139-generic, rustc 1.92.0, CPython 3.14 |
| Locus | `maturin develop --release`; `main` `531b5aa` + the Liu4K harness; `LocalMean` rows use PR #383 |
| References | OpenCV 4.10.0 (minimal build), aruco_nano `961b18b`, both unpatched; AprilTag 3 (`pupil-apriltags`) |
| Accuracy | `RAYON_NUM_THREADS=1`, detector processes in parallel (deterministic; latency from these runs is not reported) |
| Latency | serial `--jobs 1`, best of 2 per image, decode excluded, thread count only via `RAYON_NUM_THREADS` / `cv::setNumThreads` |

"Candidate" below = `standard` with `threshold.mode = LocalMean` (PR #383),
`local_mean_radius = 7`, `constant = 3`, `enable_sharpening = false`,
`quad.min_fill_ratio = min_density = max_elongation = 0`, `segmentation.connectivity = Four`.
"k·σ̂ₙ" = the same with `constant = clamp(round(k · σ̂ₙ), 2, 20)` per image (σ̂ₙ:
`compute_image_noise_floor` formula), emulated from Python.

## Liu4K

924 images (~3300×4900), 9022 GT markers, `ARUCO_MIP_36h12`. Scorer = aruco_nano
`testperf.cpp`: same id, centre distance ≤ 10 px, first-match TP/FP/FN.

| Detector / config | Recall % | Prec % | F1 | R % <20 px | 20–45 | 45–100 | 100–250 | ≥250 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aruco_nano | 66.27 | 100.00 | 79.71 | 46.7 | 69.7 | 66.1 | 67.0 | 66.8 |
| OpenCV 4.10 | 57.69 | 100.00 | 73.17 | 0.0 | 13.5 | 69.6 | 66.2 | 60.9 |
| Locus `standard` | 29.11 | 99.58 | 45.05 | 39.1 | 38.7 | 33.5 | 29.0 | 18.3 |
| Locus `grid` | 50.22 | 99.56 | 66.76 | 62.9 | 67.9 | 59.3 | 51.4 | 27.7 |
| Locus `high_accuracy` | 44.69 | 99.93 | 61.76 | 0.0 | 54.2 | 55.7 | 48.0 | 30.5 |
| `standard`, sharpening off | 50.58 | 99.80 | 67.13 | 67.7 | 67.6 | 58.1 | 50.5 | 31.6 |
| + `LocalMean` r24/C15 (PR #383 default) | 59.92 | 99.72 | 74.86 | 63.2 | 70.5 | 66.1 | 60.4 | 46.9 |
| + `LocalMean` r24/C15, sharpening on | 56.73 | 99.80 | 72.34 | 61.8 | 66.5 | 62.6 | 57.1 | 44.2 |
| `grid` + `LocalMean` r24/C15 | 60.65 | 99.73 | 75.43 | 60.9 | 70.7 | 67.3 | 61.4 | 47.4 |
| `LocalMean` r7/C3, gates **on** | 38.49 | 99.29 | 55.48 | 72.2 | 75.5 | 69.3 | 29.6 | 0.2 |
| `LocalMean` r7/C3, gates off | 65.16 | 99.42 | 78.73 | 71.4 | 75.6 | 70.1 | 63.8 | 56.6 |
| `LocalMean` r7/C3, gates off, edge-score gate off | 65.26 | 99.36 | 78.78 | 66.3 | 75.7 | 70.2 | 64.2 | 57.0 |
| **Candidate** (r7/C3, gates off, 4-conn) | **67.19** | 99.61 | **80.25** | 71.4 | 76.6 | 72.0 | 66.1 | 59.1 |
| Candidate, C = 8 | 65.22 | 99.61 | 78.83 | 68.6 | 77.6 | 69.6 | 64.2 | 56.1 |
| Candidate, k·σ̂ₙ, k = 3 / 4 / 5 | 67.22 / 67.29 / 67.17 | 99.33–99.35 | 80.15–80.24 | | | | | |
| Candidate, refinement off | 65.95 | 99.78 | 79.41 | | | | | |
| Candidate (r4/C3), 2× `INTER_AREA` pre-downscale | 62.04 | 99.68 | 76.48 | 20.7 | 70.8 | 69.8 | 63.1 | 55.3 |

### Latency (231-image stride-4 subset, serial)

| Detector | Recall % (subset) | 1 thread ms | 8 threads ms |
| :--- | ---: | ---: | ---: |
| aruco_nano | 65.44 | **52.2** | 52.9 |
| Locus candidate | **65.57** | 313.7 | 84.6 |
| OpenCV 4.10 | 56.63 | 316.3 | 132.2 |
| Locus `standard` | 29.08 | 193.6 | 77.1 |

Load average 1.0–2.4 during the run (editor helpers only). Stage busy time from the
pipeline's `tracing` spans, 1 thread, 29 images:

| Stage | `standard` ms | candidate ms | candidate, refinement off ms |
| :--- | ---: | ---: | ---: |
| `quad_extraction` | 81.8 | 239.7 | 51.8 |
| `segmentation` | 79.2 | 34.7 | 35.3 |
| `threshold_apply_map` | 8.2 | 29.9 | 29.9 |
| `decoding_pass` | 5.8 | 18.0 | 9.2 |
| total | 205.9 | 327.3 | 131.1 |

The candidate refines ~776 quads per image to decode 6, and the O(1) contrast funnel rejects
~1 of them.

## EuRoC `cam_april`

1450 frames, 6×6 Kalibr AprilGrid (tag36h11 with a **2-bit** black border), MT9V034,
strong radtan distortion. No per-corner GT, so the scorer is GT-free (`tools/bench/sota/score.py`):
presence and precision come from a RANSAC board homography fitted, after undistortion, to
corners pooled over all detectors; accuracy is leave-one-tag-out (LOO) corner error
predicted from the same detector's other tags.

| Detector / config | Recall % | LOO median px (own set) |
| :--- | ---: | ---: |
| OpenCV 4.10, `markerBorderBits = 2` | 36.9 | 1.69 |
| aruco_nano / AprilTag 3 | 4.6 / 3.1 — unsupported (2-bit border) | — |
| Locus `standard` | 20.4 | 0.99 |
| Locus `grid` | 70.2 | 0.64 |
| Locus `high_accuracy` | 29.5 | 0.40 |
| `grid` + `LocalMean` r24/C15 | 91.1 | 0.60 |
| Candidate, C = 3 | 80.4 | 0.61 |
| Candidate, C = 8 | 90.4 | 0.58 |
| Candidate, k·σ̂ₙ, k = 3 / 4 / 5 | 82.8 / 86.9 / 88.7 | 0.59–0.60 |

Corner accuracy: replacing only the refiner on `grid`'s quads with `cv2.cornerSubPix`
(half-window 4) cuts LOO error from 0.695 to 0.248 px (median, 10 591 common tags).
`grid`'s ERF corners sit 0.39 px inward of the Kalibr X-junctions, and EdLines corners
0.17 px outward.

## Generality checks

ICRA 2020 `forward/pure_tags` (7700 GT, detection level, `Metrics.match_detections`):

| Config | Recall % | Prec % | Corner RMSE px |
| :--- | ---: | ---: | ---: |
| `standard` | 73.75 | 100.00 | 0.2796 |
| `high_accuracy` | 17.04 | 100.00 | 0.7657 |
| `standard` + `LocalMean` r24/C15 (PR #383) | 73.61 | 100.00 | 0.2824 |
| Candidate, C = 3 | 73.39 | 100.00 | 0.2981 |
| Candidate, k·σ̂ₙ, k = 5 | 73.39 | 100.00 | 0.2984 |

Render-tag hub sets, detection level (no intrinsics), recall % / precision % / mean corner RMSE px:

| Set | `standard` | `standard` + candidate (k = 5) | `high_accuracy` | `high_accuracy` + candidate (k = 5) |
| :--- | :--- | :--- | :--- | :--- |
| tag36h11 640×480 | 100 / 100 / 0.759 | 100 / 100 / 0.860 | 100 / 100 / 0.210 | 100 / 100 / 0.255 |
| tag36h11 1920×1080 | 100 / 100 / 0.741 | 100 / 100 / 0.780 | 100 / 100 / 0.215 | 100 / 100 / 0.218 |
| tag36h11 3840×2160 | 100 / 100 / 0.777 | 100 / 100 / 0.781 | 100 / 100 / 0.178 | 100 / 100 / 0.223 |
| high_iso | 100 / 100 / 0.739 | 100 / 96.2 / 0.765 | 100 / 100 / 0.210 | 100 / 100 / 0.263 |
| low_key | **14** / 100 / 0.921 | **98** / 100 / 0.441 | 100 / 100 / 0.527 | 98 / 100 / 0.824 |
| raw_pipeline | 60 / 100 / 1.456 | 100 / 96.2 / 1.812 | 100 / 100 / 0.292 | 100 / 100 / 0.606 |
| tag16h5 | 100 / **83.3** / 0.753 | 100 / **50.3** / 0.956 | 98 / 48.5 / 0.176 | 100 / 45.5 / 0.213 |

## Root causes

| | Root cause | Lesson |
| :--- | :--- | :--- |
| RC1 | CCL threshold = min/max midpoint, no validity gate, no bright-side guard band: markers merge with texture or shatter | [recall 2026-10-01](../lessons/recall-quad-icra.md#2026-10-01-real-image-recall-liu4k-euroc-the-segmentation-model) |
| RC2 | Quad pre-gates assume a filled blob (and 8-connectivity bridges to clutter); they also carry FP control for weak dictionaries | same |
| RC3 | Threshold offset in grey levels instead of noise units | same |
| RC4 | Border width not in the layout model: Kalibr 2-bit tags decode via a 0.9 scale retry, with false ids | same |
| L1–L3 | Sub-pixel refinement before decode on every candidate; funnel ineffective on texture | same |
| RC5 | Corner refiners assume linear photometry (sRGB gamma bias ∝ blur); ERF 1-DOF seed-direction scatter | [rotation-tail 2026-10-01](../lessons/rotation-tail-and-edge-refinement.md#2026-10-01-corner-bias-is-photometric-not-a-psf-floor) |
| RC5b | EdLines ~+0.5 px outward bias in linear light, cancelled by sRGB on render-tag | [EdLines 2026-10-01](../lessons/edlines-segmentation.md#2026-10-01-intrinsic-outward-bias-masked-by-srgb) |
| RC6 | Pixel-centre interop: Locus +0.5 vs OpenCV/Kalibr/Liu4K integer centres (measured +0.50 px); `CameraIntrinsics` convention unspecified | [coordinates](../../explanation/coordinates.md) |

Findings of the superseded 2026-09-19 report were resolved by #382 (`upscale_factor`
corner mapping) and #385 (`Detector(threads=)` wired, dead config removed); its inert
`threshold.constant` / `gradient_threshold` / `min_radius` / `max_radius` knobs are
removed or wired by PR #383.

## Next steps

Ordered so each step is independently measurable; every step re-runs render-tag, ICRA,
board hub, Liu4K and EuRoC.

1. **Threshold model**: land PR #383 and extend it with the noise-calibrated offset
   (σ̂ₙ from a subsampled |Laplacian| histogram, no `Vec` in `detect()`).
2. **Topology-invariant candidate validation**: contour-shape checks plus dictionary-aware
   false-positive control in place of fill/density/elongation, and 4-connectivity for the
   local-mean foreground.
3. **Decode-first ordering**: bound 131 ms/img at 1 thread before the near-miss retry.
4. **Border width as a layout parameter** (Kalibr AprilGrid), retiring the scale retry as
   a layout workaround.
5. **Photometric response model** (f32 LUT inside the samplers) together with the EdLines
   outward-bias fix; then re-evaluate GWLF-style corners (rotation-tail re-attempt condition).
6. **ERF seed-direction decoupling**, within the constraints of the 2-DOF Tukey negative.
7. **Area-averaged decimation** as the 4K latency Pareto option (`decimate_to` subsamples).

## External proposals evaluated

| Idea | Verdict |
| :--- | :--- |
| PR #383 `LocalMean` | Correct root cause (RC1): +31 pp Liu4K with sharpening off, +21 pp EuRoC. Needs RC2/RC3 for SOTA, and a corner-accuracy gate. |
| PR #384 shoot-limited sharpening | Ties sharpening-off on Liu4K (49.47 vs 49.68 %), slower than off at 1 and 8 threads; superseded on Liu4K by fixing RC1. |
| Contrast-scaled offset (Sauvola/Wolf) | Not measured; the noise-scaled offset (RC3) is measured robust on both datasets. |
| Octave discovery + full-res refinement | Supported: 62 % at about a quarter of the cost, losing < 20 px markers. Pareto option, not default. |
| Gate run generation by tile range | Equivalent to the validity gate (−0.18 pp, PR #383): latency lever at most. |
| Decouple corners from the binary mask | Partly true (ERF keeps the seed direction); the dominant corner effect is photometric (RC5). |

## Harness notes

- Liu4K GT corner winding is counter-clockwise (0, 3, 2, 1) vs Locus/OpenCV clockwise; the
  centre-based scorer is unaffected, any corner-error metric must remap. No EXIF rotation;
  GT is in each image's own pixel frame, integer-centre convention.
- Liu4K is CC BY 4.0 (Zenodo 10.5281/zenodo.18667018), fetched at runtime and never
  committed or packaged.
- EuRoC runs record which detectors were skipped as unsupported; references are compared
  as published and never patched.
