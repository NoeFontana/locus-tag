# EuRoC, the real-data benchmark: SOTA scoreboard (2026-10-04)

Follows the [#434 checkpoint](sota_scoreboard_20261004.md). EuRoC `cam_april` is the only
real-camera benchmark with a verifiable board model, so this pass starts from it. The
analysis is in the
[recall lessons](../lessons/recall-quad-icra.md#2026-10-04-euroc-the-only-real-data-benchmark-scorer-connectivity-clipped-and-gross-corners).

!!! note "Corner cells re-derived with the fixed undistortion inverse (2026-10-08)"

    The `LOO corner error` cell below has been **recomputed from the same stored detections**
    with the converged undistortion inverse of PR #455, so it is a measurement, not a rescaled
    figure. The original cell read `0.316 / 0.634 | 0.283 / 0.575 | 0.516 / 0.996`; the inexact
    inverse had made it ~4 % low for Locus against ~0.8 % for OpenCV APRILTAG, overstating
    Locus's relative margin by about 3 pp. Locus still wins the cell by a wide margin.

    Provenance: `main` is the archived run `locus_main`, **This build** is `locus_final`, and the
    reference is `opencv_apriltag`, scored together over 1047 reference frames and 9923 common
    tags. The pre-fix recompute reproduces the original cell to the thousandth, which is what
    identifies the runs. Recall (20.86 / 86.02 %) and false positives (6 / 1) move by at most
    0.01 pp, and latency is untouched.

    This metric is still a self-consistency residual with a ~1.37x gain over the underlying
    corner noise and a ~0.26 px floor no detector change can move. It is not a corner accuracy
    and must not be compared against a render-tag RMSE:
    see [the EuRoC error budget](euroc_error_budget.md).

**This build** is `main` (`629a557`) plus:
- **EuRoC scorer fixes:** a judgeable lens-model radius, corner error in image pixels, and a
  self-consistent reference pool.
- **Frame-clipping rejection** (every profile).
- **Gross-corner repair** (`decoder.corner_subpix`).
- **`standard`:** 4-connectivity, with the blob-shape quad gates off.

**Result:** 65 of 93 judged cells won; `main` wins 57 on the same scorer, references and data.

## EuRoC

| Metric | `main` | This build | Best reference |
| :-- | --: | --: | --: |
| Recall | 20.85 % | **86.02 %** | 57.11 % (OpenCV NONE / SUBPIX, `markerBorderBits = 2`) |
| Precision | 99.910 % (6 FP) | **99.996 %** (1 FP) | 99.993 % (OpenCV APRILTAG, 1 FP) |
| LOO corner error, common tags, median / p90 (px) | 0.325 / 0.650 | **0.295 / 0.594** | 0.520 / 1.008 (OpenCV APRILTAG) |
| ms/img, 1 thread | 4.3 | 9.5 | 8.3 (OpenCV NONE) |

- **Latency:** Locus now decodes four times as many tags per frame. The per-frame latency cell is
  lost on that count; per decoded tag, Locus is faster than every reference.
- **Scorer:** before the fixes below, the same Locus detections scored 98.5 % precision.
  - The published radtan model holds only within 380 px of the principal point. Beyond that,
    every detector shows the same inward radial residual: −1.5 px at 380–400 px, −4.9 px at
    400–420 px, −10.4 px at 420–440 px.
  - The error was measured after undistortion, which stretches the periphery 1.5–2×.
  - The reference was pooled with OpenCV NONE/SUBPIX corners, which sit about 1.5 px inside on
    the 2-bit border.

## Whole scoreboard: what moved

Cells that flipped relative to `main`:

| Benchmark | Metric | `main` | This build | Best reference |
| :-- | :-- | --: | --: | --: |
| EuRoC | Recall % | 20.85 | 86.02 | 57.11 |
| EuRoC | Precision % | 99.91 | 99.996 | 99.993 |
| EuRoC | ms/img | 4.3 | 9.5 | 8.3 |
| raw_pipeline | Corner mean / p90, debiased (px) | 1.180 / 1.787 | 0.095 / 0.188 | 0.194 / 0.252 |
| AprilGrid | Precision % | 99.42 | 99.98 | 99.91 |
| AprilGrid BC | Precision %, corner mean (px) | 99.51, 0.071 | 99.94, 0.066 | 99.80, 0.070 |
| AprilGrid KB | Recall %, precision %, F1 | 82.35, 98.64, 89.76 | 96.26, 99.93, 98.06 | 89.77, 99.92, 94.41 |
| ChArUco | Precision % | 99.39 | 100.00 | 99.96 |
| tag16h5 | Recall %, F1 | 100, 100 | 99, 99.5 | 100, 100 |

Other recall gains that do not flip a cell:

| Benchmark | `main` | This build | Best reference |
| :-- | --: | --: | --: |
| low_key | 14 % | 66 % | 100 % |
| raw_pipeline | 60 % | 96 % | 100 % |
| Liu4K | 28.9 % | 37.2 % | 66.3 % |
| ICRA circle | 83.1 % | 85.1 % | 80.6 % |

**Costs:**
- **tag16h5:** one 300 px tag is lost, because 4-connectivity fragments its hollow ring.
- **ChArUco:** marker recall 99.6 → 99.2 %. In one render with 16 px markers (about 2 px per
  cell, whose rings connect only diagonally) only 3 of 14 markers remain. The ChArUco refiner's
  board pose fails on that frame (regression suite mean rotation 0.11° → 0.61°).

## Latency

1 thread, serial, ms/img, image decode excluded:

| Benchmark | `main` | This build | aruco_nano |
| :-- | --: | --: | --: |
| Liu4K (stride 8) | 127.0 | 162.5 | 52.3 |
| ICRA forward (stride 2) | 44.7 | 56.3 | 10.3 |
| ICRA circle (stride 4) | 56.4 | 69.3 | 12.6 |
| ICRA random (stride 8) | 55.6 | 65.8 | 13.2 |
| render-tag 1080p (stride 2) | 18.1 | 21.5 | 7.6 |
| render-tag 4K (stride 2) | 72.1 | 84.6 | 30.6 |
| tag16h5 (stride 4) | 17.7 | 21.4 | 7.5 |
| AprilGrid (stride 6) | 22.9 | 27.0 | 9.8 |
| ChArUco (stride 6) | 18.8 | 22.9 | 9.4 |

Each change's share of the increase, measured as separate serial runs:

| Change | render-tag 1080p | ICRA forward | Liu4K |
| :-- | --: | --: | --: |
| Repair + clipping rule | +0.3 ms | +0.5 ms | +0.4 ms |
| 4-connectivity | +2.1 ms | +7.5 ms | +28 ms |
| Gates off | +1.0 ms | +3.6 ms | +7 ms |

Stage spans (ICRA forward) place the 4-connectivity cost in quad extraction (9.3 → 13.9 ms) and
segmentation (10.3 → 12.2 ms). Texture splits into more components above `min_area`, each traced
and reduced, while the decoder sees only 7 % more candidates. The fix is a cheaper per-component
trace (plan M8), not a coarser front end.

## Not shipped, measured

| Variant | Result |
| :-- | :-- |
| Threshold cut below the local midpoint (`f` = 0.35 / 0.40) | EuRoC 98.1 / 96.0 %, Liu4K 58.7 / 52.6 %, low_key 100 / 98 %, raw_pipeline 100 / 100 %; but render-tag 640 recall 94 / 98 %, tag16h5 93 / 96 % (scoreboard 64 / 63 cells vs 65) |
| 8-connectivity linking diagonals only through thin structures | No neighbourhood count keeps both EuRoC and the 16 px ChArUco markers |
| Kalibr's 2-bit border decoded on its true lattice | EuRoC FP 66 → 256, LOO worse |
| Declaring the EuRoC / Hub camera model (`non_rectified` path) | EuRoC recall 86 → 66 %, Hub BC corners 0.07 → 0.36 px; deferred to the next release |

## Setup

| Item | Value |
| :--- | :--- |
| CPU | AMD EPYC-Milan, 8 vCPU, AVX2, KVM guest (`lscpu`, measurement session) |
| OS / toolchain | Linux 6.8.0-139-generic, rustc 1.92.0 |
| Build | `--release` wheel (`maturin develop --release`) |
| Accuracy runs | Full datasets, `--jobs 6` |
| Timing runs | `--jobs 1 --threads 1` (`RAYON_NUM_THREADS=1`), best of 2 per image, serial, idle machine |
| References | OpenCV 4.10 (pinned; NONE / SUBPIX / APRILTAG refinement), aruco_nano `961b18b` |
