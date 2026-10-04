# Performance

Locus optimises for **high recall**, **low corner error**, and **low
latency**. This page summarises the current comparison against other
detectors and keeps older snapshots for context, each labelled with its
date. The [benchmarking deep-dive](../engineering/benchmarking.md)
documents methodology, hardware, and per-stage timing.

## Profiles

The shipped profiles are authored in JSON
(`crates/locus-core/profiles/*.json`) and embedded into the wheel.
Start from a profile, edit one or two fields, and hand the result
back to the detector — see the [Detection guide](../tutorials/guide.md)
for the `DetectorConfig` API and what each profile runs.

--8<-- "README.md:performance-profiles"

## Current results (2026-10-04)

Measured with `cargo xtask sota` against **pinned** references —
OpenCV `aruco` 4.10.0 with its three corner refiners (`NONE`, `SUBPIX`,
`APRILTAG`) and aruco_nano `961b18b` — run as published. Each cell
compares Locus `standard` with the best reference operating point on that
metric. 1 thread for latency; AMD EPYC-Milan, `--release`.

| Benchmark | Data | Metric | Locus `standard` | Best reference |
| :--- | :--- | :--- | :---: | :---: |
| EuRoC `cam_april` | Real camera, Kalibr AprilGrid (2-bit border), strong lens distortion | Recall | **86.0 %** | 57.1 % (OpenCV) |
| | | Precision | **99.996 %** | 99.993 % (OpenCV `APRILTAG`) |
| | | Leave-one-out corner error, median / p90 | **0.283 / 0.575 px** | 0.516 / 0.996 px |
| | | Latency per image | 9.5 ms | 8.3 ms |
| Liu4K | Real 4K photos, ArUco MIP 36h12 | Recall | 37.2 % | **66.3 %** (aruco_nano) |
| render-tag 1080p | Blender, single tag36h11 | Corner error, debiased mean / p90 | **0.062 / 0.109 px** | 0.230 / 0.323 px |
| render-tag 4K | | Corner error, debiased mean / p90 | **0.050 / 0.083 px** | 0.221 / 0.314 px |
| ChArUco render | Board, ArUco 6x6_250 | Corner error, debiased mean / p90 | **0.032 / 0.052 px** | 0.170 / 0.233 px |
| ICRA 2020 forward | Rendered, dense tag36h11 | Recall | **73.8 %** | 53.7 % (aruco_nano) |

- **EuRoC** is the real-camera benchmark with a verifiable board model:
  presence and precision come from a board homography fitted after
  undistortion, and corner error from leave-one-tag-out on tags every
  detector found. Locus decodes about four times as many tags per frame as
  before the 2026-10-04 changes; per decoded tag it is faster than every
  reference, per frame it is slower.
- **Liu4K** recall is an open gap, traced to the threshold model
  ([recall lessons](../engineering/lessons/recall-quad-icra.md)).
- **Debiased corner error** removes each detector's mean radial offset on
  the benchmark before scoring. A tone curve shifts every gradient edge the
  same way, so that offset belongs to the dataset; it is reported separately
  in the scoreboard (references sit 0.6–1.1 px inward on render-tag, Locus
  within 0.03 px after its [photometric inset calibration](algorithms.md#26-marker-photometric-inset-calibration)).

Sources: the [EuRoC report](../engineering/benchmarking/euroc_sota_20261004.md)
(EuRoC and Liu4K rows, current `standard`) and the
[2026-10-04 scoreboard](../engineering/benchmarking/sota_scoreboard_20261004.md)
(render-tag, ChArUco and ICRA rows, measured on the #434 build that
preceded the EuRoC changes). Both list the full win tables, latency per
benchmark, and verified hardware.

## Single-tag pose on render-tag (2026-07 snapshot)

`render-tag` is our in-house render suite — Blender with calibrated PSF,
exposure, sensor noise, and lens distortion models, with pixel-accurate ground
truth for corners and 6-DOF pose. The table below is the **2026-07-13**
single-threaded snapshot on the 1080p 50-scene subset (OpenCV 5.0.0 via the
Python wheel, re-tuned), with the `high_accuracy` row refreshed 2026-07-19 for
v0.7.0 model-edge refinement. It **predates** decode-first ordering,
4-connectivity and the corner stage in `standard` (2026-10), so the `standard`
row is historical. See
[`render_tag_sota_20260713.md`](../engineering/benchmarking/render_tag_sota_20260713.md)
for methodology and the 2160p table.

| Detector (2026-07) | Recall | Trans p50 | Trans p99 | Rot p50 | Rot p99 | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Locus (`high_accuracy`)** | **100 %** | **0.4 mm** | **20.1 mm** | **0.041 °** | **0.249 °** | **15.2 ms** |
| Locus (`standard`, pre-2026-10) | 100 % | 3.5 mm | 50.3 mm | 0.288 ° | 27.248 ° | 32.7 ms |
| OpenCV (`cv2.aruco`, subpix) | 100 % | 3.5 mm | 66.6 mm | 0.127 ° | 0.569 ° | 101.1 ms |
| OpenCV (`cv2.aruco`, apriltag) | 100 % | 3.0 mm | 55.3 mm | 0.067 ° | 0.376 ° | 195.8 ms |
| AprilTag-C (pupil) | 100 % | 2.9 mm | 54.4 mm | 0.061 ° | 65.365 ° | 78.5 ms |

In that snapshot `high_accuracy` had the lowest translation **and** rotation
tails (0.249° p99, below OpenCV `apriltag`'s 0.376°). AprilTag-C's median
rotation was best in class but its p99 reached 65° on symmetric-tag
branch-ambiguity failures. The single-tag pose effect of the 2026-10 corner
changes on `standard` is analysed in the
[rotation-tail lessons](../engineering/lessons/rotation-tail-and-edge-refinement.md#2026-10-04-the-marker-calibrates-its-own-photometric-inset).

> **Model-edge pose refinement.** As of v0.7.0, `high_accuracy` ships with
> `pose.pose_edge_refinement_enabled = True`: an Accurate-mode stage that refines
> each decoded tag's pose against its ~40 internal bit-grid edges (rotation from
> the distributed edges; translation re-anchored to the corners). It took
> `high_accuracy` rotation p99 from 0.600° to **0.249°** (p95 **0.180°**) at ~2.7×
> better translation and +~1 ms/frame. It requires camera intrinsics +
> `tag_size` (a no-op without them). `standard` and `grid` leave it off. See
> [`model_edge_refinement_20260715.md`](../engineering/benchmarking/model_edge_refinement_20260715.md).

!!! note "Retired: ICRA 2020 table (April 2026)"
    Earlier versions of this page reported ICRA 2020 forward recall of 96.2 % at
    0.315 px RMSE for `standard`. Those numbers come from the
    [2026-04-18 release report](../engineering/benchmarking/release_performance_20260418.md),
    measured on v0.3.1 with a soft-decision decoding mode that has since been
    removed, through the Python bench CLI rather than `cargo xtask sota`. The
    current ICRA comparison is the `cargo xtask sota` row above.

## How to read these numbers

- **Recall** — fraction of ground-truth tags whose ID was correctly
  decoded. Recall counts a detection toward the corner / pose
  distributions even if its corners or pose are poor, so per-percentile
  columns are how we surface fail-loudly cases (an `r p99` of 65 ° is the
  symptom of a few catastrophic branch-ambiguity failures, not a
  distribution-wide regression).
- **Corner error** — per-tag RMSE of the four corners against ground
  truth, in pixels, order-preserving (a wrong orientation counts as a large
  error). The comparative tables use the debiased form described above.
- **Translation / rotation percentiles** — `t p99` and `r p99` are
  the **tail metrics** we care most about for AV / robotics work.
  A robot that loses pose once per thousand frames is more dangerous
  than one that's slightly less accurate on every frame. Medians
  hide tail failures; we never accept a profile change that
  improves median at the cost of p99.
- **Latency** — wall-clock per-frame on a single rayon thread
  (`RAYON_NUM_THREADS=1`), image decode excluded. Multi-thread scaling is
  documented in the [Concurrent detection how-to](../how-to/concurrent_detection.md).

## Choosing a profile

| Workload | Recommended profile | Why |
|---|---|---|
| General detection, real cameras | `"standard"` | Highest real-data recall and the best debiased corner accuracy in the 2026-10-04 comparison. |
| Calibration boards with low-contrast prints | `"grid"` | `standard`'s pipeline with the blob-shape gates kept, sharpening off and lower contrast gates. |
| Single-tag metrology, high-resolution near-field, AV pose | `"high_accuracy"` | EdLines + adaptive PPB + model-edge pose refinement: the lowest single-tag rotation tail in the 2026-07 render-tag snapshot. Needs intrinsics + `tag_size`. |

## Related reading

- [Benchmarking methodology](../engineering/benchmarking.md) — how
  the recall / corner / latency numbers are measured, what hardware
  they ran on, and the regression suites that keep them honest.
- [Detection pipeline](pipeline.md) — what each profile runs.
- [System architecture](architecture.md) — why the pipeline is
  shaped to release the GIL and avoid the system allocator on the
  hot path.
- [Memory model](memory_model.md) — SoA `DetectionBatch`, arena
  allocation, and the FFI zero-copy contract that makes the latency
  numbers possible.
