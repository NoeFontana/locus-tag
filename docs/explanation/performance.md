# Performance

Locus optimises for **high recall**, **low corner RMSE**, and **low
latency**. This page surfaces the headline numbers across the two
datasets we regression-test against. The
[benchmarking deep-dive](../engineering/benchmarking.md) documents
methodology, hardware, and per-stage timing.

## Profiles

The shipped profiles are authored in JSON
(`crates/locus-core/profiles/*.json`) and embedded into the wheel.
Start from a profile, edit one or two fields, and hand the result
back to the detector — see the [Detection guide](../tutorials/guide.md)
for the `DetectorConfig` API.

--8<-- "README.md:performance-profiles"

## Two benchmark suites

The performance numbers below come from two regression-tested
benchmarks that exercise complementary regimes. We track them
independently and **do not trade gains on one for regressions on the
other** (see `feedback_dataset_priority` in the engineering memory).

| Suite | Frames | Render quality | Ground truth | Used for |
|---|---|---|---|---|
| ICRA 2020 Forward | 50 | Lower-fidelity synthetic | Tag IDs + corners | Continuity with the published AprilTag-community comparison |
| `render-tag` | 50 (1080p subset) | High-fidelity Blender + PSF + sensor model | IDs + corners + 6-DOF pose | Pose-accuracy SOTA tracking, internal CI gate |

### ICRA 2020 Forward (community benchmark)

[ICRA 2020 Forward](https://github.com/aprilrobotics/apriltag-comparison)
is the closest thing the AprilTag community has to a neutral benchmark.
The 50-frame subset we report on is **synthetic** (not real-camera), but
it's public, peer-reviewed, and the basis for prior detector comparisons —
we report on it for continuity with the literature.

| Detector | Recall | Corner RMSE |
| :--- | :---: | :---: |
| **Locus (`standard`)** | **96.2 %** | **0.315 px** |
| AprilTag 3 (UMich) | 62.3 % | 0.22 px |
| OpenCV (`cv2.aruco`) | 52.6 % | 0.98 px |

The OpenCV row is its recall-best OpenCV 5.0 config (tuned `subpix`); the
tag-aware `apriltag` refinement more than halves corner RMSE (0.39 px) but
rejects ICRA's marginal small tags, dropping recall to ~30 %.

### render-tag (high-fidelity Blender + PSF)

`render-tag` is our in-house render suite — Blender with calibrated PSF,
exposure, sensor noise, and lens distortion models. The detection scenes
carry pixel-accurate ground truth for both corners and 6-DOF pose, which
lets us report translation / rotation percentiles in addition to recall.
Numbers below are the 2026-07-13 single-threaded SOTA snapshot on the 1080p
50-scene subset (OpenCV 5.0.0, re-tuned), with the `high_accuracy` row
refreshed 2026-07-19 for the v0.7.0 model-edge-refinement default (same
hardware; competitor and `standard` rows unchanged). See
[`render_tag_sota_20260713.md`](../engineering/benchmarking/render_tag_sota_20260713.md)
for methodology, the 2160p table, and OpenCV's two operating points.

| Detector | Recall | Trans p50 | Trans p99 | Rot p50 | Rot p99 | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Locus (`high_accuracy`)** | **100 %** | **0.4 mm** | **20.1 mm** | **0.041 °** | **0.249 °** | **15.2 ms** |
| Locus (`standard`) | 100 % | 3.5 mm | 50.3 mm | 0.288 ° | 27.248 ° | 32.7 ms |
| OpenCV (`cv2.aruco`, subpix) | 100 % | 3.5 mm | 66.6 mm | 0.127 ° | 0.569 ° | 101.1 ms |
| OpenCV (`cv2.aruco`, apriltag) | 100 % | 3.0 mm | 55.3 mm | 0.067 ° | 0.376 ° | 195.8 ms |
| AprilTag-C (pupil) | 100 % | 2.9 mm | 54.4 mm | 0.061 ° | 65.365 ° | 78.5 ms |

Locus `high_accuracy` wins the translation tail **and** the rotation tail
(0.249° p99, below OpenCV `apriltag`'s 0.376°) while staying ~13× faster than
OpenCV's best-accuracy `apriltag` config — the model-edge pose refinement that
ships on in `high_accuracy` (v0.7.0) is what closes that rotation gap. OpenCV
ships two operating points — fast `subpix` and accurate-but-~2×-slower
`apriltag`. AprilTag-C's median rotation is best in class (0.06°) but its p99
explodes to 65° on symmetric-tag IRLS branch-ambiguity failures.

> **Model-edge pose refinement.** As of v0.7.0, `high_accuracy` ships with
> `pose.pose_edge_refinement_enabled = True`: an Accurate-mode stage that refines
> each decoded tag's pose against its ~40 internal bit-grid edges (rotation from
> the distributed edges; translation re-anchored to the corners). It takes
> `high_accuracy` rotation p99 from 0.600° to **0.249°** (p95 **0.180°**) at ~2.7×
> better translation and +~1 ms/frame, with 2D corner RMSE unchanged and
> reprojection RMSE improved. It requires camera intrinsics + `tag_size` (a no-op
> without them). `standard` and `grid` leave it off; set the flag to `False` to opt
> out on `high_accuracy`. See
> [`model_edge_refinement_20260715.md`](../engineering/benchmarking/model_edge_refinement_20260715.md).

## How to read these numbers

- **Recall** — fraction of ground-truth tags whose ID was correctly
  decoded. Recall counts a detection toward the corner / pose
  distributions even if its corners or pose are poor, so per-percentile
  RMSE / translation / rotation columns are how we surface
  fail-loudly cases (an `r p99` of 65 ° is the symptom of a small
  number of catastrophic branch-ambiguity failures, not a
  distribution-wide regression).
- **Corner RMSE** — root-mean-square Euclidean error of detected
  corners against ground truth, in pixels. Lower is better; the LM
  pose solver consumes the per-corner covariance and propagates it
  into the 6-DOF pose covariance, so corner RMSE is a leading
  indicator of pose precision.
- **Translation / rotation percentiles** — `t p99` and `r p99` are
  the **tail metrics** we care most about for AV / robotics work.
  A robot that loses pose once per thousand frames is more dangerous
  than one that's slightly less accurate on every frame. Medians
  hide tail failures; we never accept a profile change that
  improves median at the cost of p99.
- **Latency** — wall-clock per-frame on a single rayon thread
  (`RAYON_NUM_THREADS=1`). Multi-thread scaling is documented in
  the [Concurrent detection how-to](../how-to/concurrent_detection.md).

## Choosing a profile

| Workload | Recommended profile | Why |
|---|---|---|
| General detection | `"standard"` | Highest ICRA recall + balanced corner accuracy. |
| Calibration boards (ChArUco, AprilGrid) | `"grid"` | 4-connectivity recovers touching tags that `"standard"` rejects as overlapping. |
| Sub-pixel metrology, high-resolution near-field, AV pose | `"high_accuracy"` | EdLines + adaptive PPB + axis-imbalance gate. **Best on `render-tag` across every translation percentile**; trades 6 pp of ICRA recall for the pose-tail control. |

## Related reading

- [Benchmarking methodology](../engineering/benchmarking.md) — how
  the recall / RMSE / latency numbers are measured, what hardware
  they ran on, and the regression suite that keeps them honest.
- [render-tag SOTA snapshot](../engineering/benchmarking/render_tag_sota_20260713.md)
  — the source of the render-tag table above, with the detector-by-detector
  deep dive, the 2160p table, and OpenCV's two operating points.
- [Engineering lessons](../engineering/lessons/README.md) — distilled
  conclusions on the pose rotation tail, EdLines, covariance, and recall/quad.
- [System architecture](architecture.md) — why the pipeline is
  shaped to release the GIL and avoid the system allocator on the
  hot path.
- [Memory model](memory_model.md) — SoA `DetectionBatch`, arena
  allocation, and the FFI zero-copy contract that makes the latency
  numbers possible.
