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

Snippet-included from the README so the numbers stay consistent
across the GitHub landing, the PyPI page, and this docs page —
single source of truth.

--8<-- "README.md:icra-comparison"

--8<-- "README.md:render-tag-comparison"

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
- [`render_tag_sota_20260425.md`](https://github.com/NoeFontana/locus-tag/blob/main/docs/engineering/benchmarking/render_tag_sota_20260425.md)
  — the snapshot the render-tag table above is sourced from, with
  detector-by-detector deep dive on rotation tails and recall gaps.
- [System architecture](architecture.md) — why the pipeline is
  shaped to release the GIL and avoid the system allocator on the
  hot path.
- [Memory model](memory_model.md) — SoA `DetectionBatch`, arena
  allocation, and the FFI zero-copy contract that makes the latency
  numbers possible.
