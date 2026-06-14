# Performance

Locus is built for bounded latency and high recall on real-world
robotics / AV camera imagery. This page surfaces the headline
numbers; the [benchmarking deep-dive](../engineering/benchmarking.md)
documents methodology, hardware, and per-stage timing.

## Headline metrics

The tables below are reproduced from the project README so the
numbers stay consistent across the GitHub landing, the PyPI page, and
the docs site (snippet-included from `README.md` — single source of
truth).

--8<-- "README.md:performance-tables"

## How to read these numbers

- **Recall** measures the fraction of ground-truth tags the detector
  correctly localised and decoded (correct ID + four corners within
  RMSE budget). 100 % is unachievable in practice — heavy motion blur,
  oblique incidence, and partial occlusion cap real-world recall.
- **Corner RMSE** is the root-mean-square Euclidean error of detected
  corners against ground truth, in pixels. Lower is better for
  pose-recovery accuracy; the LM solver consumes the per-corner
  covariance and propagates it into the 6-DOF pose covariance.
- **ICRA 2020 Forward** is a public, peer-reviewed dataset of 50
  real-camera frames. We use it because it's the closest thing the
  community has to a neutral benchmark; render-tag and Hugging Face
  Hub datasets exercise different regimes (see
  [benchmarking](../engineering/benchmarking.md)).

## Choosing a profile

| Workload | Recommended profile | Why |
|---|---|---|
| General detection | `"standard"` | Highest ICRA recall + balanced corner accuracy. |
| Calibration boards (ChArUco, AprilGrid) | `"grid"` | 4-connectivity recovers touching tags that `"standard"` rejects as overlapping. |
| Sub-pixel metrology, near-field | `"high_accuracy"` | EdLines + GN optimizer trades recall for ~2× corner precision. |

The shipped profiles are authored in JSON
(`crates/locus-core/profiles/*.json`) and embedded into the wheel.
Start from a profile, edit a single field, and hand the result back to
the detector — see the [Detection guide](../tutorials/guide.md) for
the `DetectorConfig` API.

## Related reading

- [Benchmarking methodology](../engineering/benchmarking.md) — how the
  recall / RMSE numbers are measured, what hardware they ran on, and
  the regression suite that keeps them honest.
- [System architecture](architecture.md) — why the pipeline is
  shaped to release the GIL and avoid the system allocator on the
  hot path.
- [Memory model](memory_model.md) — SoA `DetectionBatch`, arena
  allocation, and the FFI zero-copy contract that makes the latency
  numbers possible.
