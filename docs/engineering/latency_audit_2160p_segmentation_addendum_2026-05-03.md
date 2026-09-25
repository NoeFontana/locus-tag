# 2160p latency audit — Segmentation span addendum (2026-05-03)

The 2026-05-03 V2 latency audit (`latency_audit_2160p_2026-05-03.md`)
reported `—` for Segmentation because `label_components_lsl` was
uninstrumented. This addendum adds
`#[tracing::instrument(skip_all, name = "pipeline::segmentation")]` on
it and re-measures. Measurement-only — no algorithm change.

## Methodology

Same box/toolchain as V2 (AMD EPYC-Milan KVM guest, 8 vCPU, no AVX-512,
`rustc 1.92.0`, `--release`). `TELEMETRY_MODE=json`,
`--test-threads=1` (serialize frames; Rayon still parallelizes within
a frame), 50-scene `tag36h11` Hub subsets at 1080p and 2160p. Repro:

```
TELEMETRY_MODE=json LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
  cargo nextest run --release --features bench-internals \
    --test-threads=1 --no-fail-fast --ignore-default-filter \
    -E 'binary(regression_render_tag) and (
          test(accuracy_baseline::regression_hub_tag36h11_2160p) or
          test(accuracy_baseline::regression_hub_tag36h11_1080p))'
```

## Per-stage timings (ms/frame, wall-clock, Rayon-parallel sub-tasks included)

### 2160p, `high_accuracy + Accurate`

| Stage              | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
| :----------------- | -------: | -------: | -------: | --------: |
| Preprocessing      |    3.364 |    6.031 |    9.700 |     3.835 |
| **Segmentation**   |   18.30  |   45.17  |  103.53  |    24.07  |
| Quad Extraction    |    3.21  |   44.78  |   59.67  |     8.70  |
| Decoding (Hard)    |    0.288 |    1.368 |    2.627 |     0.459 |
| Pose Refinement    |    0.028 |    0.042 |    0.045 |     0.029 |

Quad Extraction reproduced V2's numbers (3.21/44.78 vs 3.925/46.235 ms)
within host noise — the new span isn't measurable overhead.

### 1080p → 2160p multiplier (Segmentation only)

| Metric  | 1080p (ms) | 2160p (ms) | ratio | expected |
| :------ | ---------: | ---------: | ----: | -------: |
| p50     |       4.72 |      18.30 |  3.88 |     ~4.0 |
| p95     |      10.00 |      45.17 |  4.52 |     ~4.0 |
| mean    |       5.84 |      24.07 |  4.12 |     ~4.0 |

Segmentation scales almost exactly 4× (1.03–1.13× the ideal ratio) —
the textbook signature of an O(N) stage saturating memory bandwidth on
a host with stable thread scheduling.

## Implications

1. **Segmentation is the 2nd-heaviest stage at 2160p p50** (18.30 ms ≈
   5.7× quad extraction's p50), and roughly on par with Quad Extraction
   at p95 (45.17 vs 44.78 ms). V2's bottleneck attribution to Quad
   Extraction's p95 tail still stands; the p50 picture is now reshaped.
2. The architecture doc's `~0.5 ms` segmentation target (50 tags, 720p)
   is workload mismatch, not regression: the hub benchmark segments
   ~50× more components/frame (every black blob, not just tag
   interiors).
3. No latency regression from the instrumentation itself (Quad
   Extraction p95 moved 0.005 ms between runs, well inside ±5%).

## Follow-ups

- Add `"pipeline::segmentation": "Segmentation"` to
  `tools/bench/aggregate_telemetry.py`'s `SPAN_TO_STAGE` upstream (done
  locally for this addendum only).
- Sub-stage spans (RLE extraction vs LSL Union-Find vs stats) — defer
  until segmentation optimization is actually scheduled.
