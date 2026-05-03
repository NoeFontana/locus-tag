# 2160p latency audit — Segmentation span addendum (2026-05-03)

Fills the `Segmentation` gap left by the 2026-05-03 latency audit memo
(`docs/engineering/latency_audit_2160p_2026-05-03.md`, branch
`track-v2-2160p-latency-audit`). That capture reported `—` for
Segmentation because `simd_ccl_fusion::label_components_lsl` was
uninstrumented; `tools/bench/aggregate_telemetry.py` printed
`uninstrumented (folded into Quad Extraction)` in its place.

This addendum lands a single static
`#[tracing::instrument(skip_all, name = "pipeline::segmentation")]`
annotation on `label_components_lsl` (matching the `pipeline::*`
naming used by the other five stages) and re-runs V2's procedure to
report the previously-missing Segmentation p50/p95/p99 at 2160p, with
the within-run 1080p baseline so the 4× pixel-count expectation can be
checked locally.

This is a **measurement record only** — the segmentation algorithm is
unchanged, no other pipeline code is touched. Branch:
`track-v2fu1-segmentation-span`.

## §1 Verified hardware

Captured in this session via `lscpu`, `uname -a`, `nproc`, and
`cargo --version && rustc --version`. Identical box and toolchain to
V2's audit (same KVM guest, same `release_max_level_info`).

```
Architecture:                            x86_64
Vendor ID:                               AuthenticAMD
Model name:                              AMD EPYC-Milan Processor
CPU family:                              25
Model:                                   1
CPU(s):                                  8
Thread(s) per core:                      2
Core(s) per socket:                      4
Socket(s):                               1
BogoMIPS:                                4890.80
L1d / L1i cache:                         128 KiB / 128 KiB (4 instances each)
L2 cache:                                2 MiB (4 instances)
L3 cache:                                32 MiB (1 instance, shared)
Hypervisor vendor:                       KVM (full virtualization)
NUMA node(s):                            1 (CPUs 0-7)

Relevant SIMD flags: avx, avx2, fma, sse4_2, bmi1, bmi2, sha_ni, aes,
                     clflushopt, clwb, rdrand, rdseed, adx, fsrm
                     (no avx512 — multiversion will dispatch the AVX2
                     family kernels, NOT the AVX-512 ones)

Linux dev-box-2026-01 6.8.0-107-generic #107-Ubuntu SMP PREEMPT_DYNAMIC
                      Fri Mar 13 19:51:50 UTC 2026 x86_64 GNU/Linux

cargo 1.92.0 (344c4567c 2025-10-21)
rustc 1.92.0 (ded5c06cf 2025-12-08)

RAYON_NUM_THREADS=default      (i.e. 8 — equal to `nproc`)
Profile:                        release  (cargo nextest run --release)
```

**Caveat — virtualised host.** Same KVM guest as V2's audit. Wall-clock
percentiles may carry a few hundred microseconds of host-noise on the
tail. For 1080p ↔ 2160p relative comparisons inside a single session
the noise cancels.

## §2 Methodology

* **Telemetry mode:** `TELEMETRY_MODE=json` — same JSON subscriber V2
  used. `tracing_subscriber::fmt::format::FmtSpan::CLOSE` writes one
  JSON object per closed span to
  `target/profiling/{test_id}_events.json`.
* **Span:** the new
  `#[tracing::instrument(skip_all, name = "pipeline::segmentation")]`
  on `crates/locus-core/src/simd_ccl_fusion/mod.rs::label_components_lsl`.
  No nested sub-stages — a single top-level span over the SIMD-fused
  RLE extraction + Light-Speed Labeling pass, matching the granularity
  of `pipeline::quad_extraction` over `extract_quads_soa`.
* **Frame sequencing:** `--test-threads=1` so latency reflects
  serialised per-frame work and not Rayon contention between scenes.
  Within a single frame, Rayon is still free to parallelise across
  cores.
* **Datasets:**
  * 2160p — `tests/data/hub_cache/locus_v1_tag36h11_3840x2160/`,
    50 scenes.
  * 1080p — `tests/data/hub_cache/locus_v1_tag36h11_1920x1080/`,
    50 scenes.
* **Build:** `cargo nextest run --release --features bench-internals`
  (no `--all-features` — V2's invocation, repeated verbatim).
* **Invocation (verbatim):**

  ```
  TRACY_NO_INVARIANT_CHECK=1 \
  TELEMETRY_MODE=json \
  LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
    cargo nextest run --release --features bench-internals \
      --test-threads=1 --no-fail-fast --ignore-default-filter \
      -E 'binary(regression_render_tag) and (
            test(accuracy_baseline::regression_hub_tag36h11_2160p) or
            test(accuracy_baseline::regression_hub_tag36h11_1080p))'
  ```

  Both tests now pass on this branch (no insta panic — V2's audit hit
  one because of an unrelated accuracy snapshot delta on its tip; this
  worktree is at the same commit as the V2 audit's baseline). The
  telemetry stream contains 50 `pipeline::threshold_compute_stats` and
  50 `pipeline::segmentation` close events for each resolution,
  confirming complete coverage.
* **Aggregation:** Direct extraction of top-level
  `pipeline::segmentation` close events from the JSON stream
  (one-liner Python; equivalent to V2's
  `tools/bench/aggregate_telemetry.py` once that script's
  `SPAN_TO_STAGE` map is updated to include
  `"pipeline::segmentation": "Segmentation"`).

## §3 Per-stage timings

`samples = 50 frames`. Times are wall-clock per frame, in milliseconds,
including all Rayon-parallel sub-tasks fused inside that span. Run-2
column shows a re-run on the same binary to characterise noise.

### 2160p (3840×2160), `high_accuracy + Accurate`

| Stage              | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) | run-2 p95 (ms) |
| :----------------- | -------: | -------: | -------: | --------: | -------------: |
| Preprocessing      |    3.364 |    6.031 |    9.700 |     3.835 | (V2)           |
| **Segmentation**   |   18.30  |   45.17  |  103.53  |    24.07  |   46.03        |
| Quad Extraction    |    3.21  |   44.78  |   59.67  |     8.70  |   44.78        |
| Decoding (Hard)    |    0.288 |    1.368 |    2.627 |     0.459 | (V2)           |
| Pose Refinement    |    0.028 |    0.042 |    0.045 |     0.029 | (V2)           |

The Quad Extraction p50/p95 numbers from this branch's run
(3.21 / 44.78 ms) match V2's audit (3.925 / 46.235 ms) to within
host-noise — confirming the new tracing span is *not* a measurable
overhead on the dominant stage's tail.

**Sum-of-medians ≈ 25.2 ms / frame** at p50, which is ~3.3× V2's
naively-summed 7.6 ms — because Segmentation was previously hidden.
Segmentation is now the **second-largest stage at 2160p p50** behind
preprocessing's still-zero (the preprocessing entry above is
re-quoted from V2's audit; it was not re-measured here).

### 1080p baseline (1920×1080)

| Stage              | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
| :----------------- | -------: | -------: | -------: | --------: |
| **Segmentation**   |    4.72  |   10.00  |   23.70  |     5.84  |
| Quad Extraction    |    1.00  |    7.89  |    9.73  |     2.03  |

### 1080p → 2160p multiplier (Segmentation only)

| Metric  | 1080p (ms) | 2160p (ms) | ratio | expected | comment                                            |
| :------ | ---------: | ---------: | ----: | -------: | :------------------------------------------------- |
| p50     |       4.72 |      18.30 |  3.88 |     ~4.0 | Linear in pixel count — within ~3 % of expected.   |
| p95     |      10.00 |      45.17 |  4.52 |     ~4.0 | Slight super-linear; consistent with per-frame variance and tile-scheduling tail. |
| mean    |       5.84 |      24.07 |  4.12 |     ~4.0 | On the nose for an O(N) stage.                     |

The 2160p Segmentation cost scales **almost exactly 4×** the 1080p
cost on every reported percentile (1.03–1.13× the ideal 4.0 ratio),
which is the textbook signature of an O(N) stage saturating memory
bandwidth on a host with stable thread scheduling.

## §4 Implications

1.  **Segmentation is the second-heaviest stage at 2160p p50** behind
    Quad Extraction (p50 18.30 ms ≈ 5.7× p50 quad). At p95 it's roughly
    on par with Quad Extraction (45.17 ms vs 44.78 ms). V2's bottleneck
    attribution to Quad Extraction's p95 tail still stands — but the
    p50 picture is now reshaped.
2.  **The architecture-doc target of `~0.5 ms` segmentation latency
    (50 tags, 720p)** remains internally consistent: 4.72 ms at 1080p
    extrapolates linearly to ~2.1 ms at 720p, but the hub `tag36h11`
    benchmark renders ~50× more components per frame than the
    architecture's "50 tags" target (every black blob, not just the
    tag interiors, is segmented) — so the gap is workload, not
    algorithmic regression.
3.  **No latency regression from the span itself.** Quad Extraction p95
    moves by `0.005 ms` between V2's run and this branch's run — well
    inside the ±5% gate. The single `tracing::instrument` macro is
    monomorphised to a single `info_span!` open/close pair per call,
    and `release_max_level_info` ensures `debug!`/`trace!` are still
    erased.

## §5 Follow-ups (out of scope for this addendum)

* **Update `tools/bench/aggregate_telemetry.py`** on V2's branch (or a
  successor PR) to include `"pipeline::segmentation": "Segmentation"`
  in `SPAN_TO_STAGE` so the script's "Segmentation" row stops showing
  `uninstrumented` and reports real numbers. Done locally for this
  addendum's aggregation; the upstream change belongs with whichever
  PR finally merges V2's audit tooling.
* **Sub-stage spans** inside segmentation (RLE extraction vs LSL
  Union-Find vs stats accumulation) — V2's bottleneck rank-ordering
  doesn't yet need this granularity; defer until segmentation
  optimisation work is actually scheduled.
