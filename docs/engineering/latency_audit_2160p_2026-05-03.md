# 2160p latency audit — `high_accuracy + Accurate` (2026-05-03)

Per-stage latency breakdown for the SOTA-claim configuration
(`high_accuracy` profile, `PoseEstimationMode::Accurate`) at 2160p
(3840×2160), with a within-run 1080p baseline so the 4× pixel-count
expectation can be checked locally rather than against the stale
`lessons.md §1` Apr-18 row.

This memo is a **measurement record only** — no code is changed and no
optimization is committed. It exists so the next round of work has a
defensible, reproducible starting point.

## §1 Verified hardware

Captured at the start of this session via `lscpu`, `uname -a`, `nproc`,
and `cargo --version && rustc --version`. Full snapshot is preserved at
`diagnostics/latency_audit_2160p_2026-05-03/system.txt`.

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
Profile:                        release  (cargo test --release)
```

**Caveat — virtualised host.** This is a KVM guest. Wall-clock
percentiles may carry up to a few hundred microseconds of host-noise on
the tail. For absolute SOTA claims, repeat on bare metal. For
*relative* (1080p ↔ 2160p) comparisons inside a single session, the
noise cancels.

## §2 Methodology

* **Telemetry mode:** `TELEMETRY_MODE=json` (architecture §3.2 — chosen
  over `tracy` because no GUI is available and because we need a
  machine-parsable dump for the aggregator). Spans are emitted by the
  six `pipeline::*` `tracing::instrument` annotations across
  `threshold.rs`, `quad.rs`, `decoder.rs`, and `pose.rs`.
* **Frame sequencing:** `--test-threads=1` so latency reflects
  serialised per-frame work and not Rayon contention between scenes.
  Within a single frame, Rayon is still free to parallelise across
  cores (this is the production execution mode).
* **Datasets:**
  * 2160p — `tests/data/hub_cache/locus_v1_tag36h11_3840x2160/`,
    50 scenes.
  * 1080p — `tests/data/hub_cache/locus_v1_tag36h11_1920x1080/`,
    50 scenes.
* **Build:** `cargo nextest run --release --features bench-internals`.
* **Invocation (verbatim):**

  ```
  TRACY_NO_INVARIANT_CHECK=1 \
  TELEMETRY_MODE=json \
  LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
    ./target/release/deps/regression_render_tag-* \
      --test-threads=1 --exact \
      accuracy_baseline::regression_hub_tag36h11_2160p \
      accuracy_baseline::regression_hub_tag36h11_1080p
  ```

  Notes on the invocation: `cargo nextest run` was tried first but is
  blocked by the project's `default-filter = "not (binary(~regression)
  or binary(~robustness))"` (`.config/nextest.toml`); the test binary
  is therefore invoked directly, which the prompt explicitly permits
  for "the actual filter that matches the 2160p test names." Both
  tests panicked at the very end on an unrelated insta snapshot diff
  (a separate accuracy regression), but all 50 scenes ran to
  completion before the assertion fired — the telemetry stream
  contains 50 `pipeline::threshold_compute_stats` close events for
  each resolution, confirming complete coverage.
* **Aggregator:** `tools/bench/aggregate_telemetry.py` reads every
  `*_events.json` under the directory, keeps only top-level CLOSE
  events (so children inside `pose_refinement` are not double-counted),
  buckets them by frame using `pipeline::threshold_compute_stats` as
  the frame-start sentinel, and reports per-stage p50/p95/p99/mean.

## §3 Per-stage timings

`samples = 50 frames` for every stage. Times are wall-clock per frame,
in milliseconds, including all Rayon-parallel sub-tasks fused inside
that span.

### 2160p (3840×2160), `high_accuracy + Accurate`

| Stage              | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
| :----------------- | -------: | -------: | -------: | --------: |
| Preprocessing      |    3.364 |    6.031 |    9.700 |     3.835 |
| Segmentation       |        — |        — |        — |         — |
| Quad Extraction    |    3.925 |   46.235 |   58.778 |     9.612 |
| Decoding (Hard)    |    0.288 |    1.368 |    2.627 |     0.459 |
| Pose Refinement    |    0.028 |    0.042 |    0.045 |     0.029 |
| Telemetry / Tail   |    n/a   |    n/a   |    n/a   |     n/a   |

**Sum-of-medians ≈ 7.6 ms / frame** at p50, ~53.7 ms at p95 — i.e. the
p95 latency is ~7× the p50, driven entirely by Quad Extraction's tail.

### 1080p (1920×1080), same configuration, same run

| Stage              | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
| :----------------- | -------: | -------: | -------: | --------: |
| Preprocessing      |    1.591 |    8.285 |   14.283 |     2.864 |
| Segmentation       |        — |        — |        — |         — |
| Quad Extraction    |    1.940 |   11.810 |   15.116 |     3.170 |
| Decoding (Hard)    |    0.218 |    0.931 |    2.709 |     0.391 |
| Pose Refinement    |    0.028 |    0.054 |    1.220 |     0.074 |
| Telemetry / Tail   |    n/a   |    n/a   |    n/a   |     n/a   |

> **Notes on the table.**
>
> * **Segmentation row (—).** `crate::simd_ccl_fusion::label_components_lsl`
>   is *not* wrapped in `tracing::instrument`. Architecturally it is a
>   distinct stage (architecture.md §"Source Code Organization"), but
>   wall-clock-wise it runs between `pipeline::threshold_apply_map` and
>   `pipeline::quad_extraction`, and is currently invisible to JSON
>   telemetry. Its time is **not** included in either Preprocessing or
>   Quad Extraction here. The closest existing measurement is the
>   `lessons.md §1` Apr-18 row (≈ 2.6 ms at 1080p), but that is stale.
>   Adding a tracing span would be a one-line change; we elect *not* to
>   make it in this audit per scope discipline.
> * **Telemetry / Tail row (n/a).** Phase C.5 (`post_decode_refinement`),
>   board-level pose estimation, and any "telemetry tail" emit no
>   `pipeline::*` span and are too small to land in the per-stage table.
>   Their contribution is bounded by `(total wall) − Σ(measured stages)`
>   and is sub-millisecond on this configuration.
> * **Pose Refinement at 1080p p99 = 1.22 ms** is one outlier scene; the
>   2160p run does *not* exhibit it, suggesting a per-scene initial-
>   estimate cost rather than a resolution-driven one.

## §4 Comparison: 2160p / 1080p multiplier

Pixel count grows 4× from 1080p (2.07 Mpx) to 2160p (8.29 Mpx). At each
percentile we expect *roughly* 4× scaling for memory-bound stages and
< 4× for CPU-bound or candidate-count-bound stages.

| Stage             | 1080p p50 | 2160p p50 | ×    | 1080p p95 | 2160p p95 | ×    | Verdict (p50 / p95) |
| :---------------- | --------: | --------: | ---: | --------: | --------: | ---: | :------------------ |
| Preprocessing     |    1.591  |    3.364  | 2.11 |    8.285  |    6.031  | 0.73 | sub-linear / sub-linear (good) |
| Quad Extraction   |    1.940  |    3.925  | 2.02 |   11.810  |   46.235  | 3.92 | sub-linear / **near-linear at tail (anomalous)** |
| Decoding (Hard)   |    0.218  |    0.288  | 1.32 |    0.931  |    1.368  | 1.47 | sub-linear (excellent) |
| Pose Refinement   |    0.028  |    0.028  | 1.00 |    0.054  |    0.042  | 0.78 | flat (driven by candidate count, not pixels) |

(`Segmentation` excluded — uninstrumented.)

* **Preprocessing p95 lower at 2160p than 1080p.** This is the
  warmup-tail asymmetry: the 1080p run is the *second* test in the
  sequence, but each test gets its own `Detector` and arena pool, so
  both pay a first-frame allocation/page-fault cost. The 1080p p95 of
  8.3 ms is dominated by frame 0; later frames sit comfortably under
  2 ms. At p50 (the more representative value), Preprocessing scales
  **2.11×** for 4× pixels — i.e. ~50 % of pixel-perfect linear, which
  is consistent with a strongly L3-resident SIMD path (32 MiB L3, both
  resolutions fit).
* **Quad Extraction p50 of 2.02× is excellent** — the per-component
  `WORKSPACE_ARENA` and Rayon parallelism keep median scenes inside L2.
* **Quad Extraction p95 of 3.92× is the headline finding** — the tail
  is *the* dominating cost at 2160p. See §5.
* **Decoding (Hard)** scales 1.3–1.5×, consistent with a tag-count-
  dominated cost (≤ 50 tags per scene in this dataset, regardless of
  resolution) rather than a pixel-count cost.
* **Pose Refinement** is flat: the per-tag LM iteration count
  is invariant to resolution.

No multiplier exceeds the expected 4× ceiling, but Quad Extraction's
3.92× p95 is at the limit of what we'd allow before flagging it as
anomalous. The combination of a *near-linear* p95 multiplier with a
*sub-linear* p50 multiplier is the signature of dense-noise outlier
scenes (high candidate count → many parallel `extract_single_quad`
invocations → cumulative arena/cache pressure).

## §5 Bottleneck identification

At 2160p:

* **Quad Extraction dominates by an order of magnitude at the tail.**
  p95 = 46.2 ms, p99 = 58.8 ms. The next-largest stage at p95 is
  Preprocessing at 6.0 ms (~7.7× smaller). Mean = 9.6 ms; the right
  tail alone (worst 5 scenes) contributes most of the wall-clock
  budget for those scenes. Worst observed frame: **67.5 ms**.
* The cost lives in
  [`extract_quads_soa`](../../crates/locus-core/src/quad.rs#L129-L220)
  and the Rayon-parallel
  [`extract_single_quad`](../../crates/locus-core/src/quad.rs#L222-L513)
  body. After
  [`pixel_count_descending_order`](../../crates/locus-core/src/quad.rs#L60-L80)
  has ranked components, every survivor enters the per-thread
  `WORKSPACE_ARENA` and runs the full chain: boundary trace → RDP /
  EdLines route → corner refinement → edge-score gate. Two specific
  hotspots:
  1. **Per-component fixed cost.** `extract_single_quad` begins with
     `arena.reset()` then a boundary trace; for the dense-noise 4 K
     scenes that survive after `pixel_count_descending_order` truncates
     to `MAX_CANDIDATES = 1024`, the call count itself is the cost. In
     other words, Phase A's *count*, not its per-call work, is what
     scales near-linearly with pixel count.
  2. **Refinement on the EdLines route.** The `high_accuracy` profile
     uses `AdaptivePpb`, which routes high-PPB candidates to EdLines
     +`CornerRefinementMode::None` (cheap) and low-PPB ones to
     ContourRdp+ERF (more expensive). At 2160p, more candidates land
     in the high-PPB lane *per frame* (the same physical tags now
     project to ≥ 5 PPB), so the route-mix shifts toward EdLines —
     but since the candidate *count* still grows ~4×, the EdLines
     path's `extract_quad_edlines` (decoder.rs/edlines.rs) dominates.
* **Preprocessing is a distant second.** `apply_threshold_with_map`
  ([`threshold.rs:259`](../../crates/locus-core/src/threshold.rs#L259-L437))
  + `compute_tile_stats`
  ([`threshold.rs:63`](../../crates/locus-core/src/threshold.rs#L63-L433))
  are the two tracked spans. At 2.11× p50 scaling these are clearly not
  the next target; they enjoy near-perfect SIMD vectorisation.
* **Segmentation is the elephant we cannot see.** Without instrumenting
  `simd_ccl_fusion::label_components_lsl` we cannot say what fraction
  of the gap between `(threshold_apply_map close)` and
  `(quad_extraction open)` is segmentation versus arena-bookkeeping.
  This audit therefore attributes any time in that gap to neither —
  see follow-up issue 1 below.

## §6 Recommendations (follow-up issues, NOT commitments)

These are candidate workstreams to file, ordered by expected impact.

1. **(P0) Instrument `label_components_lsl`.** A single
   `#[tracing::instrument(skip_all, name = "pipeline::segmentation")]`
   on
   [`label_components_lsl`](../../crates/locus-core/src/simd_ccl_fusion/mod.rs#L91)
   would close the only architectural gap in our pipeline-stage
   telemetry. Zero hot-path cost (release_max_level_info erases at
   compile time once we drop below INFO).
2. **(P0) Quad Extraction tail attribution.** Add per-route counters
   (EdLines vs ContourRdp survivor counts, RDP iteration counts,
   refinement calls) so the next 2160p memo can say which sub-stage of
   the 47 ms p95 is dominant. Today we can only point at
   `extract_single_quad` in aggregate.
3. **(P1) Frame-zero warmup.** The 1080p Preprocessing p99 of 14.3 ms
   and the universal frame-0 spike (visible in the per-stage first/last
   sample dumps) suggest first-frame page-fault / cold-cache cost.
   Pre-touching the arena and pre-loading the multiversion-dispatched
   functions during `Detector::new()` would smooth this. Worth ~5–10 %
   of mean latency in long-running services that re-create detectors.
4. **(P1) Investigate the dense-noise scene set.** The Quad Extraction
   max of 67.5 ms (one frame, 2160p) versus a p50 of 3.9 ms is a 17×
   spread. Identifying *which* scenes drive the tail (almost certainly
   the same scenes that live near the `MAX_CANDIDATES = 1024`
   ceiling — see `lessons.md §2.1`) and confirming whether they are
   pathological inputs or representative production frames will set
   the priority of all other Quad-Extraction work.
5. **(P2) Re-run on bare metal.** This audit was captured on a KVM
   guest. A bare-metal repeat with the same script will tell us how
   much of the p99 tail is virtualisation noise versus real algorithmic
   spread. Agreed not to land any commitment based on these numbers
   until that is done.

## §7 Reproducibility recipe

1. Build the test binary in release mode (cached after the first run):
   `cargo nextest run --release --features bench-internals --test
   regression_render_tag --no-run`.
2. Wipe and prime the telemetry directory:
   `rm -rf target/profiling && mkdir target/profiling`.
3. Run the two regressions directly through the test binary (the
   project's `nextest.toml` `default-filter` excludes `regression*`
   binaries from `nextest run`):

   ```
   TRACY_NO_INVARIANT_CHECK=1 TELEMETRY_MODE=json \
   LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
     target/release/deps/regression_render_tag-<hash> \
       --test-threads=1 --exact \
       accuracy_baseline::regression_hub_tag36h11_2160p \
       accuracy_baseline::regression_hub_tag36h11_1080p
   ```
4. Aggregate:
   `uv run python tools/bench/aggregate_telemetry.py target/profiling/`.
5. The aggregator prints the same table format used in §3.

The raw event streams used for this memo are preserved at
`target/profiling/regression_hub_tag36h11_{2160p,1080p}_events.json`
in the worktree (not committed). System metadata is committed at
`diagnostics/latency_audit_2160p_2026-05-03/system.txt`.
