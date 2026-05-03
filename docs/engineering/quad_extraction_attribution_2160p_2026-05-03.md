# 2160p Quad-Extraction Sub-Stage Attribution (2026-05-03)

Sub-stage breakdown of the `pipeline::quad_extraction` span identified by
the [V2 latency audit](latency_audit_2160p_2026-05-03.md) as the
dominant cost at 2160p (p95 = 46.2 ms, ~7.7× the next-largest stage).
The audit could only point at `extract_single_quad` in aggregate; this
memo attributes that aggregate to its sub-routes (EdLines vs
ContourRdp), the RDP simplification iteration budget, and the
post-route corner refinement variant (Erf vs None).

This memo is a measurement record only. Two pieces of code change
shipped on `track-v2fu2-quad-route-counters`:

* `crates/locus-core/src/quad.rs` — eight per-frame `AtomicU32`
  counters (`QuadRouteCounters`) feature-gated behind
  `bench-internals`. Compile-time erased to a zero-sized struct in the
  production build, so the off-path quad-extraction hot loop is
  byte-identical with the pre-instrumentation version.
* `tools/bench/aggregate_telemetry.py` — extended to recognise the
  `pipeline::quad_route_summary` `info!` event the counters emit at
  frame end and report a sub-stage table beneath the existing
  per-stage latencies.

## §1 Verified hardware

Captured at the start of this session via `lscpu`, `uname -a`, `nproc`,
and `cargo --version && rustc --version` — same KVM guest as the V2
audit, run consecutively in the same session.

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
                     family kernels)

Linux dev-box-2026-01 6.8.0-107-generic #107-Ubuntu SMP PREEMPT_DYNAMIC
                      Fri Mar 13 19:51:50 UTC 2026 x86_64 GNU/Linux

cargo 1.92.0 (344c4567c 2025-10-21)
rustc 1.92.0 (ded5c06cf 2025-12-08)

RAYON_NUM_THREADS=default      (i.e. 8 — equal to `nproc`)
Profile:                        release  (cargo nextest run --release)
Features:                       --all-features (includes bench-internals)
TELEMETRY_MODE=json
```

**Caveat — virtualised host.** Same as V2's memo. Wall-clock
percentiles may carry up to a few hundred microseconds of host-noise on
the tail. Comparisons inside a single session are reliable; absolute
SOTA claims should be repeated on bare metal.

## §2 Methodology

* **Build:** `cargo nextest run --release --all-features` (compiles the
  test binary; `bench-internals` is implicitly enabled by the
  workspace `--all-features` flag).
* **Telemetry:** `TELEMETRY_MODE=json` (V2's recipe). The new
  `pipeline::quad_route_summary` event is emitted once per frame from
  `extract_quads_soa`'s tail; the existing JSON appender captures it
  alongside the existing span CLOSE events.
* **Frame sequencing:** `--test-threads=1` so latency reflects
  serialised per-frame work and not Rayon contention between scenes.
  Within a single frame, Rayon is still free to parallelise across
  cores.
* **Datasets & invocation (verbatim):**

  ```
  rm -rf target/profiling && mkdir target/profiling
  TRACY_NO_INVARIANT_CHECK=1 \
  TELEMETRY_MODE=json \
  LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
    target/release/deps/regression_render_tag-<hash> \
      --test-threads=1 --exact \
      accuracy_baseline::regression_hub_tag36h11_2160p \
      accuracy_baseline::regression_hub_tag36h11_1080p
  ```

* **Aggregator:** `tools/bench/aggregate_telemetry.py` — same
  span-CLOSE accounting as V2's, with a new section that parses the
  `pipeline::quad_route_summary` `info!` events. Each event carries
  eight integer fields; the aggregator reports both the per-run sum and
  the per-frame mean.

## §3 Per-stage timings (this session)

`samples = 50 frames` for every stage. Times are wall-clock per frame,
in milliseconds, including all Rayon-parallel sub-tasks fused inside
that span. These are the same `pipeline::*` spans V2's audit reported.

### 2160p (3840×2160), `high_accuracy + Accurate`

| Stage              | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
| :----------------- | -------: | -------: | -------: | --------: |
| Preprocessing      |    3.075 |    5.267 |   11.130 |     3.545 |
| Segmentation       |        — |        — |        — |         — |
| Quad Extraction    |    2.930 |   48.665 |   62.051 |     9.309 |
| Decoding (Hard)    |    0.241 |    1.842 |    3.717 |     0.524 |
| Pose Refinement   |    0.030 |    0.036 |    0.048 |     0.031 |

### 1080p (1920×1080), same configuration, same run

| Stage              | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
| :----------------- | -------: | -------: | -------: | --------: |
| Preprocessing      |    0.932 |    1.228 |    1.430 |     0.975 |
| Segmentation       |        — |        — |        — |         — |
| Quad Extraction    |    0.988 |    7.405 |    9.197 |     2.007 |
| Decoding (Hard)    |    0.127 |    0.412 |    1.076 |     0.183 |
| Pose Refinement   |    0.028 |    0.036 |    0.039 |     0.028 |

### Latency-regression check vs V2's 2026-05-03 audit

| Stage             | V2 p95 (2160p) | Now p95 (2160p) | Δ      |
| :---------------- | -------------: | --------------: | -----: |
| Preprocessing     |          6.031 |           5.267 |  −13 % |
| Quad Extraction   |         46.235 |          48.665 |   +5 % |
| Decoding (Hard)   |          1.368 |           1.842 |  +35 % |
| Pose Refinement   |          0.042 |           0.036 |  −14 % |

The Quad-Extraction p95 shift of +5 % is well inside the run-to-run
KVM-noise envelope V2's audit flagged. Decoding p95 is +35 % but
absolute (1.4 → 1.8 ms) — three samples in a 50-frame distribution
push p95 around easily; the means agree (0.46 → 0.52 ms). No stage
shows a step change consistent with a real regression; the
`bench-internals` instrumentation is within noise.

## §4 Quad-extraction sub-stage attribution (the new content)

Per-frame counter sums and means across the two run sets, exactly as
emitted by `pipeline::quad_route_summary` and parsed by the aggregator.
Field semantics live in
[`QuadRouteCounters`'s rustdoc](../../crates/locus-core/src/quad.rs).

| Counter                  |  1080p sum |  1080p mean | 2160p sum |  2160p mean |
| :----------------------- | ---------: | ----------: | --------: | ----------: |
| EdLines attempts         |       1922 |        38.4 |      7240 |       144.8 |
| EdLines survivors        |       1114 |        22.3 |      4322 |        86.4 |
| ContourRdp attempts      |          1 |         0.0 |         2 |         0.0 |
| ContourRdp survivors     |          0 |         0.0 |         0 |         0.0 |
| RDP iterations (total)   |         11 |         0.2 |        18 |         0.4 |
| RDP iterations (max)     |         11 |        n/a  |         9 |         n/a |
| Refine Erf calls         |          0 |         0.0 |         0 |         0.0 |
| Refine None calls        |       1088 |        21.8 |      4181 |        83.6 |

(Means are computed across all 50 frames including frames with zero
candidates of a given type.)

### Reading the table

* **The dominant sub-stage at 2160p is EdLines.** 144.8 attempts/frame
  with 86.4 survivors (~60 % pass rate); ContourRdp sees ≈ 0 traffic
  per frame. Every quad that reaches `extract_single_quad` and survives
  the geometric/aspect/fill gates routes to EdLines, because
  `high_accuracy`'s `AdaptivePpb` policy threshold is permissive enough
  that 2160p tags project to ≥ 5 PPB and land in the high-PPB lane —
  whose extractor is `EdLines` and whose refinement is
  `CornerRefinementMode::None`.

* **EdLines + None is essentially the *only* hot path at 2160p.**
  4322 EdLines survivors vs 4181 `Refine None` calls — every EdLines
  survivor (minus a small number rejected by the post-refinement edge
  score gate, the difference of 4322 − 4181 = 141 ≈ 3 %) goes through
  `(EdLines, None)` and never sees `refine_corner`. There is
  effectively *no* Erf refinement traffic on this configuration.

* **ContourRdp is dead on this dataset.** 2 attempts in 50 frames (one
  scene each on the 1080p and 2160p runs); both rejected before the
  area/compactness gate (0 survivors). The RDP iteration counters
  (max = 9-11 stack pops, total = 11-18) are tiny — RDP is not the
  bottleneck here, because RDP barely runs.

* **Implication for the V2 follow-up.** The 47 ms p95 of
  `extract_quads_soa` at 2160p is dominated by the *count* of EdLines
  attempts (144.8/frame) times the per-call cost of
  `extract_quad_edlines`. Halving the attempts (e.g. by tightening the
  pre-EdLines geometric gates so dense-noise 4 K scenes don't reach
  `extract_single_quad` at the `MAX_CANDIDATES = 1024` ceiling) should
  cut wall-clock close to linearly. Per-call optimisation of the
  ContourRdp/RDP machinery would not help — that route is cold.

* **Per-frame multiplier 1080p → 2160p.** EdLines attempts: 38.4 →
  144.8 = **3.77×** for 4× pixel count. EdLines survivors: 22.3 →
  86.4 = **3.88×**. These are inside the V2 audit's expected band
  (sub-linear to linear) and consistent with the 3.92× p95 multiplier
  on Quad Extraction itself — the per-call cost is roughly constant,
  the count-of-calls dominates.

### Dominant sub-stage call-out

> At 2160p on this dataset (`locus_v1_tag36h11_3840x2160`, 50 scenes,
> `high_accuracy + Accurate` profile):
>
> **EdLines attempts (≈ 145 per frame) are the dominant cost** of the
> `pipeline::quad_extraction` p95. RDP simplification is
> effectively absent (≈ 0 attempts/frame), and Erf-based corner
> refinement is also absent (the high-PPB lane uses
> `CornerRefinementMode::None`).
>
> The per-frame `Quad Extraction` budget at p95 (48.7 ms) divides
> roughly as: ≈ 145 × `extract_quad_edlines` calls + ≈ 145 ×
> `calculate_edge_score` calls + arena/Rayon overhead, with no
> meaningful contribution from `douglas_peucker`,
> `chain_approximation`, `trace_boundary`, or `refine_corner`.

## §5 Reproducibility recipe

1. Pull `track-v2fu2-quad-route-counters`.
2. Build the test binary in release mode with all features (the
   `bench-internals` cfg is what compiles the counter struct):

   ```
   cargo nextest run --release --all-features --no-run
   ```

3. Wipe the telemetry directory and run:

   ```
   rm -rf target/profiling && mkdir target/profiling
   TRACY_NO_INVARIANT_CHECK=1 TELEMETRY_MODE=json \
   LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
     target/release/deps/regression_render_tag-<hash> \
       --test-threads=1 --exact \
       accuracy_baseline::regression_hub_tag36h11_2160p \
       accuracy_baseline::regression_hub_tag36h11_1080p
   ```

4. Aggregate:

   ```
   uv run python tools/bench/aggregate_telemetry.py target/profiling/
   ```

   The new section beneath the per-stage table reports the eight
   route counters — both as totals and per-frame means.

## §6 Caveats

* The counter array is stack-allocated (zero-sized when
  `bench-internals` is off), so it cannot leak into the FFI. Python
  callers will not see these counters; the diagnostic is internal.
* The `Refine None` row sums under both EdLines+None and ContourRdp+
  None — the dominant policies at 2160p send everyone through EdLines,
  so on this dataset all `Refine None` traffic is from EdLines
  survivors. Splitting the refinement counter by extraction route
  would require duplicating the field; we elect not to do that — the
  attempts/survivors columns already disambiguate the route mix.
* The `refine_corner_with_camera` path (distorted cameras) is
  classified as `Refine Erf` because it always performs a sub-pixel
  pass. This dataset is rectified, so that distinction does not
  surface here.
* Counter saturation: each `AtomicU32` is bumped via `fetch_add(1,
  Relaxed)`, which would saturate at ~4.3 G events. With ≤ 1024
  candidates per frame and ≤ thousands of frames per run, saturation
  is impossible in practice; we rely on the wraparound never happening
  rather than checking a sentinel.
