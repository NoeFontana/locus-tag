# Render-tag latency + ERF corner-exclusion investigation (2026-07-05)

First-principles pass at improving render-tag latency and corner/pose accuracy.
One change shipped (Track B, latency, byte-identical); one hypothesis falsified
(Track A, accuracy); the per-stage latency diagnostic (Track C) is recorded to
scope future work.

## §1 Verified hardware / build metadata

```
$ lscpu
Architecture:            x86_64
Vendor ID:               AuthenticAMD
Model name:              AMD EPYC-Milan Processor
CPU(s):                  8  (4 cores × 2 threads)
Flags include:           avx2, fma, sha_ni   (no avx512)
$ uname -r                6.8.0-134-generic
$ rustc --version         rustc 1.92.0 (ded5c06cf 2025-12-08)
```

Build: `--release --features bench-internals`. All latency numbers are `divan`
micro-benchmarks pinned to a **single rayon thread**
(`ThreadPoolBuilder::num_threads(1)`, `hub_bench.rs`), 100 samples each.
`LOCUS_HUB_DATASET_DIR` set to the local `tests/data/hub_cache`.

**Noise caveat.** The host is a shared 8-vCPU KVM guest with competing
background load. Absolute `full_pipeline` medians drifted ~30 % between runs
(1080p 35.5 ms → 46.3 ms) purely from system load, so **cross-run** pipeline
comparisons are unreliable. Only **within-run** stage ratios are trusted below.

## §2 Track B — drop the dead binary-threshold pass (SHIPPED)

`ThresholdEngine::apply_threshold_with_map` wrote two per-pixel outputs: the
threshold map (consumed by segmentation) and a full-image binary via
`threshold_row_simd`. Segmentation (`simd_ccl_fusion::extract_rle_segments`)
re-thresholds from the map and never reads the binary; the `binarized` buffer is
consumed only by Python debug telemetry (`binarized_ptr`). The binary pass — and
the `tile_valid` computation that feeds only it — is therefore **dead on the
non-telemetry `detect()` hot path**.

New `apply_threshold_map_only` writes just the map (shared `compute_tile_thresholds`
prologue, no `tile_valid`, no binary). `detector.rs` dispatches to it when
`debug_telemetry` is off and only then allocates `binarized`.

Within-run threshold-stage medians (`with_map` → `map_only`):

| Res   | with_map | map_only | Δ      |
|-------|----------|----------|--------|
| 480p  | 412.9 µs | 277.5 µs | −33 %  |
| 720p  | 1.110 ms | 739.6 µs | −33 %  |
| 1080p | 2.390 ms | 1.582 ms | −34 %  |
| 2160p | 9.537 ms | 6.190 ms | −35 %  |

Detection output is **byte-identical** (the binary was dead): `regression_render_tag`
+ `regression_render_tag_robustness` snapshots and the `contract_*_zero_alloc`
tests are unchanged. Threshold is ~5 % of the pipeline, so end-to-end this is
~2 %; it is free and carries zero accuracy risk.

**Aside (pre-existing, not caused by this change).** `regression_board_hub`
fails 7/9 with ~1e-13 drift in the board-pose snapshots (e.g. p99 rotation
`1.295316293736689` vs `1.2953162937344753`). This reproduces **identically on
clean `HEAD`** with the Track-B change stashed, so it is pre-existing
non-determinism in the joint board-pose reduction (non-associative floating-point
under rayon), independent of the threshold work. Flagged here for whoever
verifies the suite; it should be pinned or given a tolerance separately.

## §3 Track C — per-stage latency diagnostic

Per-stage divan medians (single run, so within-run ratios are apples-to-apples):

| Stage           | 480p    | 720p    | 1080p    | 2160p     |
|-----------------|---------|---------|----------|-----------|
| thresholding    | 0.41 ms | 1.11 ms | 2.39 ms  | 9.54 ms   |
| segmentation    | 2.27 ms | 6.76 ms | 16.7 ms  | **126.8 ms** |
| quad_extraction | 2.88 ms | 7.95 ms | 21.2 ms  | 102.3 ms  |

**Conclusion: threshold is NOT the bottleneck** (~5 % at every resolution). The
resolution-scaling mass is **quad-extraction (~46 % @1080p)** and **segmentation
(~36 % @1080p)**. At 2160p segmentation grows **super-linearly** — 16.7 ms →
126.8 ms is 7.6× for a 4× pixel increase (per-MP cost ~2× the 1080p rate),
consistent with the CCL working set spilling cache and/or a union-find /
moment-accumulation cost that scales worse than linear at 4K.

The remaining first-principles latency win therefore lives in segmentation (CCL)
and quad extraction, not threshold. Both are already SIMD + rayon-parallel and
are correctness-sensitive across every corpus (see
`benchmarking/lessons.md` on SoA-ordering fragility), so they warrant a dedicated,
separately-verified investigation rather than a bundled tweak. The 2160p
segmentation super-linearity is the highest-value lead.

## §4 Track A — ERF corner-adjacent sample exclusion (FALSIFIED)

The `standard`-profile corner refiner (`refine_corner` → `refine_edge_erf` →
`SampleConfig::for_quad`) fits a 1-DOF ERF offset per edge with
`t_range = (-0.1, 1.1)` — sampling *past* each endpoint into the shared L-corner
where the perpendicular scan crosses the adjacent edge's transition. Hypothesis:
excluding a per-end pixel margin removes that contamination and lowers RMSE.

Swept `erf_corner_exclude_margin_px ∈ {0,1,2,3}` px on the four standard-profile
robustness sets (1080p). Mean corner RMSE rose **monotonically**, recall flat:

| Dataset      | m=0    | m=1    | m=2    | m=3    |
|--------------|--------|--------|--------|--------|
| tag16h5      | 1.0799 | 1.0876 | 1.0890 | 1.0883 |
| high_iso     | 1.3361 | 1.3563 | 1.3585 | 1.3650 |
| raw_pipeline | 0.7594 | 0.7668 | 0.7693 | 0.7696 |
| low_key*     | 0.2578 | 0.2746 | 0.2591 | 0.2593 |

(*low_key recall 10 % — tiny matched sample.) p99 rotation is dominated by
decode/pose outliers (31–119°) and does not move. **Not strict-Pareto; reverted
in full** (no dead config knob shipped).

**Why it fails.** On clean synthetic Blender edges the corner contamination is
minimal, so the extra near-endpoint samples reduce variance more than they add
bias — trimming them raises RMSE. Same mechanism as the falsified EdLines Phase-5
chord-decoupling: on clean synthetic data, more data / more regularization wins.

**Scoping trap (important).** The primary render-tag `accuracy_baseline`
(`regression_render_tag.rs`) uses **high_accuracy**, whose large markers route to
**EdLines + None** (AdaptivePpb @ 2.5). The ERF `refine_corner` path is exercised
only by the `standard` profile (robustness suite) and the `max_recall_adaptive`
low route — it can never move the committed clean-render baseline. Re-attempt
only with real-camera evidence where contamination bias actually dominates
variance.
