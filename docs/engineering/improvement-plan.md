# locus-core Improvement Plan

> Generated from static analysis of `crates/locus-core/src/` on 2026-03-14.
> Items are ordered by expected impact (correctness → performance → ergonomics).

---

## 1. Replace `.expect()` Panics in `detector.rs` with Proper Error Propagation

### Problem
`detect()` contains multiple `.expect()` calls on fallible image operations (decimation, upscaling, buffer sizing). A bad configuration or unexpected image geometry silently panics instead of returning a recoverable `Err`.

Key sites (detector.rs):
- `decimated_image(...)expect("decimation failed")`
- `upscale(...)expect("upscale failed")`
- Several buffer-length assertions via `expect`

### Plan
1. Introduce a `DetectorError` enum in `lib.rs` (or a dedicated `error.rs`) using `thiserror`:
   ```rust
   #[derive(Debug, thiserror::Error)]
   pub enum DetectorError {
       #[error("decimation failed: {0}")]
       Decimation(String),
       #[error("image dimensions invalid: {0}")]
       InvalidDimensions(String),
       // …
   }
   ```
2. Change `detect()` signature to `-> Result<DetectionBatchView<'_>, DetectorError>`.
3. Replace every `.expect()` with `?` (mapping via `.map_err(DetectorError::Decimation)`).
4. Propagate through the PyO3 boundary (`PyErr::new::<PyValueError, _>`).

### Testing
- Unit test: pass a 0×0 image — expect `Err(DetectorError::InvalidDimensions)`, not a panic.
- Unit test: configure `decimation_factor = 7` on a 6-pixel-wide image — expect clean error.
- Existing test suite must still pass (`cargo nextest run --release`).

---

## 2. Decompose the Monolithic `detect()` Function

### Problem
`detector.rs::detect()` is >500 lines (suppressed via `#[allow(clippy::too_many_lines)]`). Each pipeline stage is inlined sequentially, making isolated testing, profiling attribution, and future refactoring very hard.

### Plan
Extract the following private helpers, each accepting a shared `&mut DetectorState` slice:

| New function | Covers |
|---|---|
| `run_preprocessing()` | Decimation → bilateral filter → thresholding |
| `run_segmentation()` | CCL → component filtering |
| `run_quad_extraction()` | Contour tracing → Douglas-Peucker → sub-pixel refinement |
| `run_decoding()` | Homography computation → bit sampling → Hamming decode |
| `run_pose_refinement()` | Fast / Accurate mode pose estimation |
| `emit_telemetry()` | All `#[cfg(feature = "rerun")]` debug logging |

Each function takes `batch: &mut DetectionBatch`, `image: &ImageView`, relevant config slices, and the `arena: &Bump`, returning `Result<usize, DetectorError>` (candidate count).

### Testing
- After extraction, run full regression suite — output must be bit-identical.
- Each helper becomes independently unit-testable with synthetic inputs.
- Add a unit test per helper with minimal synthetic data (e.g., a 64×64 checkerboard for `run_thresholding()`).

---

## 3. Centralise All Magic Constants into `DetectorConfig`

### Problem
Numerically significant constants are scattered across modules with no exposure to users or documentation of their derivation:

| Constant | Location | Current value |
|---|---|---|
| Huber δ | `pose.rs` | `1.5` px |
| Tikhonov α_max | `pose_weighted.rs` | `0.25` px² |
| Pixel noise σ²_n | `pose_weighted.rs` | `4.0` |
| Soft LLR scale | `strategy.rs` | `60` |
| Coarse rejection ratio | `strategy.rs` | `2×` |
| Structure tensor radius | `pose_weighted.rs` | `2` px |
| Douglas-Peucker factor | `quad.rs` | `0.02 × perimeter` |

### Plan
1. Add fields to `DetectorConfig` (with `#[doc]` explaining the formula or paper reference):
   ```rust
   /// Huber δ for LM reprojection (pixels). Residuals beyond this threshold
   /// are down-weighted linearly. Default 1.5 px (≈ 1 pixel blur radius).
   pub huber_delta_px: f32,

   /// Maximum Tikhonov regularisation α (px²) for ill-conditioned corners.
   /// See Accurate pose mode, Structure Tensor gain scheduling.
   pub tikhonov_alpha_max: f32,
   // …
   ```
2. Thread values through function signatures instead of `const` literals.
3. Keep the current defaults identical to the existing hard-coded values.

### Testing
- Snapshot tests: re-run the ICRA-2020 regression with all defaults → must match prior snapshots.
- Property test: set `huber_delta_px = f32::INFINITY` → solver degenerates to standard LM (verify equal output to unclamped path).

---

## 4. Eliminate Per-Thread `Bump::new()` in `quad.rs`

### Problem
`quad.rs` calls `Bump::new()` inside the `rayon` parallel iterator body (one per candidate). Each allocation triggers the system allocator, defeating the hot-loop zero-allocation goal stated in `constraints.md`.

```rust
// quad.rs ~line 91 — current (bad)
candidates.par_iter_mut().for_each(|cand| {
    let arena = Bump::new();   // ← system alloc per candidate
    // …
});
```

### Plan
Option A (preferred): Use `thread_local!` to hold a reusable `RefCell<Bump>`, reset at the start of each task:
```rust
thread_local! {
    static QUAD_ARENA: RefCell<Bump> = RefCell::new(Bump::with_capacity(8 * 1024));
}

candidates.par_iter_mut().for_each(|cand| {
    QUAD_ARENA.with(|a| {
        let mut arena = a.borrow_mut();
        arena.reset();
        extract_quad_inner(cand, &arena);
    });
});
```

Option B: Pre-allocate a `Vec<Bump>` sized to rayon's thread count during `Detector::new()`, and distribute via index.

### Testing
- Add a `divan` micro-benchmark (`benches/quad_extraction.rs`) measuring quad extraction for N=512 candidates.
- Run with `--profile-time` to confirm zero `malloc` calls (use `MALLOC_CONF=stats_print:true` or `jemalloc` introspection).
- Regression: ICRA-2020 snapshot must be unchanged.

---

## 5. Move Segmentation Run-Storage to the Frame Arena

### Problem
`segmentation.rs` allocates a `Vec<Run>` per row inside the parallel pass. For 4K images this can be hundreds of small heap allocations per frame.

```rust
// segmentation.rs ~line 131 — current
let runs: Vec<Run> = extract_runs(row);  // ← heap alloc
```

### Plan
1. Pre-compute the upper bound of runs per row (`width / 2 + 1`).
2. Allocate a flat `bumpalo::Bump`-backed slab sized `height × max_runs_per_row` once during `DetectorState::new()`.
3. Hand out row-length slices during the parallel pass; `reset()` at frame start.

This matches the same pattern used for the `binarized_image` buffer in `threshold.rs`.

### Testing
- Benchmark: `cargo bench --bench comprehensive -- segmentation` — should show reduced allocation count.
- Correctness: run `cargo nextest run --release segmentation` — all unit tests pass.
- Snapshot: ICRA-2020 regression unchanged.

---

## 6. Add Range-Validated `DetectorBuilder` Guards

### Problem
`DetectorBuilder` accepts any value for 20+ parameters and defers all validation to deep pipeline code (or panics). Invalid configs only surface at runtime.

Examples of unchecked invariants:
- `threshold_tile_size` must be ≥ 4 and a power of 2.
- `quad_min_area` must be < image area at runtime.
- `decimation_factor` must be ≥ 1 and divide evenly into expected image sizes.
- `max_hamming_distance` must be ≤ the family's error-correction capacity.

### Plan
1. Add a `validate(&self) -> Result<(), ConfigError>` method to `DetectorConfig`.
2. Call it at the end of `DetectorBuilder::build()` returning `Result<Detector, ConfigError>`.
3. Each invalid combination emits a human-readable message: `"threshold_tile_size must be a power of 2, got 7"`.

```rust
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("threshold_tile_size {0} is not a power of 2")]
    TileSizeNotPow2(usize),
    // …
}
```

### Testing
- Unit tests: one per invariant — pass bad value, assert `Err(ConfigError::…)`.
- Doc-test in `DetectorBuilder::build()` showing the expected error.
- Fuzz test with `cargo fuzz` target hitting `DetectorBuilder` with arbitrary `u32` inputs.

---

## 7. Resurrect or Remove the Dead `refine_corners_with_homography` Function

### Problem
`decoder.rs` contains `#[allow(dead_code)] fn refine_corners_with_homography()`. Dead code accumulates technical debt, confuses contributors, and may contain subtle bit-rot.

### Decision tree
- **If still needed** (e.g., as a fallback when ERF sub-pixel refinement diverges): wire it into `QuadExtractor`, add a config flag `CornerRefinementMode::Homography`, and test with challenging imagery.
- **If obsolete**: delete it and add a comment to the `CHANGELOG` noting removal.

### Plan (if wired in)
1. Expose via `config.rs`: `CornerRefinementMode::Erf | Homography | Hybrid`.
2. Dispatch inside `run_quad_extraction()`.
3. Add a snapshot benchmark comparing corner RMSE under both modes on synthetic checkerboard images.

### Testing
- Remove path: confirm `cargo clippy` emits no dead-code warnings.
- Wire-in path: property test — homography-refined corners must be within 0.5 px of ERF-refined corners on noise-free synthetic images.

---

## 8. Add a Latency Regression Micro-Benchmark Suite

### Problem
There is no automated guard preventing performance regressions in individual pipeline stages. A change to `threshold.rs` or `segmentation.rs` can silently slow down the hot loop between releases.

### Plan
Create `crates/locus-core/benches/pipeline_stages.rs` with `divan` benchmarks for each stage:

```rust
#[divan::bench(args = [360, 720, 1080])]
fn bench_threshold(height: usize) -> u8 {
    let img = synthetic_image(height * 16 / 9, height);
    let mut engine = ThresholdEngine::new(&cfg);
    divan::black_box(engine.compute(&img))
}
```

Stages to cover:
- `bench_threshold` (adaptive thresholding)
- `bench_segmentation` (CCL)
- `bench_quad_extraction` (contour + DP + ERF)
- `bench_decoding_hard` / `bench_decoding_soft`
- `bench_pose_fast` / `bench_pose_accurate`

Integrate into CI:
- Compare against stored baselines with `critcmp` or `divan`'s built-in comparison.
- Fail CI if any stage regresses by >10 %.

### Testing
- The benchmarks themselves are the test — run with `cargo bench --bench pipeline_stages`.
- Gate via `cargo bench -- --baseline main` on PR branches.

---

## 9. Make Structure Tensor Window Size Configurable

### Problem
`pose_weighted.rs` hard-codes a 5×5 (radius=2) window for Structure Tensor computation. For small tags (<30 px per side), this window may cover multiple edges and produce misleading corner uncertainty estimates, degrading pose accuracy at range.

### Plan
1. Add `structure_tensor_radius: u8` to `DetectorConfig` (default 2, range 1–4).
2. Pass through to `compute_structure_tensor()` in `gradient.rs`.
3. Document the trade-off in the config docstring: larger radius → smoother estimate, but bleeds across adjacent edges for small tags.

### Testing
- Synthetic benchmark: render tags at 20 px, 40 px, 60 px per side; measure pose RMSE for radius ∈ {1, 2, 3}.
- Snapshot: ICRA-2020 regression with default radius=2 must be unchanged.
- Property test: tensor eigenvalues must remain positive-semidefinite for any radius on any valid grayscale image.

---

## 10. Instrument Soft Decode Path with a Dedicated Benchmark and Fix the Magic-60 Constant

### Problem
`strategy.rs` uses `soft_threshold = max_error * 60` where `60` is an undocumented scale factor converting Hamming units to LLR units (derived from typical LLR range [−127, 127] across a 36-bit tag). Without a comment or constant name this is a maintenance hazard, and the Soft strategy has no isolated benchmark — its latency contribution to the decode stage is invisible.

### Plan
1. **Document the constant:**
   ```rust
   /// Scale factor mapping a Hamming distance (integer bit-flips) to the
   /// equivalent total LLR penalty. Derived from the typical saturated LLR
   /// magnitude (~60 per bit for 8-bit image gradients).
   const LLR_PER_HAMMING_BIT: f32 = 60.0;
   ```
2. **Benchmark** (`benches/soft_decode.rs`):
   ```rust
   #[divan::bench(args = [0, 1, 2, 3])]  // hamming error levels
   fn bench_soft_decode(hamming: usize) { … }
   ```
   This gives visibility into MIH search depth as a function of corruption level.
3. **Expose `coarse_rejection_threshold` multiplier** via `DetectorConfig::soft_coarse_rejection_ratio: f32` (default 2.0) so users in high-noise environments can widen the initial candidate window without recompiling.

### Testing
- Unit test: inject a code with exactly `k` bit-flips — assert soft decoder finds it for k ≤ max_error.
- Property test: for random bitstrings at Hamming distance exactly max_error+1, decoder must return `None`.
- Benchmark confirms sub-linear search growth with tag family size (MIH correctness indicator).

---

## Summary Table

| # | Area | Impact | Effort | Risk |
|---|---|---|---|---|
| 1 | Error propagation (`detect()`) | Safety / Correctness | Medium | Low |
| 2 | Decompose `detect()` | Maintainability | Medium | Low |
| 3 | Centralise magic constants | Config / Ergonomics | Low | Minimal |
| 4 | Thread-local arenas in `quad.rs` | Performance | Low-Med | Low |
| 5 | Arena-backed segmentation runs | Performance | Medium | Low |
| 6 | Builder validation | Safety / DX | Low | Minimal |
| 7 | Dead code resolution | Code Health | Low | Minimal |
| 8 | Pipeline stage micro-benchmarks | Observability | Medium | Minimal |
| 9 | Configurable tensor window | Accuracy | Low | Low |
| 10 | Soft decode constant + benchmark | Observability / Config | Low | Minimal |

## Recommended Execution Order

```
Phase 1 (correctness, 1–2 days):  #1 → #6 → #7
Phase 2 (performance, 2–3 days):  #4 → #5 → #8
Phase 3 (ergonomics, 1–2 days):   #3 → #10 → #2 → #9
```

Each phase ends with:
```bash
cargo nextest run --release --all-features
LOCUS_DATASET_DIR=tests/data/icra2020 \
LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
cargo insta test --release --all-features --features bench-internals
```
