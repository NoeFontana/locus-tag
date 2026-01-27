# Agent Guide: Locus Perception Engineer

This guide provides actionable instructions and workflows for AI agents working on the Locus (`locus-vision`) codebase.

## 🛠 Core Workflows

### 1. Build & Test Loop
Always ensure the Rust core is built and available to Python after changes.
- **Full Sync & Build:**
  ```bash
  uv sync
  maturin develop --release --manifest-path crates/locus-py/Cargo.toml
  ```
- **Rust Fast Test:** `cargo nextest run --all-features`
- **Python Fast Test:** `pytest`
- **Linting:** `cargo clippy --all-targets --all-features -- -D warnings` and `uv run ruff check .`

### 2. Performance Evaluation
Use the Forward dataset to evaluate recall and latency impact of changes.
- **Run Forward Evaluation:**
  ```bash
  uv run python tests/evaluate_forward_performance.py
  ```
- **Regression Benchmark (ICRA 2020):**
  Use the Rust-based regression suite for latency/accuracy checks. For accurate timing, always run sequentially.
  ```bash
  cargo test --release --test regression_icra2020 -- --test-threads=1
  ```
- **Profile with Tracy (Rust):**
  Add `tracing::span!(...)` in hot paths and run with the `tracy` feature.

### 3. Visual Debugging with Rerun
Locus uses [Rerun](https://rerun.io/) for high-performance visualization.
- **Usage:** Set `LOCUS_VIZ=1` (if implemented) or use the `locus.viz` Python module.
- **Agent Action:** If a detector fails on a specific image, write a script using `rerun` to visualize the thresholding, segmentation, and quad candidates.

## 🚀 Slash Commands

- `/eval`: Trigger the full performance evaluation suite on `tests/data`.
- `/bench`: Run micro-benchmarks using `cargo bench` or `divan`.
- `/simd-check`: Analyze the generated assembly for SIMD kernels using `cargo-show-asm` or `cargo-expand`.

## 📐 Implementation Principles

### Zero-Allocation Hot loop
The detection loop (Threshold -> Segmentation -> Quad -> Decode) must not perform any heap allocations.
- **Arena:** Use `bumpalo::Bump` for all ephemeral frame data.
- **Stack:** Use `nalgebra::SMatrix` and `SVector` for geometric transforms.
- **No `std::collections`:** Avoid `HashMap` or `Vec` inside per-pixel or per-segment loops. Use pre-allocated slices or `SmallVec`.

### SIMD & Runtime Dispatch
- Use `multiversion` macro for all pixel-processing kernels.
- Prefer `std::simd` (portable-simd) when possible, falling back to architecture-specific intrinsics only if critical.

### Memory Safety
- **Unsafe blocks:** Every `unsafe` block MUST have a `// SAFETY:` comment explaining the invariant (e.g., "strides are validated to be contiguous at the FFI boundary").
- **Bounds Checks:** Use `.get_unchecked()` ONLY in SIMD loops where bounds have been pre-verified.

## 📁 Project Map
- `crates/locus-core`: The high-performance Rust engine.
- `crates/locus-py`: PyO3 bindings with zero-copy NumPy integration.
- `benchmarks/`: Criterion-based Rust benchmarks.
- `tests/`: Integration tests and dataset evaluation scripts.
