# Micro-Benchmarking Guide

This document describes the 3-Tier validation loop for micro-optimizations in Locus.

## Overview

A micro-optimization in Locus is only considered successful if it survives the transition through all three tiers of the tooling stack. This process ensures that mathematical improvements translate into real-world pipeline speedups.

---

### Tier 3: Mathematical & Memory Isolation (Divan)

**Goal:** Prove the mathematical or memory optimization works in isolation (e.g., reducing SIMD bilinear sampling cost).

- **Framework:** [Divan](https://github.com/nvzqz/divan)
- **Location:** `crates/locus-core/benches/`
- **Constraint:** Must run **strictly single-threaded** to ensure pristine Instructions Per Clock (IPC) and L1 cache utilization metrics.
- **Data:** Use the `BenchDataset` utility to load real-world image gradients from the ICRA 2020 dataset or generate realistic Structure of Arrays (SoA) states.

**Execution:**
```bash
# Run a specific benchmark suite
cargo bench --bench comprehensive --features "extended-bench bench-internals" -- "bench_thresholding" --threads 1

# Run full resolution sweep and update baselines
cargo bench --bench comprehensive --features "extended-bench bench-internals" -- "bench_thresholding" "bench_segmentation" "bench_quad_extraction" --threads 1 > target/profiling/divan_output.txt
PYTHONPATH=. python3 tools/bench/update_micro_baselines.py target/profiling/divan_output.txt
```

---

### Tier 2: Pipeline Visual Confirmation (Tracy)

**Goal:** Visually confirm the specific span in the pipeline has shrunk and didn't push the bottleneck elsewhere (e.g., to a following memory allocation).

- **Tool:** [Tracy Profiler](https://github.com/wolfpld/tracy)
- **Build:** Recompile with the `tracy` feature enabled.

**Execution:**
```bash
# Build and run with tracy instrumentation
uv run maturin develop -r -F tracy
# Run a trace using the Python CLI or a test script
uv run tools/cli.py bench real --num-frames 100
```

---

### Tier 1: Real-World Ground Truth (Python CLI)

**Goal:** The ultimate validation. If the total pipeline latency doesn't drop here, the optimization didn't work.

- **Tool:** Locus Python CLI
- **Benchmark:** `bench real` command

**Execution:**
```bash
# Run the end-to-end benchmark
uv run tools/cli.py bench real
```

> **Note:** "If a micro-optimization doesn't lower this number, it didn't actually work."

---

## Technical Details

### Single-Threaded Mandate
To prevent the OS scheduler or Rayon from thrashing the L1 cache, micro-benchmarks must enforce a single-threaded environment. This is handled globally in the benchmark's `main()` function:

```rust
fn main() {
    let _ = rayon::ThreadPoolBuilder::new().num_threads(1).build_global();
    divan::Divan::from_args().threads([1]).run_benches();
}
```

### Realistic Data Layouts
- **Early Stages (Thresholding/Segmentation):** Load grayscale frames into `ImageView` using `BenchDataset::icra_forward_0()`.
- **Late Stages (Decoding/Pose Estimation):** Populate a `DetectionBatch` (SoA) with a realistic distribution of candidates (e.g., 50 valid tags and 200 false-positives) using `BenchDataset::generate_bench_batch()`.
