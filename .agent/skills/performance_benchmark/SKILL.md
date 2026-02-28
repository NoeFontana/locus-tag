---
name: Performance Benchmark
description: Instructions for running and analyzing performance benchmarks for Locus.
---

# Performance Benchmark Skill

This skill guides you through the process of benchmarking the Locus library to ensure latency and throughput goals are met.

## 1. End-to-End Evaluation (Python)
Use this for a quick check of recall and end-to-end latency on the standard dataset.

First, compile the core library with release optimizations:
```bash
uv run maturin develop --release
```

Then, run the benchmark suite:
```bash
uv run python -m scripts.bench.run real --compare
```

**Success Criteria:**
- Latency (Median): < 1.1ms (640x480)
- Recall: > 98%

## 2. Regression Suite (Rust)
Use this for strict regression testing before merging PRs. It isolates the detector and runs on a controlled set of images.

```bash
cargo test --release --test regression_icra2020 -- --test-threads=1
```

**Note:** Always run with `--test-threads=1` to avoid CPU contention affecting timing results.

## 3. Micro-benchmarks (Rust)
Use `criterion` based benchmarks for specific functions (e.g., thresholding, quad-fitting).

```bash
cargo bench --workspace
```

## 4. Profiling with Tracy
To investigate performance implementations:
1. Enable the `tracy` feature in `Cargo.toml`.
2. Run your application or benchmark.
3. Open the Tracy profiler GUI to visualize spans.
