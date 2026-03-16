# Benchmarking & Diagnostics

Locus is built with a focus on extreme performance. To maintain this, we provide a suite of tools for benchmarking and diagnosing failures, covering both the core Rust engine and the Python bindings.

## The 3-Tier Tooling Stack

To tune the Locus codebase for maximum throughput, we enforce strict boundaries between measurement tools to avoid the "Observer Effect".

### Tier 1: End-to-End Regression (The Python CLI)
*   **Tool**: `uv run tools/cli.py bench real` (using the ICRA 2020 dataset).
*   **Purpose**: Measures the true wall-clock time from Python memory ingestion, through the FFI boundary, across the Rust math kernels, and back to Python.
*   **Rule**: This is the ultimate ground truth for latency. If a micro-optimization doesn't lower this number, it didn't actually work.

### Tier 2: Macro-Profiling (Tracy)
*   **Tool**: `tracing-tracy` combined with the Tracy GUI profiler (`cargo test --features tracy`).
*   **Purpose**: Identifies which pipeline stage is the bottleneck (e.g., proving that `decode_batch_soa` is taking 10ms while `extract_quads` is taking 2ms).
*   **Rule**: Never run Tracy concurrently with JSON loggers or console formatters. The string allocation overhead will pollute the nanosecond lock-free ring buffers.

### Tier 3: Micro-Benchmarking (Divan)
*   **Tool**: `cargo bench` using the **Divan** framework in `crates/locus-core/benches/`.
*   **Purpose**: Measures the Instructions Per Clock (IPC) and L1 cache utilization of isolated, single-threaded mathematical kernels (like SIMD bilinear sampling).
*   **Rule**: Run these strictly single-threaded to prevent the OS scheduler or Rayon from thrashing the L1 cache.

---

## Rust Benchmarking (Core Engine)

The Rust benchmarking suite is the source of truth for core engine performance and regressions.

### Regression Suite (ICRA 2020)
The regression suite validates that `Locus` matches or exceeds ground truth for thousands of images.

1. **Set Dataset Path**:
   ```bash
   export LOCUS_DATASET_DIR=/path/to/icra2020
   ```
2. **Run Benchmarks**:
   ```bash
   # Core check (Forward dataset + Fixtures, approx 15s)
   cargo test --release --test regression_icra2020 --features bench-internals

   # Extended check (Circle, Random, Rotation, approx 1-2 mins)
   LOCUS_EXTENDED_REGRESSION=1 cargo test --release --test regression_icra2020 --features bench-internals

   # Accurate latency measurement (sequential)
   cargo test --release --test regression_icra2020 --features bench-internals -- --test-threads=1
   ```
   > [!IMPORTANT]
   > `--release` is mandatory for running `regression_icra2020` tests. Running in debug mode is blocked and will panic.

### Hub Regression Suite (Hugging Face)
Locus supports running regressions against large-scale datasets hosted on the Hugging Face Hub.

> [!IMPORTANT]
> `--release` is mandatory for running Hub regression tests. Running in debug mode is extremely slow and will likely timeout in CI or developer environments.

1. **Synchronize Data**:
   Download the datasets to a local cache. This requires the `bench` dependency group.
   ```bash
   uv run python tools/bench/sync_hub.py --configs single_tag_locus_v1_std41h12_1920x1080
   ```

2. **Run Hub Tests**:
   Point the test runner to the local cache directory:
   ```bash
   LOCUS_HUB_DATASET_DIR=tests/data/hub_cache cargo test --release --test regression_render_tag regression_hub_ --features bench-internals -- --nocapture
   ```

### Logic-Specific Benchs (Micro-benchmarking)
For fine-grained benchmarking of specific components, we use **Divan**. These are located in `crates/locus-core/benches`.

```bash
# Run all micro-benchmarks
cargo bench

# Run specific micro-benchmark (e.g., real-world data)
cargo bench --bench real_data_bench
cargo bench --bench decoding_real_bench
```

### Mutually Exclusive Telemetry Matrix
Locus implements a zero-cost, mutually exclusive telemetry architecture for its regression tests to avoid the "Observer Effect". You cannot simultaneously emit structured JSON logs and capture high-fidelity Tracy profiles without the JSON serialization skewing the nanosecond timings. 

To resolve this, we decouple the profilers at the CI level using `TELEMETRY_MODE`.

#### Human Mode (Tracy)
Captures pristine binary traces for GUI analysis.
```bash
# Tracy client is assumed to be running or capturing headlessly
TRACY_NO_INVARIANT_CHECK=1 TELEMETRY_MODE=tracy cargo test --release --test regression_icra2020 --features tracy,bench-internals -- --test-threads=1
```

#### Agent/CI Mode (JSON)
Dumps structured pipeline timings to `target/profiling/*_events.json` for AI analysis and automated regression tracking.
```bash
TELEMETRY_MODE=json cargo test --release --test regression_icra2020 --features bench-internals -- --test-threads=1
```

---

## Python Developer CLI

The `tools/cli.py` tool is the central entry point for high-level evaluations and development tasks.

### Data Preparation
Before running benchmarks, download all required datasets (AprilTag Mosaic and ICRA 2020):
```bash
PYTHONPATH=. uv run --group bench tools/cli.py bench prepare
```

### Real-World Evaluation
Evaluate performance on the ICRA 2020 dataset scenarios (`forward`, `circle`):
```bash
# Basic run on Locus
PYTHONPATH=. uv run --group bench tools/cli.py bench real --scenarios forward

# Compare against OpenCV and AprilTag 3
PYTHONPATH=. uv run --group bench tools/cli.py bench real --scenarios forward --compare
```

### Regression Tracking (Baselines)
You can save a "Golden Baseline" and compare current performance against it.

Historical Performance Profiles:
- [DDA-SIMD Decoding Profile (2026-03-16)](./benchmarking/funnel_dda_20260316.md)
- [SoA Migration Profile (2026-03-03)](./benchmarking/soa_migration_20260303.md)
- [Initial Baseline (2026-03-02)](./benchmarking/baseline_20260302.md)

```bash
# Save a baseline
PYTHONPATH=. uv run --group bench tools/cli.py bench real --scenarios forward --save-baseline docs/benchmarking/baseline.json

# Compare current run against baseline
PYTHONPATH=. uv run --group bench tools/cli.py bench real --scenarios forward --baseline docs/benchmarking/baseline.json
```

### Deep Profiling (Tracy)
Locus supports high-fidelity profiling using the [Tracy Profiler](https://github.com/wolfpld/tracy).

1. **Rebuild with Tracy support**:
   ```bash
   uv run maturin develop -r -F tracy
   ```
2. **Start the Tracy GUI client**.
3. **Run benchmark with profiling flag**:
   ```bash
   # Add --profile to any 'bench real' command
   PYTHONPATH=. uv run --group bench tools/cli.py bench real --profile --limit 5
   ```
   *Note: On some Linux systems, you may need `TRACY_NO_INVARIANT_CHECK=1` if your CPU doesn't support invariant TSC.*

---

## Visual Debugging with Rerun

For diagnosing recall issues or tuning parameters, use the specialized visualization tool:

```bash
uv run tools/cli.py visualize --scenario forward --limit 5
```

Locus provides a high-fidelity debugging pipeline integrated with the [Rerun SDK](https://rerun.io). 

### Features
- **Convergence Tracking**: Visualize subpixel jitter (yellow arrows) and reprojection errors (scalar plots) for every tag.
- **Failure Diagnosis**: Differentiate between geometric rejections (Red) and decoding failures (Orange).
- **Remote & Edge Ready**: Debug edge devices remotely using `--rerun-addr` to stream to a local Rerun viewer.

For a comprehensive walkthrough, see the **[How-to Guide: Debugging with Rerun](./how-to/debug_with_rerun.md)**.
