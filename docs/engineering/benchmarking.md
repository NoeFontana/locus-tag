# Benchmarking & Diagnostics

Locus is built with a focus on extreme performance. To maintain this, we provide a suite of tools for benchmarking and diagnosing failures, covering both the core Rust engine and the Python bindings.

## The 3-Tier Tooling Stack

To tune the Locus codebase for maximum throughput, we enforce strict boundaries between measurement tools to avoid the "Observer Effect".

### Tier 1: End-to-End Regression (The Python CLI)
*   **Tool**: `uv run tools/cli.py bench real` (ICRA 2020 scenarios or Hugging Face Hub datasets via `--hub-config`).
*   **Purpose**: Measures the true wall-clock time from Python memory ingestion, through the FFI boundary, across the Rust math kernels, and back to Python. Also reports recall and pose error (translation RMSE in metres) when ground truth poses are available.
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
   export LOCUS_ICRA_DATASET_DIR=/path/to/icra2020
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
   Download all Hub subsets to the local cache (`tests/data/hub_cache/`). The script auto-discovers every available config by default:
   ```bash
   uv run python tools/bench/sync_hub.py --configs all
   ```
   Or sync a specific subset:
   ```bash
   uv run python tools/bench/sync_hub.py --configs \
     locus_v1_tag36h11_640x480 \
     locus_v1_tag36h11_1280x720 \
     locus_v1_tag36h11_1920x1080 \
     locus_v1_tag36h11_3840x2160 \
     charuco_golden_v1_1920x1080 \
     aprilgrid_golden_v1_1920x1080
   ```

2. **Run Hub Tests**:
   ```bash
   # Tag-level regression (regression_render_tag)
   # Covers 4 resolutions × Erf/GWLF/EdLines variants and Fast/Accurate pose modes.
   # Requires LOCUS_HUB_DATASET_DIR to locate the cache.
   LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
     cargo test --release --test regression_render_tag --features bench-internals -- --nocapture

   # Board-level regression (regression_board_hub)
   # Validates ChAruco and AprilGrid golden datasets.
   # Uses workspace-relative tests/data/hub_cache/ automatically — no env var needed.
   cargo test --release --test regression_board_hub --features bench-internals -- --nocapture
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

## Performance Reports

### Current
- [Release Performance Report (2026-04-18)](benchmarking/release_performance_20260418.md)
- [SOTA Presets + GN Covariance (2026-03-21)](benchmarking/sota_metrology_20260321.md)
- [EDLines + Joint Gauss-Newton (2026-03-21)](benchmarking/edlines_gauss_newton_20260321.md)
- [SIMD CCL Fusion (2026-03-19)](benchmarking/simd_ccl_fusion_20260319.md)
- [Current Micro-Benchmark Baseline (2026-03-19)](benchmarking/baseline_micro_20260319.md)
- [Micro-Benchmarking Guide](benchmarking/micro-benchmarking-guide.md)

### Historical
- [Release Performance Report (2026-03-22)](benchmarking/release_performance_20260322.md)
- [Performance Evolution (Mar 2–16)](benchmarking/historical_evolution.md) — consolidated timeline of superseded reports

---

## Python Developer CLI

The `tools/cli.py` tool is the central entry point for high-level evaluations and development tasks.

### Data Preparation
Download all required datasets (ICRA 2020 and Hugging Face Hub subsets):
```bash
PYTHONPATH=. uv run --group bench tools/cli.py bench prepare
```
This command downloads the ICRA 2020 scenarios and auto-discovers and syncs all Hub dataset subsets from the configured HF repository to `tests/data/hub_cache/`.

### Real-World Evaluation (ICRA 2020)
Evaluate performance on the ICRA 2020 dataset scenarios (`forward`, `circle`):
```bash
# Basic run on Locus
PYTHONPATH=. uv run --group bench tools/cli.py bench real --scenarios forward

# Compare against OpenCV and AprilTag 3
PYTHONPATH=. uv run --group bench tools/cli.py bench real --scenarios forward --compare
```

### Hub Dataset Evaluation
Evaluate against rendered Hugging Face Hub datasets. These datasets include ground-truth 6-DOF poses, so the CLI reports both recall and pose error (translation RMSE in metres).

> [!NOTE]
> **Pose convention:** Hub ground truth poses use a center origin (the pose describes the tag center). Locus reports poses at the top-left corner origin. The CLI automatically applies the rigid center-to-top-left shift via `Metrics.align_pose` before computing the error.

```bash
# Single-tag evaluation
PYTHONPATH=. uv run --group bench tools/cli.py bench real \
  --hub-config locus_v1_tag36h11_1920x1080

# Board-level evaluation (AprilGrid or ChAruco)
# The board topology is inferred automatically from the dataset's rich_truth.json.
PYTHONPATH=. uv run --group bench tools/cli.py bench real \
  --hub-config aprilgrid_golden_v1_1920x1080

PYTHONPATH=. uv run --group bench tools/cli.py bench real \
  --hub-config charuco_golden_v1_1920x1080

# Limit frames and use a custom cache directory
PYTHONPATH=. uv run --group bench tools/cli.py bench real \
  --hub-config aprilgrid_golden_v1_1920x1080 \
  --data-dir tests/data/hub_cache \
  --limit 50
```

Hub evaluation is mutually exclusive with ICRA scenarios — passing `--hub-config` skips the `--scenarios` loop.

### Regression Tracking (Baselines)
You can save a "Golden Baseline" and compare current performance against it.

```bash
# Save a baseline
PYTHONPATH=. uv run --group bench tools/cli.py bench real --scenarios forward --save-baseline docs/engineering/benchmarking/baseline.json

# Compare current run against baseline
PYTHONPATH=. uv run --group bench tools/cli.py bench real --scenarios forward --baseline docs/engineering/benchmarking/baseline.json
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

For a comprehensive walkthrough, see the **[How-to Guide: Debugging with Rerun](../how-to/debug_with_rerun.md)**.
