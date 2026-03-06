# Benchmarking & Diagnostics

Locus is built with a focus on extreme performance. To maintain this, we provide a suite of tools for benchmarking and diagnosing failures, covering both the core Rust engine and the Python bindings.

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
   cargo test --release --test regression_icra2020

   # Extended check (Circle, Random, Rotation, approx 1-2 mins)
   LOCUS_EXTENDED_REGRESSION=1 cargo test --release --test regression_icra2020

   # Accurate latency measurement (sequential)
   cargo test --release --test regression_icra2020 -- --test-threads=1
   ```
   > [!IMPORTANT]
   > `--release` is mandatory for running `regression_icra2020` tests. Running in debug mode is blocked and will panic.

### Hub Regression Suite (Hugging Face)
Locus supports running regressions against large-scale datasets hosted on the Hugging Face Hub (e.g., `NoeFontana/locus-tag-bench`).

1. **Synchronize Data**:
   Download the datasets to a local cache. This requires the `bench` and `etl` dependency groups.
   ```bash
   uv sync --group bench --group etl
   PYTHONPATH=. uv run --group bench --group etl python scripts/bench/sync_hub.py --configs single_tag_locus_v1_std41h12
   ```

2. **Run Hub Tests**:
   Point the test runner to the local cache directory:
   ```bash
   export LOCUS_HUB_DATASET_DIR=tests/data/hub_cache
   cargo test --release --test regression_icra2020 -- regression_hub_datasets --nocapture
   ```

### Logic-Specific Benchs (Micro-benchmarking)
For fine-grained benchmarking of specific components, we use **Divan**. These are located in `crates/locus-core/benches`.

```bash
# Run all micro-benchmarks
cargo bench

# Run specific micro-benchmark (e.g., real-world data)
cargo bench --bench real_data_bench
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

#### CI Implementation (GitHub Actions)
In GitHub Actions, utilize a build matrix to run these jobs in parallel, entirely isolated environments:
```yaml
jobs:
  telemetry:
    strategy:
      matrix:
        mode: [tracy, json]
    steps:
      - run: |
          if [ "${{ matrix.mode }}" == "tracy" ]; then
            tracy-capture -o out.tracy &
            TRACY_NO_INVARIANT_CHECK=1 TELEMETRY_MODE=tracy cargo test --release --test regression_icra2020 --features tracy,bench-internals -- --test-threads=1
            # Upload out.tracy as artifact
          else
            TELEMETRY_MODE=json cargo test --release --test regression_icra2020 --features bench-internals -- --test-threads=1
            # Upload target/profiling/*.json as artifact
          fi
```

---

## Python Benchmarking CLI

The `scripts/locus_bench.py` tool is the central entry point for high-level evaluations.

### Data Preparation
Before running benchmarks, download all required datasets (AprilTag Mosaic and ICRA 2020):
```bash
PYTHONPATH=. uv run --group bench python scripts/locus_bench.py prepare
```

### Real-World Evaluation
Evaluate performance on the ICRA 2020 dataset scenarios (`forward`, `circle`):
```bash
# Basic run on Locus
PYTHONPATH=. uv run --group bench python scripts/locus_bench.py run real --scenarios forward

# Compare against OpenCV and AprilTag 3
PYTHONPATH=. uv run --group bench python scripts/locus_bench.py run real --scenarios forward --compare
```

### Regression Tracking (Baselines)
You can save a "Golden Baseline" and compare current performance against it.

Historical Performance Profiles:
- [SoA Migration Profile (2026-03-03)](./benchmarking/soa_migration_20260303.md)
- [Initial Baseline (2026-03-02)](./benchmarking/baseline_20260302.md)

```bash
# Save a baseline
PYTHONPATH=. uv run --group bench python scripts/locus_bench.py run --save-baseline docs/benchmarking/baseline.json real --scenarios forward

# Compare current run against baseline
PYTHONPATH=. uv run --group bench python scripts/locus_bench.py run --baseline docs/benchmarking/baseline.json real --scenarios forward
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
   # Add --profile to any 'run' command
   PYTHONPATH=. uv run --group bench python scripts/locus_bench.py run --profile real --limit 5
   ```
   *Note: On some Linux systems, you may need `TRACY_NO_INVARIANT_CHECK=1` if your CPU doesn't support invariant TSC.*

---

## Visual Debugging with Rerun

For diagnosing recall issues or tuning parameters, use the specialized visualization tool:

```bash
uv run python scripts/debug/visualize.py --scenario forward --limit 5
```

### Features
- **Pipeline Stages**: View the output of the binarizer and segmentation engine side-by-side with raw imagery.
- **Candidate Inspection**: See every quad candidate found before Hamming rejection.
- **Failure Diagnosis**: The tool automatically identifies tags in the ground truth that were missed and logs:
    - The reason for rejection (e.g., Hamming distance too high).
    - The 6x6 extracted bit grid for visual inspection of the bit-sampling quality.
