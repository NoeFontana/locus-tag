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
   > `--release` is mandatory for performance benchmarking.

### Hub Regression Suite (Hugging Face)
Locus supports running regressions against large-scale datasets hosted on the Hugging Face Hub (e.g., `NoeFontana/locus-tag-bench`).

1. **Synchronize Data**:
   Download the datasets to a local cache. This requires the `benchmark` extras.
   ```bash
   uv sync --extra benchmark
   PYTHONPATH=. uv run python scripts/bench/sync_hub.py --configs single_tag_locus_v1_std41h12
   ```

2. **Run Hub Tests**:
   Point the test runner to the local cache directory:
   ```bash
   export LOCUS_HUB_DATASET_DIR=tests/data/hub_cache
   cargo test --release --test regression_icra2020 -- regression_hub_datasets --nocapture
   ```

### Logic-Specific Benchs
For fine-grained benchmarking of specific components (e.g., thresholding, segmentation), use the built-in benches:
```bash
cargo bench
```

---

## Python Benchmarking CLI

The `scripts/locus_bench.py` tool is the central entry point for all performance evaluations.

### Data Preparation
Before running benchmarks, download all required datasets (AprilTag Mosaic and ICRA 2020):
```bash
uv run python scripts/locus_bench.py prepare
```

### Real-World Evaluation
Evaluate performance on the ICRA 2020 dataset scenarios (`forward`, `circle`):
```bash
# Basic run on Locus
uv run python scripts/locus_bench.py run real --scenarios forward

# Compare against OpenCV and AprilTag 3
uv run python scripts/locus_bench.py run real --scenarios forward --compare
```

### Synthetic Benchmarking
Test how the detector scales with the number of tags in the image:
```bash
uv run python scripts/locus_bench.py run synthetic --targets 1,10,50,100 --iterations 50
```

### Bottleneck Profiling
Identify which pipeline stage (Threshold, Segmentation, Quad Extraction, Decoding) is the bottleneck for a given workload:
```bash
uv run python scripts/locus_bench.py profile --targets 100
```

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
- **Performance timelines**: Directly log per-frame latency stats to Rerun for time-series analysis.
