# Quality Gates & Verification

Before any code is merged, it MUST pass the following automated and manual quality gates.

## 1. Mandatory Pre-Commit Checks

Run these commands locally to ensure your code is ready for CI:

```bash
# 1. Rust Formatting & Linting
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings

# 2. Python Formatting & Linting
uv run ruff check . --fix
uv run ruff format .

# 3. Static Type Checking (Python)
uv run mypy .

# 4. Unit Testing
# We use cargo-nextest for fast, concurrent test execution.
# By default, heavy regression tests are excluded (see .config/nextest.toml).
# Always use --release for Rust tests as debug performance is non-representative.
cargo nextest run --release --all-features

# Build the Python extension in release mode before running Python tests.
uv run maturin develop --release --manifest-path crates/locus-py/Cargo.toml
uv run pytest
```

## 2. Performance & Regression Gates

If you are modifying the detection pipeline, math kernels, or SIMD dispatch, you must empirically validate that latency and recall remain within acceptable bounds.

### Micro-Optimization Protocol
Before performing any micro-performance optimization (e.g., SIMD kernels, hot-loop refactoring):
1. **Setup Benchmark:** Create or update a realistic, isolated benchmark in `crates/locus-core/benches/` using the `divan` framework.
2. **Establish Baseline:** Run the benchmark on the `main` branch or current stable state to establish a statistically significant baseline.
3. **Verify Gain:** Demonstrate a measurable improvement in the targeted metric without regressing other pipeline stages.

```bash
# Example: Running a specific micro-benchmark
cargo bench --bench comprehensive -- "bench_thresholding"
```

### System-Level Verification
```bash
# 1. Forward Evaluation (Accuracy & Yield)
uv run --group bench tools/cli.py bench real --compare

# 2. Regression Testing (Sequential for accurate latency)
cargo test --release --test regression_icra2020 -- --test-threads=1

# 3. Snapshot Verification (if output changes are intentional)
cargo insta test --review

# 4. Documentation Quality
Ensure the documentation builds correctly and is complete.

```bash
# 1. Sync dependencies for documentation
uv sync --group docs

# 2. Build the MkDocs static site
uv run --group docs mkdocs build
```
