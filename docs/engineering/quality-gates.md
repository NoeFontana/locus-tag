# Quality Gates & Verification

Before any code is merged, it MUST pass the following automated and manual quality gates.

## 1. Mandatory Pre-Commit Checks

Run these commands locally to ensure your code is ready for CI:

```bash
# 1. Rust Formatting & Linting
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all

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

# 5. Cross-Compilation Check (aarch64)
# Ensure NEON/SIMD paths compile correctly for ARM targets.
rustup target add aarch64-unknown-linux-gnu
cargo check --target aarch64-unknown-linux-gnu --all-features
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
# 1. Forward Evaluation — ICRA 2020 (Accuracy & Yield)
uv run --group bench tools/cli.py bench real --compare

# 2. Hub Dataset Evaluation — Python CLI (Recall + Pose RMSE)
# Requires hub_cache to be populated via `bench prepare` first.
PYTHONPATH=. uv run --group bench tools/cli.py bench real --hub-config single_tag_locus_v1_std41h12_1920x1080
PYTHONPATH=. uv run --group bench tools/cli.py bench real --hub-config aprilgrid_golden_v1
PYTHONPATH=. uv run --group bench tools/cli.py bench real --hub-config charuco_golden_v1

# 3. Rust Regression Testing (Sequential for accurate latency)
# Requires LOCUS_ICRA_DATASET_DIR to be set.
TRACY_NO_INVARIANT_CHECK=1 LOCUS_ICRA_DATASET_DIR=tests/data/icra2020 cargo test --release --test regression_icra2020 --features bench-internals -- --test-threads=1

# 4. Snapshot Verification & Update
# Runs all regression suites (ICRA, Hub tag-level, Hub board-level, distortion)
# and dictionary parity tests. LOCUS_HUB_DATASET_DIR is required by
# regression_render_tag and regression_distortion_hub; regression_board_hub
# resolves tests/data/hub_cache/ automatically from the workspace root.
#
# The distortion suite (regression_distortion_hub) additionally requires
# syncing the `aprilgrid_distortion_brown_conrady_v1` and
# `aprilgrid_distortion_kannala_brandt_v1` configs — see
# `.agent/skills/testing/SKILL.md` for the one-time setup.
TRACY_NO_INVARIANT_CHECK=1 \
LOCUS_ICRA_DATASET_DIR=tests/data/icra2020 \
LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
cargo insta test --release --all-features --features bench-internals --review
```

## 3. Documentation Quality
Ensure the documentation builds correctly and is complete.

```bash
# 1. Sync dependencies for documentation
uv sync --group docs

# 2. Build the MkDocs static site
uv run --group docs mkdocs build
```
