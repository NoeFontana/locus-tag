# Quality Gates & Verification

Before any code is merged, it MUST pass the following automated and manual quality gates.

## 1. Mandatory Pre-Commit Checks

Run these commands locally to ensure your code is ready for CI:

```bash
# 1. Rust Formatting & Linting
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings

# 2. Python Formatting & Linting
uv run ruff check .
uv run ruff format .

# 3. Static Type Checking (Python)
uv run mypy .

# 4. Unit Testing
cargo nextest run --all-features
uv run pytest
```

## 2. Performance & Regression Gates

If you are modifying the detection pipeline, math kernels, or SIMD dispatch, you must empirically validate that latency and recall remain within acceptable bounds.

```bash
# 1. Forward Evaluation (Accuracy & Yield)
uv run --group bench python scripts/locus_bench.py run real --compare

# 2. Regression Testing (Sequential for accurate latency)
cargo test --release --test regression_icra2020 -- --test-threads=1

# 3. Snapshot Verification (if output changes are intentional)
cargo insta test --review
```
