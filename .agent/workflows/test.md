---
description: Run comprehensive test suite (Rust unit + Python integration)
---

# Testing Workflow

This workflow runs the full test suite for both Rust and Python components.

## Steps

1. **Rust Tests**
   Run Rust unit and property tests.
   // turbo
   ```bash
   cargo nextest run --workspace --all-targets --all-features
   ```

2. **Python Integration Tests**
   Run Python integration tests using pytest.
   // turbo
   ```bash
   uv run pytest
   ```

3. **Regression Tests**
   Run the regression suite (latency/accuracy).
   // turbo
   ```bash
   cargo test --release --test regression_icra2020 -- --test-threads=1
   ```
