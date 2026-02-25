---
name: Testing
description: Run and evaluate the comprehensive test suite (Rust unit + Python integration).
---

# Testing Skill

This skill guides you through the process of verifying the correctness and performance stability of the Locus library.

## 1. Rust Unit & Property Tests
Use `cargo-nextest` for fast, parallel execution of the core library tests.

```bash
cargo nextest run --workspace --all-targets --all-features
```

## 2. Python Integration Tests
Verify the zero-copy Python bindings and high-level API.

```bash
uv run pytest
```

## 3. Regression & Performance Validation
Strict evaluation against standard datasets (ICRA 2020) to ensure no regressions in accuracy or latency.

```bash
cargo test --release --test regression_icra2020 -- --test-threads=1
```

**Success Criteria:**
- **Pass/Fail:** All tests must pass (exit code 0).
- **Correctness:** 0% failure rate in `proptest` suites.
- **Accuracy:** RMSE of corner detection must not increase by more than 5% compared to baseline.
- **Latency:** Must remain within 10% of targets defined in `performance_benchmark` skill.
- **Safety:** No new `unwrap()` or `panic!()` in code paths identified as "hot loops".
