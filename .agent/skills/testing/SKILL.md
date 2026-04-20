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

To run the suite:
```bash
TRACY_NO_INVARIANT_CHECK=1 cargo test --release --test regression_icra2020 -- --test-threads=1
```

**Updating Snapshots:**
If intentional changes have been made that alter the metrics, you must update the snapshots. **Always use the release profile** when updating snapshots to avoid extreme execution times:
```bash
# ICRA 2020 suite
TRACY_NO_INVARIANT_CHECK=1 INSTA_UPDATE=always cargo test --release --test regression_icra2020 -- --test-threads=1

# Render-Tag suite (if applicable)
TRACY_NO_INVARIANT_CHECK=1 LOCUS_HUB_DATASET_DIR=../../tests/data/hub_cache INSTA_UPDATE=always cargo test --release --test regression_render_tag -- --test-threads=1

# Distortion-aware suite (Brown-Conrady + Kannala-Brandt)
# One-time setup: sync the two configs into tests/data/hub_cache/ via
#   uv run python tools/bench/sync_hub.py --configs \
#     aprilgrid_distortion_brown_conrady_v1 aprilgrid_distortion_kannala_brandt_v1
# If sync_hub.py fails due to upstream dataset schema drift, fall back to
# direct parquet download via `huggingface_hub.hf_hub_download`.
TRACY_NO_INVARIANT_CHECK=1 LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
  cargo insta test --release --all-features --features bench-internals \
  --test regression_distortion_hub --review
```

**Success Criteria:**
- **Pass/Fail:** All tests must pass (exit code 0).
- **Correctness:** 0% failure rate in `proptest` suites.
- **Accuracy:** RMSE of corner detection must not increase by more than 5% compared to baseline.
- **Latency:** Must remain within 10% of targets defined in `performance_benchmark` skill.
- **Safety:** No new `unwrap()` or `panic!()` in code paths identified as "hot loops".
