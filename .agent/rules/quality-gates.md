---
description: Mandatory quality gates and definition of done.
---

# Quality Gates

Before marking a task as complete, you MUST pass all gates.

## 1. Mandatory Checks
1. **Lint**: `/lint` (Zero warnings)
2. **Type Check**: `/type_check` (Zero errors)
3. **Format**: `/format`
4. **Test**: `cargo nextest run` (Rust) AND `uv run pytest` (Python)

## 2. Performance Check
If touching the hot path:
* **Forward Eval**: `uv run python -m scripts.bench.run real --compare`
* **Regression**: `cargo test --release --test regression_icra2020 -- --test-threads=1`
