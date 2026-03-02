# Quality Gates

Before merging code or marking a task as complete, you MUST pass all of the following gates.

## 1. Mandatory Checks
1. **Lint**: Zero warnings from `cargo clippy` and `ruff check`.
2. **Type Check**: Zero errors from static analysis tools.
3. **Format**: Code must be correctly formatted via `cargo fmt` and `ruff format`.
4. **Test**: Pass all unit tests via `cargo nextest run` (Rust) AND `uv run pytest` (Python).

## 2. Performance Check
If modifying the performance-critical hot path, latency and recall must be validated:
* **Forward Evaluation**: `uv run python -m scripts.bench.run real --compare`
* **Regression Testing**: `cargo test --release --test regression_icra2020 -- --test-threads=1`