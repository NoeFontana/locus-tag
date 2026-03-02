# Python Engineering Guidelines

When modifying Python code in `crates/locus-py` or `scripts/`, focus primarily on zero-copy safety at the FFI boundary and consistent execution environments.

## 1. Core Rules
* **Zero-Copy NumPy:** All image data passed to Rust must be contiguous. Rely on `PyReadonlyArray2<u8>` on the Rust side, and avoid slicing or copying arrays in the hot loop on the Python side.
* **Type Stubs:** Ensure `locus/locus.pyi` is perfectly synchronized with any PyO3 interface changes.
* **Environment:** All dependency management and tool execution must use `uv`.

## 2. Quality Gates
All Python modifications must pass the repository's quality gates:
* **Linting:** Ensure code passes `uv run ruff check .` without errors.
* **Formatting:** Format code using `uv run ruff format .`.
* **Verification:** Run all applicable Python tests with `uv run pytest` to guarantee pipeline integrity.
