# Agent Guide

## Build & Test
### Rust
- **Build Command:** `cargo build`
- **Test Command:** `cargo nextest run --all-features`
- **Lint Command:** `cargo clippy --all-targets --all-features -- -D warnings`

### Python
- **Setup:** `uv sync` (or `pip install maturin pytest`)
- **Build Command:** `maturin develop --release --manifest-path crates/locus-py/Cargo.toml`
- **Test Command:** `pytest`
- **Lint Command:** `uv run ruff check .`

## Project Structure
- `/crates`: Source code
  - `locus-core`: Core Rust logic
  - `locus-py`: Python bindings and Python source
- `/tests`: Integration tests
- `pyproject.toml`: Python project configuration (managed by uv/maturin)

## CI Context
If you are fixing a CI failure, please:
1. Analyze the provided error logs.
2. Reproduce the failure locally using the appropriate Test Command (Rust or Python depending on the failure).
3. Fix the code.
4. Verify the fix passes the Test Command before pushing.
