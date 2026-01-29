---
description: Run all linting checks (Python & Rust)
---

# Linting Workflow

This workflow runs all static analysis and linting tools to ensure code quality.

## Steps

1. **Rust Clippy**
   Run clippy on the workspace.
   ```bash
   cargo clippy --workspace --all-targets --all-features -- -D warnings
   ```

2. **Python Ruff**
   Run ruff linter.
   ```bash
   uv run ruff check .
   ```


