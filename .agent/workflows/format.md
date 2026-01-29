---
description: Format all code (Python & Rust)
---

# Formatting Workflow

This workflow formats the codebase using standard tools.

## Steps

1. **Rust Format**
   Format Rust code.
   ```bash
   cargo fmt --all
   ```

2. **Python Ruff Format**
   Format Python code.
   ```bash
   uv run ruff format .
   ```

3. **Python Ruff Fix**
   Apply safe fixes for lint errors (imports, redundancy).
   ```bash
   uv run ruff check . --fix
   ```
