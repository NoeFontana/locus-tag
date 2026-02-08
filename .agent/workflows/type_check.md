---
description: Run type checking (Python & Rust)
---

# Type Checking Workflow

This workflow verifies type safety for both Rust and Python.

## Steps

1. **Rust Check**
   Run cargo check (fast compilation check).
   // turbo
   ```bash
   cargo check --workspace --all-targets --all-features
   ```

2. **Python Pyrefly**
   Run pyrefly type checker.
   ```bash
   uv run pyrefly check .
   ```
