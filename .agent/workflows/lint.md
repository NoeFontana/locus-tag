---
description: Run all linting checks (Python & Rust)
---

# Linting Workflow

1. `cargo clippy --workspace --all-targets --all-features -- -D warnings`
2. `uv run ruff check .`
