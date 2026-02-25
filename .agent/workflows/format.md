---
description: Format all code (Python & Rust)
---

# Formatting Workflow

1. `cargo fmt --all`
2. `uv run ruff format .`
3. `uv run ruff check . --fix`
