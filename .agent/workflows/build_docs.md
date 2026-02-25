---
description: Build and serve the project documentation
---

# Build Docs Workflow

1. `uv sync --extra docs`
2. `uv run mkdocs build`
3. (Optional) `uv run mkdocs serve`
