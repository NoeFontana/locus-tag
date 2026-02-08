---
description: Build and serve the project documentation
---
1. Install dependencies and project (includes docs extras)
// turbo
2. uv sync --extra docs

3. Build the static site
// turbo
4. uv run mkdocs build

5. (Optional) Serve the documentation
6. uv run mkdocs serve
