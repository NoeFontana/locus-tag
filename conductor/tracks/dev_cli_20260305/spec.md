# Specification: Isolated Developer CLI and Dependency Management

## Overview
Transform the repository's operational scripts into a unified, isolated developer tool. This ensures that the distributed Python wheel remains minimal (containing only core logic and NumPy) while providing a powerful interface for benchmarking, visualization, and validation.

## Functional Requirements
- **Unified CLI**: Create `tools/cli.py` using `Typer` to consolidate the following operations:
    - `bench`: Run performance benchmarks (previously `locus_bench.py`).
    - `visualize`: Launch the Rerun-based visualizer (previously `visualize.py`).
    - `validate-dicts`: Validate dictionary JSON schemas (previously `validate_dict_schemas.py`).
- **Dependency Isolation**: 
    - Strip `pyproject.toml`'s `project.dependencies` to the bare minimum (e.g., `numpy`, `pyo3`).
    - Define a `dev` or `tools` dependency group using `[tool.uv.dependency-groups]` for:
        - `typer` (CLI framework)
        - `rerun-sdk` (Visualization)
        - `matplotlib`, `tqdm`, `pandas` (Benchmarking)
        - `pytest`, `pytest-cov` (Testing)
- **Local Execution**: Ensure the tool is executed via `uv run tools/cli.py <command>`, guaranteeing it uses the correct locked environment.

## Non-Functional Requirements
- **Zero-Contamination**: The `tools/` directory and its dependencies must NOT be included in the distributed `.whl` artifact.
- **Maintainability**: Use a modular structure within `tools/` if the logic grows beyond a single file.

## Acceptance Criteria
- `uv run tools/cli.py --help` displays all consolidated commands.
- `pip install .` (or building a wheel) results in an artifact that does not include `click`, `typer`, or `rerun-sdk` in its metadata dependencies.
- `locus-tag` wheel does not contain any files from the `tools/` directory.

## Out of Scope
- Moving provenance scripts to `examples/` (reserved for a later phase/track).
- Implementation of new benchmarking or visualization features.
