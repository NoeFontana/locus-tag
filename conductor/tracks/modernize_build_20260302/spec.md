# Specification: Build System Modernization & pyproject.toml PEP 735 Migration

## 1. Overview
Modernize the `locus-tag` build system by migrating from legacy optional-dependencies to PEP 735 dependency groups. This track focuses on using `uv` for developer orchestration, pruning unnecessary public metadata, and ensuring consistent ABI3 compatibility for the Python extension.

## 2. Functional Requirements
### 2.1 pyproject.toml Restructuring
- **Pruning:** Remove `[project.scripts]` (internal tools should not be public) and `[project.optional-dependencies]`.
- **Dependency Groups (PEP 735):** Implement `[dependency-groups]` with the following buckets:
    - `dev`: `maturin`, `pytest`, `pytest-cov`, `proptest`.
    - `lint`: `ruff`.
    - `types`: `mypy`, `numpy-stubs`.
    - `bench`: `opencv-python`, `rerun-sdk`, `matplotlib`.
    - `docs`: `mkdocs`, `mkdocs-material`, `mkdocstrings[python]`.
    - `etl`: `huggingface_hub`, `tqdm`.
- **Runtime Lock:** Pin `numpy` and other core runtime dependencies to minimum required versions.
- **Maturin Alignment:**
    - Set `tool.maturin.compatibility = "abi3"`.
    - Target Python 3.10+ for the stable ABI.
    - Add PyPI classifiers for Rust extensions and typing stubs.

### 2.2 CI/CD Orchestration
- Update `ci.yml`, `docs.yml`, and `release.yml`.
- Replace `pip install` with `uv sync --group <name>`.
- Optimize jobs by installing only the necessary dependency groups.

### 2.3 Developer Experience
- Update `CONTRIBUTING.md` to document the `uv` workflow (e.g., `uv sync`, `uv run --group bench`).
- Update any existing local task runners (e.g., Makefiles) if applicable.

## 3. Acceptance Criteria
- [ ] `pyproject.toml` contains no `optional-dependencies` or `project.scripts`.
- [ ] `uv sync` successfully creates a virtual environment with all required groups.
- [ ] CI/CD pipelines (`ci.yml`, `docs.yml`, `release.yml`) pass with zero errors using the new structure.
- [ ] `maturin build --release` produces an ABI3-compatible wheel.
- [ ] `CONTRIBUTING.md` reflects the new dependency management strategy.

## 4. Out of Scope
- Modifying the Rust core or `locus-core` crate logic.
- Adding new features to the benchmarking scripts.
- Migrating the Rust build system away from `cargo`.
