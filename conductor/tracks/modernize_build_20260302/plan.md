# Implementation Plan: Build System Modernization

## Phase 1: pyproject.toml Restructuring & Metadata [checkpoint: c6bcb9f]
- [x] Task: Prune legacy manifest entries. d4f3d8e
    - [x] Remove `[project.scripts]` table from `pyproject.toml`.
    - [x] Remove `[project.optional-dependencies]` table.
    - [x] Clean up `uv.lock` if necessary.
- [x] Task: Implement PEP 735 Dependency Groups. 91dd0d2
    - [x] Define `[dependency-groups]` in `pyproject.toml`.
    - [x] Migrate `dev`, `lint`, `types`, `bench`, `docs`, and `etl` requirements.
    - [x] Pin core runtime dependencies in `[project.dependencies]`.
- [x] Task: Align Maturin and PyPI Metadata. 058a365
    - [x] Configure `tool.maturin.compatibility = "abi3"`.
    - [x] Update PyPI classifiers (Rust, Typing Stubs).
- [x] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: CI/CD Pipeline Migration [checkpoint: 10b2847]
- [x] Task: Migrate Main CI Workflow (`ci.yml`). 32b6192
    - [x] Update setup steps to use `uv sync --group dev --group lint --group types`.
    - [x] Replace `pip install .[dev]` with `uv sync`.
- [x] Task: Migrate Documentation Workflow (`docs.yml`). d062327
    - [x] Update to use `uv sync --group docs`.
- [x] Task: Migrate Release Workflow (`release.yml`). 69d432e
    - [x] Ensure wheel building uses the new `uv`-based environment.
- [x] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: Documentation & Final Alignment
- [x] Task: Update Contribution Guidelines. 95331d6
    - [x] Update `CONTRIBUTING.md` with `uv` instructions.
    - [x] Document new dependency group structure.
- [x] Task: Audit and Update Task Runners. 0cb586e
    - [x] Check for Makefiles or scripts that rely on old optional-dependencies and update them.
- [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)
