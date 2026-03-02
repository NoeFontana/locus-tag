# Implementation Plan: Build System Modernization

## Phase 1: pyproject.toml Restructuring & Metadata
- [x] Task: Prune legacy manifest entries. d4f3d8e
    - [x] Remove `[project.scripts]` table from `pyproject.toml`.
    - [x] Remove `[project.optional-dependencies]` table.
    - [x] Clean up `uv.lock` if necessary.
- [ ] Task: Implement PEP 735 Dependency Groups.
    - [ ] Define `[dependency-groups]` in `pyproject.toml`.
    - [ ] Migrate `dev`, `lint`, `types`, `bench`, `docs`, and `etl` requirements.
    - [ ] Pin core runtime dependencies in `[project.dependencies]`.
- [ ] Task: Align Maturin and PyPI Metadata.
    - [ ] Configure `tool.maturin.compatibility = "abi3"`.
    - [ ] Update PyPI classifiers (Rust, Typing Stubs).
- [ ] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: CI/CD Pipeline Migration
- [ ] Task: Migrate Main CI Workflow (`ci.yml`).
    - [ ] Update setup steps to use `uv sync --group dev --group lint --group types`.
    - [ ] Replace `pip install .[dev]` with `uv sync`.
- [ ] Task: Migrate Documentation Workflow (`docs.yml`).
    - [ ] Update to use `uv sync --group docs`.
- [ ] Task: Migrate Release Workflow (`release.yml`).
    - [ ] Ensure wheel building uses the new `uv`-based environment.
- [ ] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: Documentation & Final Alignment
- [ ] Task: Update Contribution Guidelines.
    - [ ] Update `CONTRIBUTING.md` with `uv` instructions.
    - [ ] Document new dependency group structure.
- [ ] Task: Audit and Update Task Runners.
    - [ ] Check for Makefiles or scripts that rely on old optional-dependencies and update them.
- [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)
