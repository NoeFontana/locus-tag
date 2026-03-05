# Implementation Plan: Isolated Developer CLI

## Phase 1: Dependency Refactoring [checkpoint: 207f4bf]
Strictly segregate core runtime dependencies from developer tools.

- [x] Task: Audit `locus` Python package source to confirm core dependencies (ensure only `numpy` is strictly required at runtime). 3d6d67c
- [x] Task: Update `pyproject.toml` to move `pydantic` and other non-core deps from `project.dependencies` to `[dependency-groups]`. 1302f70
- [x] Task: Add `typer` to the `dev` or `tools` dependency group. 1302f70
- [x] Task: Execute `uv lock` and verify the environment. 3d6d67c
- [x] Task: Conductor - User Manual Verification 'Phase 1: Dependency Refactoring' (Protocol in workflow.md) 207f4bf


## Phase 2: CLI Scaffolding & Command Integration [checkpoint: ea760fd]
Create the unified entry point and migrate existing scripts.

- [x] Task: Create `tools/` directory and `tools/cli.py` using `Typer`. 3aa18c5
- [x] Task: Migrate `scripts/validate_dict_schemas.py` logic to `tools/cli.py` as `validate-dicts` command. 962eb3e
- [x] Task: Migrate `scripts/locus_bench.py` logic to `tools/cli.py` as `bench` command. 7328284
- [x] Task: Migrate `scripts/debug/visualize.py` logic to `tools/cli.py` as `visualize` command. 82aa6a3
- [x] Task: Conductor - User Manual Verification 'Phase 2: CLI Scaffolding & Command Integration' (Protocol in workflow.md) ea760fd


## Phase 3: Validation & Cleanup
Finalize the isolation and remove redundant files.

- -  - [x] Task: Verify that `uv run tools/cli.py --help` and all subcommands work correctly.
- -  - [x] Task: Build a wheel and inspect metadata to ensure developer dependencies are NOT listed.
- -  - [x] Task: Remove old script files in `scripts/` that have been migrated.
- [ ] Task: Update `README.md` or developer docs to point to the new CLI usage.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Validation & Cleanup' (Protocol in workflow.md)
