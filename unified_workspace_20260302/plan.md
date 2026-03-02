# Implementation Plan: Unified Workspace Architecture

## Phase 1: Establish Canonical Knowledge Base (SSOT)
- [ ] Task: Create `docs/engineering/` directory for internal code/agent docs.
- [ ] Task: Migrate and format `conductor/code_styleguides/rust.md` to `docs/engineering/rust-style.md`.
- [ ] Task: Migrate and format `conductor/code_styleguides/python.md` to `docs/engineering/python-style.md`.
- [ ] Task: Migrate and format `conductor/code_styleguides/general.md` to `docs/engineering/general-style.md`.
- [ ] Task: Migrate `.agent/rules/` content to `docs/engineering/` and strip agent directives.
    - [ ] `architecture.md` (Integrate into `docs/architecture.md` or move to `docs/engineering/`).
    - [ ] `constraints.md` -> `docs/engineering/constraints.md`.
    - [ ] `core.md` -> `docs/engineering/core.md`.
    - [ ] `quality-gates.md` -> `docs/engineering/quality-gates.md`.
- [ ] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: The Antigravity Bridge (Symlinking)
- [ ] Task: Delete duplicate markdown files in `.agent/rules/`.
- [ ] Task: Create Git-tracked relative symlinks in `.agent/rules/`.
    - [ ] Symlink `architecture.md` -> `../../docs/architecture.md` (or `../../docs/engineering/architecture.md`).
    - [ ] Symlink `constraints.md` -> `../../docs/engineering/constraints.md`.
    - [ ] Symlink `core.md` -> `../../docs/engineering/core.md`.
    - [ ] Symlink `quality-gates.md` -> `../../docs/engineering/quality-gates.md`.
- [ ] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: The Conductor Pointer Pattern
- [ ] Task: Delete `conductor/code_styleguides/` directory.
- [ ] Task: Update `conductor/workflow.md` to point to `docs/engineering/` for style guides.
- [ ] Task: Replace content of `conductor/tech-stack.md` with pointer to `docs/architecture.md` (or relevant canonical doc).
- [ ] Task: Ensure `.gitignore` ignores `conductor/tracks/`.
- [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)