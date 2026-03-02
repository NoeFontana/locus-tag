# Implementation Plan: Unified Workspace Architecture

## Phase 1: Establish Canonical Knowledge Base (SSOT) [checkpoint: fc8d338]
- [x] Task: Create `docs/engineering/` directory for internal code/agent docs.
- [x] Task: Migrate and format `conductor/code_styleguides/rust.md` to `docs/engineering/rust-style.md`.
- [x] Task: Migrate and format `conductor/code_styleguides/python.md` to `docs/engineering/python-style.md`.
- [x] Task: Migrate and format `conductor/code_styleguides/general.md` to `docs/engineering/general-style.md`.
- [x] Task: Migrate `.agent/rules/` content to `docs/engineering/` and strip agent directives.
    - [x] `architecture.md` (Integrate into `docs/architecture.md` or move to `docs/engineering/`).
    - [x] `constraints.md` -> `docs/engineering/constraints.md`.
    - [x] `core.md` -> `docs/engineering/core.md`.
    - [x] `quality-gates.md` -> `docs/engineering/quality-gates.md`.
- [x] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: The Antigravity Bridge (Symlinking) [checkpoint: c0ea5cc]
- [x] Task: Delete duplicate markdown files in `.agent/rules/`.
- [x] Task: Create Git-tracked relative symlinks in `.agent/rules/`.
    - [x] Symlink `architecture.md` -> `../../docs/architecture.md` (or `../../docs/engineering/architecture.md`).
    - [x] Symlink `constraints.md` -> `../../docs/engineering/constraints.md`.
    - [x] Symlink `core.md` -> `../../docs/engineering/core.md`.
    - [x] Symlink `quality-gates.md` -> `../../docs/engineering/quality-gates.md`.
- [x] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: The Conductor Pointer Pattern
- [x] Task: Delete `conductor/code_styleguides/` directory.
- [x] Task: Update `conductor/workflow.md` to point to `docs/engineering/` for style guides.
- [x] Task: Replace content of `conductor/tech-stack.md` with pointer to `docs/architecture.md` (or relevant canonical doc).
- [x] Task: Ensure `.gitignore` ignores `conductor/tracks/`.
- [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)