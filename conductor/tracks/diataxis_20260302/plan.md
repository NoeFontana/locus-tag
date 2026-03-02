# Implementation Plan: Adopt the Diátaxis Framework for Documentation

## Phase 1: Subdirectory Creation & File Migration [checkpoint: a43fd26]
- [x] Task: Create Diátaxis subdirectories (`docs/tutorials/`, `docs/how-to/`, `docs/explanation/`, `docs/reference/`).
- [x] Task: Move `docs/guide.md` to `docs/tutorials/`.
- [x] Task: Create an initial `docs/how-to/add_dictionary.md` placeholder document.
- [x] Task: Move `docs/architecture.md` and `docs/coordinates.md` to `docs/explanation/`.
- [x] Task: Move `docs/api.md` to `docs/reference/`.
- [x] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

## Phase 2: Navigation & Link Updates [checkpoint: 9067d1c]
- [x] Task: Update `mkdocs.yml` navigation to use the new quadrant structure.
- [x] Task: Fix relative links in the moved markdown files to prevent 404s.
- [x] Task: Ensure `docs/index.md` appropriately acts as a landing page or pointer.
- [x] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

## Phase 3: Build & Verification
- [ ] Task: Run `uv run mkdocs build --strict` to verify site integrity and that no broken links exist.
- [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)