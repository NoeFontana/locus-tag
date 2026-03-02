# Specification: Unified Workspace Architecture

## 1. Overview
Implement a comprehensive architectural plan to unify the workspace for human engineers, Google Antigravity, Gemini CLI, and Conductor. This involves establishing `docs/` as the Canonical Knowledge Base (Single Source of Truth), bridging Antigravity via symlinks, and refactoring Conductor's pointer pattern to reduce duplication.

## 2. Functional Requirements
### Phase 1: Establish the Canonical Knowledge Base (SSOT)
- Migrate technical constraints, Rust/Python style guides, and architectural decisions into the `docs/` directory.
- Integrate into existing files (e.g., `docs/architecture.md`) when sensible.
- If the document is purely for code/agent directives, place it under a subfolder within `docs/` (e.g., `docs/engineering/`) for clarity and to avoid cluttering user documentation.
- Format these documents for engineering clarity, stripping out any tool-specific agent directives.

### Phase 2: The Antigravity Bridge (Symlinking)
- Retain the `.agent/rules/` directory (required by Antigravity).
- Delete the duplicated markdown files within `.agent/rules/`.
- Create Git-tracked relative symlinks in `.agent/rules/` pointing directly to their canonical counterparts in `docs/` or `docs/engineering/`.

### Phase 3: The Conductor Pointer Pattern
- **Code Styleguides:** Delete `conductor/code_styleguides/` entirely. Update `conductor/workflow.md` to point directly to the canonical style guides in `docs/`.
- **Informational Files:** Replace the contents of files like `conductor/tech-stack.md` with a strict, two-line markdown directive instructing the LLM to read the corresponding canonical file in `docs/` or `conductor/product.md`.
- **State Isolation:** Ensure Conductor maintains full ownership of `conductor/tracks/` and `conductor/product.md`.
- Add `conductor/tracks/` to `.gitignore` (verify it's fully ignored). Ensure `conductor/setup_state.json` is audited (currently omitting from gitignore based on feedback, unless requested otherwise).

## 3. Acceptance Criteria
- [ ] All rules in `.agent/rules/` are symlinks pointing to `.md` files in `docs/`.
- [ ] `conductor/code_styleguides/` is deleted.
- [ ] `conductor/workflow.md` points to `docs/` for style guides.
- [ ] Conductor's `tech-stack.md` points to the canonical `docs/` file.
- [ ] `conductor/tracks/` is explicitly listed in `.gitignore`.

## 4. Out of Scope
- Rewriting the content of the engineering constraints (only formatting and relocating).
- Changes to the actual CI/CD or build systems beyond documentation.