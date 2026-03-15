# Feature Development Workflow

This document outlines the mandatory process for implementing new features or fixes in the `locus-tag` repository. Adhering to this workflow ensures high code quality, consistent history, and verified stability.

## 1. Branching Strategy
All new work MUST start from the latest state of the production branch.
*   **Base Branch:** `origin/main`
*   **Feature Branch:** Create a descriptive branch name (e.g., `feat-simd-optimization` or `fix-memory-leak`).
*   **Action:** Always run `git fetch origin main` before branching to ensure you are starting from the most recent code.

## 2. Implementation & Local Validation
*   **Atomic Commits:** Keep commits focused and logically grouped.
*   **Standards:** Ensure code adheres to Rust and Python style guides found in `docs/engineering/`.
*   **Tests:** New features must include corresponding tests.
*   **Pre-flight Check:** Run `uv lock --check` and `cargo check --all-features` locally before pushing.

## 3. Pull Requests
Once implementation is complete, a Pull Request (PR) must be opened.
*   **Title:** Concise and descriptive.
*   **Description:** Briefly explain the "what" and "why" of the change.
*   **CI Dependency:** Never merge a PR until all CI status checks (linting, tests, cross-compilation) have passed.

## 4. Merging Protocol
To ensure that CI tests are fully respected and the branch history remains clean, merging should be automated via the GitHub CLI.

**The Mandatory Merge Command:**
```bash
gh pr merge --auto -r -d
```
*   `--auto`: Enables auto-merge (merges automatically once requirements are met).
*   `-r` (`--rebase`): Uses a rebase merge to maintain a linear history.
*   `-d` (`--delete-branch`): Automatically deletes the feature branch after a successful merge.

## 5. Post-Merge
After the PR is merged, switch back to `main` and pull the latest changes:
```bash
git checkout main
git pull origin main
```
