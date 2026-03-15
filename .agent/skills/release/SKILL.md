---
name: Release Process
description: Manage the end-to-end release lifecycle using Unified Lockstep Versioning and Release Branches.
---

# Release Process Skill

This skill ensures that releases are performed safely, consistently, and according to the project's versioning standards.

## 1. Pre-Release Validation
Before initiating a release, ensure the following state is achieved:

- [ ] All feature branches are merged into `main` via Pull Request.
- [ ] Working directory is clean (`git status`).
- [ ] You are on the `main` branch with the latest code pulled.
- [ ] `testing` skill has been successfully completed on the latest `main`.
- [ ] `linting` and `type_check` workflows have passed.

## 2. Versioning Strategy
We use **Unified Lockstep Versioning**. All components are versioned together.

- **Patch (0.1.x)**: Performance improvements, new methods, bug fixes.
- **Minor (0.x.0)**: Breaking changes, API renames, structural shifts.

## 3. Execution Flow

### Phase A: Branching
Create a dedicated release branch from the latest `main`:
```bash
git checkout -b release-vX.Y.Z
```

### Phase B: Version Bump & Tagging
Execute the release locally using `cargo release`. 
> **Note:** Always use `--no-publish` to delegate the actual publishing to GitHub Actions. Use `--allow-branch` to override default branch restrictions.

**Scenario 1: Patch Release**
```bash
cargo release patch --execute --no-publish --allow-branch release-vX.Y.Z --config release.toml
```

**Scenario 2: Minor Release**
```bash
cargo release minor --execute --no-publish --allow-branch release-vX.Y.Z --config release.toml
```

### Phase C: Finalizing
1.  Push the release branch and the tag:
    ```bash
    git push origin release-vX.Y.Z --tags
    ```
2.  Open a Pull Request from `release-vX.Y.Z` to `main` to bring the version bump back to the primary branch.
3.  Merge the PR once CI passes.

## 4. Post-Release Verification
- [ ] Verify `Cargo.toml` in `crates/locus-core` and `crates/locus-py` have the same version.
- [ ] Verify `pyproject.toml` version matches.
- [ ] Confirm git tag `vX.Y.Z` exists on GitHub.
- [ ] Monitor GitHub Actions to ensure the `release.yml` workflow triggers and completes successfully.

**Success Criteria:**
- **Consistency:** All manifest versions are perfectly synchronized across the workspace.
- **Atomicity:** The release process either completes fully with a tag or makes no changes.
- **Delegation:** Publishing is handled securely by CI, not the local developer environment.
