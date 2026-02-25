---
name: Release Process
description: Manage the end-to-end release lifecycle using Unified Lockstep Versioning.
---

# Release Process Skill

This skill ensures that releases are performed safely, consistently, and according to the project's versioning standards.

## 1. Pre-Release Validation
Before initiating a release, ensure the following state is achieved:

- [ ] Working directory is clean (`git status`).
- [ ] You are on the `main` branch.
- [ ] Latest code is pulled from remote.
- [ ] `testing` skill has been successfully completed on the current commit.
- [ ] `linting` and `type_check` workflows have passed.

## 2. Versioning Strategy
We use **Unified Lockstep Versioning**. All components are versioned together.

- **Patch (0.1.x)**: Performance improvements, new methods, bug fixes.
- **Minor (0.x.0)**: Breaking changes, API renames, structural shifts.

## 3. Execution

### Scenario A: Safe (Patch)
```bash
cargo release patch --execute --config release.toml
```

### Scenario B: Breaking (Minor)
```bash
cargo release minor --execute --config release.toml
```

## 4. Post-Release Verification
- [ ] Verify `Cargo.toml` in `crates/locus-core` and `crates/locus-py` have the same version.
- [ ] Verify `pyproject.toml` version matches.
- [ ] Confirm git tag `vX.Y.Z` exists.
- [ ] Check GitHub Actions to ensure the `release.yml` workflow triggers.

**Success Criteria:**
- **Consistency:** All manifest versions are perfectly synchronized.
- **Atomicity:** The release process either completes fully with a tag or makes no changes.
- **Visibility:** Release is visible in the repository and triggers the corresponding CI pipeline.
