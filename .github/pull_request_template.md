<!--
PR template for locus-tag. Keep it short — checklist items only flag
the most common drift modes that aren't caught by CI.
-->

## Summary

<!-- 1-3 sentences: what changed and why. -->

## Checklist

- [ ] **CHANGELOG**: if this PR is user-visible (API change, perf delta, ship-affecting fix, supply-chain bump, docs site change), an entry under `## Unreleased` describes it. Version bumps are deferred to a separate PR — see `feedback_defer_version_bumps`.
- [ ] **README ↔ docs landing sync**: if `README.md`'s user-facing sections changed (capabilities, install, quick-start, performance tables), `docs/index.md`'s landing was updated to match. Performance numbers (`## Performance Profiles`, `## Comparison`) are snippet-included from `README.md`, so editing them in README propagates automatically — but adding a new top-level section is a manual mirror.
- [ ] **Local CI gates** for Rust-touching changes: `cargo fmt --all -- --check` AND `cargo clippy --workspace --all-targets --all-features -- -D warnings` ran clean locally before push (see `feedback_rust_local_ci_gates`).
- [ ] **Doc-only PRs**: `ci.yml`'s `paths-ignore` skips the test suite, so branch protection's required `tests / Rust Lint & Test` check never runs. Merge with `gh pr merge --admin --squash --delete-branch` (or via the UI's admin override).

## Test plan

<!--
Bulleted markdown checklist of TODOs for testing the pull request. For
release-engineering or docs PRs, include the post-merge gh-pages /
mike / CHANGELOG-extractor verifications.
-->
