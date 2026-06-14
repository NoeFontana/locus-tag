---
name: Release Process
description: Cut a locus-tag release via cargo-release; CI handles publishing and the GitHub Release page.
---

# Release skill

This skill is a quick-reference for the operator. The full procedure —
including OIDC bootstrap, rollback, and the topology of every CI job —
lives in **[`docs/engineering/release-runbook.md`](../../../docs/engineering/release-runbook.md)**.
Read the runbook first when in doubt.

## Pre-flight
1. Working tree clean, on latest `main`, tests green.
2. `## Unreleased` block in `CHANGELOG.md` contains real entries for
   the new version. `create-github-release` fails if the resulting
   notes section is empty.

## Cut the bump

```bash
git checkout -b release-vX.Y.Z

# patch (perf / bugfix)
cargo release patch --execute --no-publish \
  --allow-branch release-vX.Y.Z --config release.toml

# minor (breaking / structural)
cargo release minor --execute --no-publish \
  --allow-branch release-vX.Y.Z --config release.toml
```

Rename `## Unreleased` → `## [X.Y.Z] - YYYY-MM-DD` in `CHANGELOG.md`,
start a fresh empty `## Unreleased` block, amend the commit, and push.

## Verify
- `gh run watch <release.yml run>` — all 8 jobs green.
- GH Release page renders the CHANGELOG body and lists all wheels +
  sdist + .crate as assets.
- PyPI shows the new version with a PEP 740 attestation badge.
- crates.io shows the new `locus-core` version.
- Docs site exposes `/locus-tag/X.Y/` and `stable` redirects to it.

## When something goes wrong
See **[release-runbook §5 — Rollback](../../../docs/engineering/release-runbook.md#5--rollback)**.
Both registries permit yank but never re-upload of the same version;
ship the fix under `X.Y.Z+1`.

## Dry-run
Tag with an `-rc.N` suffix (e.g., `v0.5.1-rc.1`) to exercise the full
pipeline without touching either registry. See
**[release-runbook §4 — RC convention](../../../docs/engineering/release-runbook.md#4--rc-convention)**.
