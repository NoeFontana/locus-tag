# Release runbook

How to ship a locus-tag patch to PyPI, crates.io, GitHub Releases, and
the versioned docs site. Two workflows fan out from a tag push:

- [`.github/workflows/release.yml`](https://github.com/NoeFontana/locus-tag/blob/main/.github/workflows/release.yml)
  — wheels (PyPI), crate (crates.io), and the GitHub Release page with
  CHANGELOG-extracted notes + attached artefacts.
- [`.github/workflows/docs.yml`](https://github.com/NoeFontana/locus-tag/blob/main/.github/workflows/docs.yml)
  — versioned docs site, deployed to `gh-pages` via [`mike`](https://github.com/jimporter/mike).

`cargo-release` (config:
[`release.toml`](https://github.com/NoeFontana/locus-tag/blob/main/release.toml))
drives the version bump locally; CI takes over once the `v*` tag
lands. This document is the operator's end-to-end view.

## 1 — Topology

A `v*` tag push fans out as follows:

```
release.yml:
  verify-tag                                            (refuses mismatched tags)
    ├─► build-rust                                       (cargo package -p locus-core)
    ├─► build-wheels-linux   (manylinux + musllinux × x86_64/aarch64)
    ├─► build-wheels-macos   (x86_64/aarch64)
    ├─► build-wheels-windows (x64)
    └─► build-sdist
                                  │
                                  ▼
                          wheel-validate                 (smoke on x86_64 + aarch64 QEMU)
                                  │
   ┌──────────────────────────────┼──────────────────────────────┐
   ▼                              ▼                              ▼
 publish-pypi             publish-crates-io           create-github-release
 (OIDC + PEP 740)         (OIDC, locus-core)          (notes + artefacts; --prerelease on RC)

docs.yml:
  build + spell ─► deploy                                (mike → gh-pages, X.Y + stable alias)
```

`verify-tag` gates publishing: tags ending in `-rc.*` set
`outputs.publish=false`, both publish jobs skip, and
`create-github-release` runs with `--prerelease`. The docs `deploy` job
deliberately skips RC tags so the `stable` alias keeps pointing at the
last real release.

A monthly cron (`0 6 1 * *`) on `release.yml` exercises only the Linux
wheel build to catch toolchain drift between releases. macOS (10×
billing) and Windows (2× billing) are reserved for actual release
builds.

Neither publish job is wrapped in a GitHub `environment:` for now —
GitHub Actions only enforces required-reviewer protection on private
repos under Pro/Team tiers. The safety net is `verify-tag` plus
deliberate tag pushes. When this repo upgrades to a tier that enforces
environment protections, set `environment: release` on `publish-pypi`
and `publish-crates-io` and add reviewers there.

## 2 — One-time setup

Done once per repo, then the per-release flow takes over.

### 2.1 PyPI Trusted Publisher

At <https://pypi.org/manage/project/locus-tag/settings/publishing/>,
add a Trusted Publisher entry:

| Field | Value |
|---|---|
| Owner | `NoeFontana` |
| Repository | `locus-tag` |
| Workflow | `release.yml` |
| Environment | *(leave blank)* |

When the repo eventually goes public — or this account upgrades to a
tier that enforces environment protection rules — set Environment to
`release` here and add `environment: release` to the `publish-pypi`
job.

### 2.2 crates.io Trusted Publisher

crates.io scopes Trusted Publishers per crate. At
<https://crates.io/me/trusted-publishers> add:

| Crate | Repository | Workflow | Environment |
|---|---|---|---|
| `locus-core` | `NoeFontana/locus-tag` | `release.yml` | *(blank)* |

`locus-py` is not published (`publish = false` in its `Cargo.toml`).

### 2.3 gh-pages bootstrap (for `mike` versioned docs)

1. Push the docs workflow → first deploy lands a `gh-pages` branch
   with `main` + `latest`.
2. Repo Settings → Pages → Source: **Deploy from a branch**, Branch:
   **gh-pages / (root)**.
3. Locally (one-shot, by a maintainer with push perms):
   ```bash
   uv run mike set-default --push latest
   ```
   This redirects `https://noefontana.github.io/locus-tag/` to
   `/locus-tag/latest/`. After the first real release tag fires (which
   updates `stable`), switch the default:
   ```bash
   uv run mike set-default --push stable
   ```

## 3 — Per-release flow (operator)

### 3.1 Pre-release checklist
- [ ] All feature branches merged into `main`.
- [ ] Working tree clean.
- [ ] On `main` with latest pulled.
- [ ] `cargo nextest run` and `pytest` green on the latest `main`
      (run via the `testing` skill).
- [ ] **CHANGELOG block exists for the new version.** Under
      `## Unreleased` confirm the entries are real and complete; the
      `create-github-release` job will fail if the resulting section
      is empty or missing.

### 3.2 Cut the version bump
Create a release branch from latest `main`:

```bash
git checkout -b release-vX.Y.Z
```

Run `cargo release` with `--no-publish` (CI handles the publish):

```bash
# patch (perf/bugfix)
cargo release patch --execute --no-publish \
  --allow-branch release-vX.Y.Z --config release.toml

# minor (breaking / structural)
cargo release minor --execute --no-publish \
  --allow-branch release-vX.Y.Z --config release.toml
```

`cargo-release` bumps `locus-core` and `locus-py` to the new version,
runs `uv lock` (via `pre-release-hook`), updates `pyproject.toml`
(via `pre-release-replacements`), commits, tags `vX.Y.Z`, and pushes.

Edit `CHANGELOG.md` in the same branch: rename `## Unreleased` to
`## [X.Y.Z] - YYYY-MM-DD` and start a fresh empty `## Unreleased` block
above it. Amend the bump commit (or add a second commit) and force-push.

### 3.3 Open the back-merge PR
Open `release-vX.Y.Z` → `main` via `gh pr create --draft`. Wait for
the standard PR-gate CI (`ci.yml`) to go green, then mark ready and
squash-merge once approved. The `v*` tag was already pushed in §3.2,
so `release.yml` may already be running — that's fine, it operates off
the tag, not the PR.

### 3.4 Watch the publish
- `gh run watch <release.yml run id>` until all 8 jobs are green.
- Verify the GitHub Release page renders the CHANGELOG body and lists
  all wheels + sdist + .crate as assets.
- Verify on <https://pypi.org/project/locus-tag/> that the new version
  is live with a sigstore attestation badge (PEP 740).
- Verify on <https://crates.io/crates/locus-core> that the new
  version is live.
- Verify the docs site: `/locus-tag/X.Y/` exists and `/locus-tag/stable/`
  redirects there. The `mike deploy` step also updates `latest` on the
  next push to `main`.

## 4 — RC convention

We use the `-rc.N` suffix on tags to dry-run the entire release
pipeline without touching either registry.

```bash
# Bump to 0.5.1-rc.1 locally (no automation — edit manifests by hand
# and tag manually, or use `cargo release rc` if available).
git tag -a v0.5.1-rc.1 -m "dry-run release build (build + smoke, publish skipped)"
git push origin v0.5.1-rc.1
```

What happens:
- `verify-tag` strips `-rc.1` before comparing against on-disk
  versions, so the manifests can stay at `0.5.0` until the real bump.
- `outputs.publish=false`, so `publish-pypi` and `publish-crates-io`
  are skipped.
- All `build-wheels-*`, `build-sdist`, and `wheel-validate` jobs run.
- `create-github-release` posts a GitHub Release with body extracted
  from `## [0.5.1]` if it exists, otherwise `## Unreleased` with a
  one-line "release candidate" preamble. The Release is marked
  `--prerelease`.
- `docs.yml` deploy job skips the tag-deploy step entirely for RC
  tags, so `stable` stays at the last real release.

When the dry-run is green, delete the RC GitHub Release (the tag can
stay for history), then follow §3 to ship the real version.

## 5 — Rollback

### 5.1 PyPI

PyPI permits **yank** but never re-upload of the same version. To
withdraw a botched release:

```bash
# from the PyPI project page → "Manage release" → "Yank"
# or via twine / pypi-cli
```

Yanked versions can no longer be `pip install`'d without an exact
`==` pin, but stay available for reproducibility. To ship a fix,
publish under `X.Y.Z+1` — there is no rollback to a previous version
number.

### 5.2 crates.io

crates.io permits `cargo yank` only as a deprecation signal:

```bash
cargo yank --version X.Y.Z --crate locus-core
```

Yanked versions can no longer be selected by a fresh resolve, but
existing `Cargo.lock` files continue to use them. To ship a fix,
publish under `X.Y.Z+1`.

### 5.3 GitHub Release

```bash
gh release delete vX.Y.Z --cleanup-tag  # drops the GH Release AND the tag
# Tag deletion is destructive — only do this if you also intend to
# re-tag the same SHA. Otherwise:
gh release delete vX.Y.Z                # keeps the tag for history
```

### 5.4 Docs (`mike`)

```bash
# Remove the versioned page entirely
uv run mike delete --push X.Y

# Or just rebuild it from a different ref
git checkout vX.Y.Z-fix
uv run mike deploy --push --update-aliases X.Y stable
```

### 5.5 Partial-failure scenario: publish-pypi succeeded, publish-crates-io failed

The most common rollback case. `publish-pypi` succeeded and the wheel
is live; `publish-crates-io` failed (e.g., transient network, OIDC
hiccup, or `crates.io` index lag). Options:

1. **Re-run only the failed job** if the failure is transient
   (`gh run rerun <id> --failed`). Most cases resolve here.
2. If the failure is permanent (e.g., a typo'd `Cargo.toml`
   `description`), yank the PyPI release and bump to `X.Y.Z+1` for
   both registries.

The reverse case (`publish-crates-io` succeeded, `publish-pypi`
failed) is symmetric: yank the crate, bump, re-tag.

## 6 — Release history

| Tag | Date | Type | Notes |
|---|---|---|---|
| `v0.5.0-rc.1` | 2026-05-20 | RC dry-run | Validated pipeline before v0.5.0 |
| `v0.5.0` | 2026-05-20 | Minor | Unified JSON-profile config refactor |

Append new entries here as releases land.
