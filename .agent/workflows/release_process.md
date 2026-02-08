---
description: How to release a new version of Locus using Unified Lockstep Versioning
---

# Release Process

This project uses **Unified Lockstep Versioning** with **0-based SemVer**. All components (`locus-core`, `locus-py`, `pyproject.toml`) are versioned together.

## Prerequisites
- `cargo-release` installed: `cargo install cargo-release`

## Release Steps

1. **Safety Check**: Ensure your working directory is clean and you are on the `main` branch.

2. **Choose the Release Type**:

    - **Scenario A: Feature / Fix (Safe)** - `0.1.0` -> `0.1.1`
      Use for: Performance improvements, new methods, bug fixes.
      ```bash
      cargo release patch --execute --config release.toml
      ```

    - **Scenario B: Breaking Change (Unsafe)** - `0.1.0` -> `0.2.0`
      Use for: Renaming methods, changing signatures, breaking API compatibility.
      ```bash
      cargo release minor --execute --config release.toml
      ```

3. **Publication**:
    - The release command will create a git tag.
    - CI/CD should automatically trigger on this tag to build wheels and upload to PyPI.
    - `locus-py` is NOT published to crates.io (`publish = false`).

## Verification
After release, verify that `Cargo.toml` files and `pyproject.toml` all have the same version number.

## CI/CD Integration

The release process is integrated with GitHub Actions via `.github/workflows/release.yml`.

- **Trigger**: Pushing a tag starting with `v*` (e.g., `v0.1.1`).
- **Core Crate**: Published to crates.io automatically.
- **Python Wheels**: Built for Linux, macOS, and Windows and uploaded to PyPI.
- **Source Distribution**: Built and uploaded to PyPI.

Ensure checking your repository secrets:
- `CARGO_REGISTRY_TOKEN`: For publishing to Crates.io.
- `PYPI_API_TOKEN`: For publishing to PyPI.

### Universal Linux Build Strategy

- **publish-linux**:
    - **Data Center**: `x86_64-unknown-linux-gnu`
    - **Edge / Robotics**: `aarch64-unknown-linux-gnu`
    - Uses **QEMU** for ARM64 emulation.
    - Enforces **manylinux_2_28** for compatibility.
    - Binaries are **stripped** and **LTO**-optimized for size.


