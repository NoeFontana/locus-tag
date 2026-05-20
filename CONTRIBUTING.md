# Contributing to Locus

Thank you for your interest in contributing to Locus! This guide covers the key workflows and rules to follow.

## 🏗️ Development Setup

Locus uses [just](https://github.com/casey/just) as a task runner — every CI
job has a 1:1 local recipe so `just <recipe>` reproduces exactly what runs
in CI.

```bash
# Install just (once): `cargo install just` or `brew install just`.

# Install dependencies and build Python bindings (debug-or-release as needed).
just bootstrap
```

See `just --list` for the full recipe surface.

## 📐 The Zero-Allocation Rule

The detection hot loop (`detect()` → Threshold → Segmentation → Quad → Decode) **must not perform heap allocations**.

### How to Verify

1. **Code Review**: Ensure no `Vec::new()`, `Box::new()`, or `HashMap::new()` inside per-pixel/per-segment loops.

2. **Use Arena Allocation**: All ephemeral data uses `bumpalo::Bump`:
   ```rust
   let contours = arena.alloc_slice_fill_copy(len, 0u8);
   ```

3. **Profile with DHAT** (heap profiler):
   ```bash
   cargo install dhat
   # Add #[global_allocator] static ALLOC: dhat::Alloc = dhat::Alloc;
   cargo test --release
   # Review dhat-heap.json for allocations in hot paths
   ```

## 🧪 Running the Regression Suite

The ICRA 2020 regression tests validate detector accuracy against golden master snapshots.

### Prerequisites

Download the ICRA 2020 dataset and set the environment variable:
```bash
export LOCUS_ICRA_DATASET_DIR=/path/to/icra2020
```

### Running Tests

```bash
# 1. Run core regression suite (Forward tests + Fixtures)
# Must use --release for meaningful performance validation
# Requires the 'bench-internals' feature.
cargo test --release --test regression_icra2020 --features bench-internals

# 2. Run extended regression suite (Circle, Random, Rotation - heavy)
LOCUS_EXTENDED_REGRESSION=1 cargo test --release --test regression_icra2020 --features bench-internals

# 3. Run fixture-based smoke test (no dataset required)
cargo test --release --test regression_icra2020 regression_fixtures --features bench-internals
```

## 📸 Golden Master Snapshots

We use [insta](https://insta.rs/) for snapshot testing. Snapshots live in `crates/locus-core/tests/snapshots/`.

### Reviewing Changes

When detector behavior changes, snapshots may need updating:

```bash
# Run tests and review pending snapshots
# TRACY_NO_INVARIANT_CHECK=1 is recommended on some Linux environments.
TRACY_NO_INVARIANT_CHECK=1 cargo insta test --release --all-features --features bench-internals --review
```

### When to Update Snapshots

- ✅ **Intentional improvements**: Better recall, lower RMSE
- ✅ **Algorithm changes**: New thresholding, refined corner detection
- ❌ **Regressions**: Lower recall or higher RMSE require investigation

Always document why snapshots changed in your PR description.

## ✅ Pre-PR Checklist

Before submitting a PR, run the full local gate:

```bash
just pre-pr     # lint + audit + test + schema-check
```

This expands to `just lint` + `just audit` (cargo-deny) + `just test` (Rust
nextest, Python pytest, Rust doctests, insta snapshot parity) + `just
schema-check`. Each recipe maps 1:1 to a CI job, so a green `just pre-pr`
mirrors a green PR gate.

To auto-fix formatting:

```bash
just fmt        # cargo fmt + ruff format + ruff check --fix
```

Individual recipes are also available — see `just --list`.

## 📁 Project Structure

| Path | Description |
|------|-------------|
| `crates/locus-core/` | High-performance Rust detection engine |
| `crates/locus-py/` | PyO3 Python bindings |
| `tests/` | Integration tests and evaluation scripts |
| `benchmarks/` | Criterion-based performance benchmarks |
| `docs/` | MkDocs documentation source |

## 📚 Building Documentation

To build and preview the documentation locally:

```bash
# 1. Install dependencies
uv sync --group docs --group dev

# 2. Build the documentation
uv run mkdocs build

# 3. Serve documentation
uv run mkdocs serve
```

## 📦 Dependency Groups (PEP 735)

We use `uv` and PEP 735 dependency groups to isolate development tools. When running scripts, use the appropriate group:

- `dev`: Core development tools (`maturin`, `pytest`, `pytest-cov`)
- `lint`: Code formatting and analysis (`ruff`)
- `types`: Type checker and stubs (`basedpyright`, `pandas-stubs`, `types-tqdm`)
- `bench`: Benchmarking and visualization (`opencv-python-headless`, `rerun-sdk`, `datasets[vision]`, `pupil-apriltags`)
- `docs`: Documentation generation (`mkdocs`, `mkdocstrings`)
- `etl`: Data tools (`huggingface-hub`, `tqdm`)

**Example:** Running the benchmark script
```bash
uv run --group bench tools/cli.py bench
```
