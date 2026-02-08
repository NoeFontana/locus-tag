# Contributing to Locus

Thank you for your interest in contributing to Locus! This guide covers the key workflows and rules to follow.

## 🏗️ Development Setup

```bash
# Install dependencies and build Python bindings
uv sync
maturin develop --release --manifest-path crates/locus-py/Cargo.toml
```

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
export LOCUS_DATASET_DIR=/path/to/icra2020
```

### Running Tests

```bash
# Run all regression tests
cargo nextest run --test regression_icra2020 --release

# Run specific subset
cargo nextest run --test regression_icra2020 test_regression_icra2020_forward --release

# Run fixture-based smoke test (no dataset required)
cargo nextest run test_fixture_forward_0037 --release
```

## 📸 Golden Master Snapshots

We use [insta](https://insta.rs/) for snapshot testing. Snapshots live in `crates/locus-core/tests/snapshots/`.

### Reviewing Changes

When detector behavior changes, snapshots may need updating:

```bash
# Run tests and review pending snapshots
cargo insta test --review

# Accept all pending snapshots
cargo insta accept
```

### When to Update Snapshots

- ✅ **Intentional improvements**: Better recall, lower RMSE
- ✅ **Algorithm changes**: New thresholding, refined corner detection
- ❌ **Regressions**: Lower recall or higher RMSE require investigation

Always document why snapshots changed in your PR description.

## ✅ Pre-Commit Checklist

Before submitting a PR:

```bash
# 1. Format and lint
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings

# 2. Run tests
cargo nextest run --all-features

# 3. Verify doctests
cargo test --doc -p locus-core

# 4. Python tests (if applicable)
uv run pytest
```

## 📁 Project Structure

| Path | Description |
|------|-------------|
| `crates/locus-core/` | High-performance Rust detection engine |
| `crates/locus-py/` | PyO3 Python bindings |
| `tests/` | Integration tests and evaluation scripts |
| `benchmarks/` | Criterion-based performance benchmarks |
| `docs/` | MkDocs documentation source |

## 📚 Building Documentation

To build and preview the documentation locally (using `uv`):

```bash
# 1. Install dependencies and project
uv sync --extra docs

# 2. Build the documentation
uv run mkdocs build

# 3. Serve documentation
uv run mkdocs serve
```
