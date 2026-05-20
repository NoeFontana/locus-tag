# Locus task runner. Run `just` to list all recipes.
# Requires: just, uv, cargo, cargo-nextest, cargo-insta, cargo-deny.

set shell := ["bash", "-cu"]

# Default recipe: list available recipes.
default:
    @just --list

# ---------------------------------------------------------------------------
# Bootstrap & build
# ---------------------------------------------------------------------------

# Set up the dev environment from a fresh clone.
bootstrap:
    uv sync --all-extras
    uv run maturin develop --release --manifest-path crates/locus-py/Cargo.toml

# Fast iteration build (debug Rust, fast link).
develop:
    uv run maturin develop --manifest-path crates/locus-py/Cargo.toml

# Optimized abi3 wheel into target/wheels/.
build:
    uv run maturin build --release --manifest-path crates/locus-py/Cargo.toml

# ---------------------------------------------------------------------------
# Tests (1:1 with PR-gate jobs in _tests.yml)
# ---------------------------------------------------------------------------

# Run all tests (Rust + Python + doc + insta).
test: test-rust test-py test-doc test-insta

test-rust:
    cargo nextest run --profile ci --workspace --all-features

test-py:
    uv run pytest

test-doc:
    cargo test --doc --all-features

# Snapshot parity check (cargo-insta in --check mode).
test-insta:
    # TRACY_NO_INVARIANT_CHECK=1 avoids a known Linux environment quirk.
    TRACY_NO_INVARIANT_CHECK=1 cargo insta test --release --all-features --features bench-internals --check

# Aarch64 cross-compile check (mirrors the CI rust-extras job).
check-aarch64:
    rustup target add aarch64-unknown-linux-gnu
    cargo check --target aarch64-unknown-linux-gnu --all-features

# ---------------------------------------------------------------------------
# Lint & format
# ---------------------------------------------------------------------------

# Run all linters (CI-equivalent, read-only).
lint: lint-rust lint-py

lint-rust:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets --all-features -- -D warnings

lint-py:
    uv run ruff check .
    uv run ruff format --check .
    uv run --group types --group bench --group etl --group docs basedpyright

# Auto-format Rust and Python (writes changes).
fmt:
    cargo fmt --all
    uv run ruff format .
    uv run ruff check --fix .

# ---------------------------------------------------------------------------
# Profile schema parity (export_profile_schema --check)
# ---------------------------------------------------------------------------

schema-check:
    uv run python tools/export_profile_schema.py --check

# ---------------------------------------------------------------------------
# Audit & docs
# ---------------------------------------------------------------------------

# Supply-chain audit (licenses, advisories, multiple-versions, sources).
audit:
    cargo deny check

# Build the rustdoc site under -D warnings (mirrors the CI rust-extras step).
doc-rust:
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features

# Build the MkDocs site into site/ (strict — mirrors CI).
build-docs:
    uv run --group docs mkdocs build --strict

serve-docs:
    uv run --group docs mkdocs serve

# Spell-check user-facing docs (excludes engineering/ internal dirs).
lint-docs:
    uv run --group docs codespell docs/ --skip "docs/engineering"

# ---------------------------------------------------------------------------
# Aggregator — what to run before pushing a PR.
# ---------------------------------------------------------------------------

pre-pr: lint audit test schema-check

# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

# Remove all build artifacts and the venv.
clean:
    cargo clean
    rm -rf .venv target/wheels
    find crates/locus-py -name "_core*.so" -delete -o -name "_core*.pyd" -delete

# Print toolchain versions (paste into bug reports).
versions:
    @echo "rustc:  $(rustc --version)"
    @echo "cargo:  $(cargo --version)"
    @echo "uv:     $(uv --version)"
    @echo "python: $(uv run python --version)"

# ---------------------------------------------------------------------------
# Diagnostics harnesses
# ---------------------------------------------------------------------------

# Phase 0 rotation-tail harness (requires bench-internals build of locus-py).
diagnostics-rotation-tail-phase0:
    # Build prerequisite:
    #   uv run maturin develop --release --manifest-path crates/locus-py/Cargo.toml --features bench-internals
    PYTHONPATH=. uv run --group bench tools/cli.py bench rotation-tail-diag \
        --hub-config locus_v1_tag36h11_1920x1080 \
        --profile high_accuracy \
        --pose-mode Accurate
