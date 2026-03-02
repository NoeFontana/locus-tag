# Rust Engineering Guidelines

When modifying Rust code in `crates/locus-core`, strictly adhere to the following standards:

## 1. Core Principles
* **Architectural Invariants:** Follow the cache-locality and SIMD dispatch expectations defined in [Architecture](architecture.md).
* **Strict Constraints:** Adhere to the zero-allocation arena rules and the strict zero-panic policy (`Result` types, `thiserror`, no `.unwrap()`). See [Constraints](constraints.md).
* **Core Logic Guide:** Follow mathematical and geometric standards (e.g., using `nalgebra::SMatrix`). See [Core Guidelines](core.md).

## 2. Quality Gates
When refactoring or verifying changes, ensure the code passes the following checks:
* **Format:** Code must be formatted using `cargo fmt --all`.
* **Lint:** Code must pass `cargo clippy --all-targets --all-features -- -D warnings` with zero warnings.
* **Test & Benchmark:** Run the full test suite and performance benchmarks to ensure no regressions.
