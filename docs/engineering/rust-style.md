# Rust Engineering Guidelines

When modifying the `locus-core` crate, prioritize mechanical sympathy, safety, and strict alignment with the project's data-oriented architecture.

## 1. Core Principles
* **Data-Oriented Architecture:** Respect the cache-locality rules defined in [Architecture](../architecture.md). Design data structures as flat arrays rather than nested pointers.
* **Allocation Policy:** Adhere strictly to the zero-allocation hot-loop rule. Utilize `bumpalo` for all ephemeral frame data (see [Constraints](constraints.md)).
* **Geometric Math:** Use `nalgebra::SMatrix` for small, stack-allocated matrices and vectors. Avoid dynamic `DMatrix` unless absolutely necessary outside the hot path.
* **Zero Panics:** Ensure no code path can panic. Rely on `Result` types and `#![deny(clippy::unwrap_used)]`.

## 2. SIMD & Optimization
* **Runtime Dispatch:** Use the `multiversion` crate to compile targeted kernels for AVX2, AVX-512, and NEON architectures.
* **Branchless Logic:** Inside pixel-processing loops, prefer branchless operations (masks, CMOVs) over unpredictable conditional jumps.

## 3. Documentation & Modularity
* **Module Boundaries:** Keep modules focused (e.g., `threshold`, `quad`, `decoder`). Hide internal implementation details behind `pub(crate)`.
* **Docstrings:** All public interfaces must be documented. We compile with `#![warn(missing_docs)]`. Provide code examples in docstrings where applicable.

## 4. Verification Workflow
* Always build and test in `--release` mode when evaluating algorithmic changes, as debug performance is not representative of SIMD paths.
* Run `cargo clippy` frequently and resolve all warnings before committing.
