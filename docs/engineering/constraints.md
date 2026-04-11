# Strict Constraints

These constraints represent the non-negotiable laws of the Locus codebase. Violations will result in immediate PR rejection.

## 1. Memory Safety & Allocations
Locus achieves its latency targets by strictly avoiding the system allocator (`malloc`/`free`) in the hot loop.
* **Hot Loop (`detect()`):**
  * ❌ **Forbidden:** `Vec::new()`, `Box::new()`, `HashMap::new()`, or any implicit heap allocations.
  * ✅ **Required:** Use the `bumpalo::Bump` arena for all ephemeral per-frame data.
  * ✅ **Required:** Adhere to the [DetectionBatch (SoA) Contract](./detection-batch-contract.md) to ensure zero-allocation performance and cache efficiency.
  * ✅ **Allowed:** Stack-allocated structures like `SmallVec`, `arrayvec`, or fixed-size arrays `[T; N]`.

## 2. FFI & Zero-Copy Boundaries
The Rust-Python boundary must be invisible to performance.
* **Image Data:**
  * ❌ **Forbidden:** Copying or cloning NumPy arrays into Rust `Vec<u8>`. 
  * ❌ **Forbidden:** Passing non-contiguous views (stride_x != 1) to high-performance detection methods.
  * ✅ **Required:** Use `PyReadonlyArray2<u8>` to leverage the Python Buffer Protocol. Validate strides early and once.
  * ✅ **Required:** Throw a `ValueError` for non-contiguous arrays to force users to use `.ascontiguousarray()`.
  * ✅ **Required:** Image buffers used with SIMD-vectorized kernels (e.g., AVX2 gather) MUST have at least **3 bytes of padding** at the end to prevent out-of-bounds reads during 32-bit pixel fetching.

## 3. Unsafe Rust
* ❌ **Forbidden:** Naked `unsafe` blocks.
* ✅ **Required:** Every `unsafe` block must be immediately preceded by a `// SAFETY: ` comment that rigorously justifies why the invariant holds (e.g., "Strides were checked at the FFI boundary").

## 4. Error Handling
* ❌ **Forbidden:** `unwrap()` or `expect()` in library code. (Enforced via `#![deny(clippy::unwrap_used)]`).
* ✅ **Required:** Propagate errors gracefully using `Result<T, E>` and `thiserror` for structured error definitions.

## 5. Dependency Hygiene
Keeping the dependency graph lean is as important as keeping the hot loop lean — every unnecessary crate adds compile time, binary size, and a potential instability vector.

* **Randomness:**
  * ❌ **Forbidden:** `rand`, `rand_distr`, or any stochastic crates in `[dependencies]` (even as a non-optional entry).
  * ✅ **Required:** Place randomness crates in `[dev-dependencies]` or, when needed by `bench-internals` library code (e.g., `test_utils`), as `optional = true` deps activated exclusively by the `bench-internals` feature.
  * **Rationale:** The production binary must be mathematically deterministic. Stochastic code has no place in a library that targets reproducible robotic perception pipelines.

* **Feature Discipline for Math Crates:**
  * ❌ **Forbidden:** Bare version strings (e.g., `nalgebra = "0.34"`) for heavy mathematical crates. Default features can silently expand across minor releases.
  * ✅ **Required:** Always set `default-features = false` with an explicit `features = [...]` allowlist for `nalgebra` and `ndarray`. Pin only what is actually used.
  * **Current allowlists:** `nalgebra = { ..., default-features = false, features = ["std"] }` · `ndarray = { ..., default-features = false, features = ["std"] }`.

* **Telemetry Erasure:**
  * ❌ **Forbidden:** `tracing` dependencies without `release_max_level_info` in any Locus crate.
  * ✅ **Required:** Every `tracing` entry — in both `locus-core` and `locus-py` — must carry `features = ["release_max_level_info"]` to guarantee compile-time erasure of `debug!`/`trace!` spans in release builds. This is the only way to uphold the zero-overhead telemetry contract.

## 6. Performance Reporting & Benchmarking
To ensure reproducibility and scientific integrity, all benchmark reports must be grounded in verified system state.
* ❌ **Forbidden:** Placeholder or assumed hardware specifications (e.g., "Intel CPU" without verification).
* ✅ **Required:** Every performance or accuracy report MUST include verified hardware metadata obtained via system tools (e.g., `lscpu`, `/proc/cpuinfo`) during the same session the benchmark was executed.
* ✅ **Required:** State the build profile (`--release`), thread count, and environment variables (e.g., `RAYON_NUM_THREADS`) used during the run.
