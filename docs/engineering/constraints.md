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

## 3. Unsafe Rust
* ❌ **Forbidden:** Naked `unsafe` blocks.
* ✅ **Required:** Every `unsafe` block must be immediately preceded by a `// SAFETY: ` comment that rigorously justifies why the invariant holds (e.g., "Strides were checked at the FFI boundary").

## 4. Error Handling
* ❌ **Forbidden:** `unwrap()` or `expect()` in library code. (Enforced via `#![deny(clippy::unwrap_used)]`).
* ✅ **Required:** Propagate errors gracefully using `Result<T, E>` and `thiserror` for structured error definitions.
