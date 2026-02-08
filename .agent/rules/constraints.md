---
description: Strict prohibitions and safety rules.
---

# Strict Constraints

> [!IMPORTANT]
> These constraints are blocking. Code violating them will be rejected.

## 1. Memory Safety
* **Hot Loop Allocations:**
    * ❌ `Vec::new()`, `Box::new()`, `HashMap::new()` in `detect()`.
    * ✅ `bumpalo::Bump` for all temporary frame data.
    * ✅ `SmallVec` or `[T; N]` for fixed-size lists.

## 2. FFI Safety
* **Python Bindings:**
    * ❌ Copying pixel data to `Vec<u8>`.
    * ✅ `PyReadonlyArray2<u8>` (Zero-Copy).
* **Unsafe usage:**
    * ❌ Naked `unsafe` blocks.
    * ✅ `// SAFETY: ...` describing the invariant (e.g. "Strides checked").

## 3. Error Handling
* ❌ `unwrap()`, `expect()` in library code.
* ✅ `Result<T, E>` with `thiserror`.
