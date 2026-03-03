# Specification: Zero-Copy Pointer Enforcement (Project 1)

## Overview
This track implements "Zero-Copy Pointer Enforcement" for the `locus-tag` FFI boundary. By shifting from a "fallback-copy" model to a strict "zero-copy" model, we force hardware sympathy and eliminate hidden performance bottlenecks (up to 110ms per ingestion) caused by implicit memory copies of non-contiguous NumPy arrays.

## Functional Requirements
1. **Audit & Isolate Boundary:** Identify all PyO3 wrapper functions in `crates/locus-py/src/lib.rs` that accept image data via `PyReadonlyArray2<u8>`.
2. **Strict NumPy View Enforcement:** Update `prepare_image_input` to strictly require C-contiguous memory.
3. **Hard Error on Non-Contiguity:** Remove the `ImageInput::Owned` fallback path and the `eprintln!` warning. Instead, throw a `ValueError` with a clear message: `"Array must be C-contiguous. Use .ascontiguousarray() to avoid performance-killing copies."`
4. **Targeted Scope:** Apply this enforcement to the following high-performance methods:
   - `Detector.detect()`
   - `Detector.detect_with_options()`
   - `Detector.detect_with_stats()`
   - `Detector.extract_candidates()`
   - `Detector.detect_full()`
   - `detect_tags()` (Legacy)
   - `detect_tags_with_stats()` (Legacy)
5. **Zero-Copy Verification:** Ensure no heap allocations (`Vec::with_capacity`, `to_owned_array`, etc.) occur for image data within the `prepare_image_input` utility.

## Non-Functional Requirements
- **Latency:** Eliminate the implicit 110ms ingestion bottleneck for non-contiguous arrays by shifting the responsibility to the caller.
- **Developer Experience:** Provide an actionable error message directing users to `.ascontiguousarray()`.

## Acceptance Criteria
- [ ] Passing a non-contiguous array (e.g., `img[::2, ::2]`) to `Detector.detect()` results in a `ValueError`.
- [ ] Passing a C-contiguous array works correctly with zero copies.
- [ ] Benchmarking confirms that the Rust wrapper does not allocate for image data.
- [ ] No regressions in detection accuracy for valid contiguous inputs.

## Out of Scope
- Modifying the internal `ImageView` or detection logic in `locus-core`.
- Updating debugging functions (`debug_threshold`, `debug_segmentation`) if the user prefers keeping fallback support there (Targeted Enforcement).
- Automatic re-ordering of strides in Rust (the goal is to *expose* the hit in Python).
