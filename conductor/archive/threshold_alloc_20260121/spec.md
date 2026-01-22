# Specification: Elimination of Hot-Path Allocations (Thresholding)

## 1. Overview
The current adaptive thresholding implementation in `threshold.rs` allocates temporary heap buffers (`row_thresholds`, `row_valid`) for every tile row within the parallel iterator. For high-resolution images, this causes thousands of allocation/deallocation cycles per frame, leading to allocator thrashing and synchronization overhead. This track aims to eliminate these allocations by reusing buffers via Rayon's `for_each_init`.

## 2. Goals
-   **Eliminate Hot-Path Allocations:** Refactor the parallel loop to allocate temporary buffers once per thread instead of once per task.
-   **Maintain Correctness:** Ensure the thresholding output remains bit-exact or functionally equivalent (Recall/RMSE must not degrade).
-   **Improve Performance:** Reduce allocator pressure, aiming for improved latency and throughput, especially on high-core-count systems.

## 3. Functional Requirements
-   **Refactor `apply_threshold`:**
    -   Replace existing `output.par_chunks_mut(...).enumerate().for_each(...)` with `for_each_init` (or `map_init` if appropriate).
    -   Initialize auxiliary buffers (`row_thresholds`, `row_valid`) once per thread context.
    -   Ensure buffers are correctly sized and reset/reused for each tile row iteration.
-   **HPC Best Practices:**
    -   Minimize initialization overhead (avoid zeroing if data is fully overwritten).
    -   Maintain cache locality.

## 4. Verification & Acceptance Criteria
-   **Regression Testing:**
    -   Run `tests/regression_icra2020.rs` before and after changes.
    -   **Success:** No decrease in Mean Recall or increase in Mean RMSE.
    -   **Success:** Latency statistics (Total MS) should show improvement or neutrality (no regressions).
-   **Code Quality:**
    -   No `unsafe` blocks added unless strictly necessary and documented.
    -   Clean, idiomatic Rust/Rayon usage.

## 5. Out of Scope
-   Changing the core thresholding algorithm logic (Otsu/min-max modifications).
-   Modifying other pipeline stages (Segmentation, Quad Extraction).
