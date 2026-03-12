# Plan: Fix Subpixel Refinement Regression

## Objective
Identify and fix the regression introduced in commit `5b26ae05c9a0c893997ee216f8fd7cabbd2a8615` that degraded recall and RMSE.

## Background & Motivation
The suspect commit modified the mapping of decimated boundary coordinates to full-resolution coordinates. It removed the `+ 0.5` pixel center offset in `trace_boundary` and the `0.5px` outward expansion in `extract_single_quad`. 

Because `trace_boundary` now returns the top-left corner of boundary pixels (`curr_x as f64`), the right and bottom edges of the unrefined quad are shifted inward by 1 decimated pixel. When scaled by decimation factor `d`, the initial line for the subpixel solver is off by `d` full pixels. This misaligns the `dist > 0.0` check used to initialize the `light_sum` and `dark_sum`, causing the Gauss-Newton optimizer to fail or degrade in accuracy.

## Proposed Solution
We will systematically test restoring the subpixel coordinate mapping while retaining the improved Gauss-Newton solver and outward normal conventions.

### Implementation Steps

1.  **Restore `trace_boundary` Center Mapping:**
    *   In `crates/locus-core/src/quad.rs`, restore the `+ 0.5` offset in `trace_boundary` so that points represent pixel centers, avoiding the top-left bias.
2.  **Restore Quad Expansion:**
    *   In `extract_single_quad`, restore the code that expands the initial quad by `0.5px` outward to correctly bound the object.
3.  **Adjust `refine_edge_intensity` if needed:**
    *   Ensure the initial distance `d` calculation is robust to the restored coordinate system.
4.  **Run Benchmarks:**
    *   Since I cannot run commands in Plan mode, I will request the user to run the tests or exit Plan mode to perform the validation.

## Verification & Testing
- Run `cargo test --release --test regression_icra2020 regression_icra_forward` to check if recall and RMSE are restored to baseline levels.
- If successful, run the full test suite to ensure no other regressions.