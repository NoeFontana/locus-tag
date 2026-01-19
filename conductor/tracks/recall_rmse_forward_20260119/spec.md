# Specification: Restore Forward Dataset Performance

## Overview
This track aims to restore and optimize the performance of the Locus detector on the ICRA 2020 `forward` dataset. Recent changes or current defaults have resulted in performance below the desired targets for robotics applications.

## Goals
- **Recall:** Achieve $>75\%$ recall on the ICRA 2020 `forward` benchmark.
- **Accuracy:** Achieve an average RMSE (Root Mean Square Error) of $<0.3$ pixels for detected tag corners.
- **Verification:** Use the existing integration test in `crates/locus-core/tests/regression_icra2020.rs`.

## Functional Requirements
- **Pipeline Optimization:** Investigate and refine the following components to meet targets:
    - **Thresholding:** Optimize tiling and local min/max statistics.
    - **Segmentation:** Improve edge detection and quad candidate filtering.
    - **Quad Fitting:** Enhance sub-pixel refinement for better corner precision.
    - **Decoding:** Ensure robust bit extraction via optimized homography sampling.
- **Configuration Defaults:** Identify and apply optimal default parameters (or sequence-specific overrides if applicable) for the `forward` sequence.

## Non-Functional Requirements
- **Balanced Performance:** Maintain a balance between recall and accuracy; do not sacrifice one significantly to improve the other.
- **No Regression (Targeted):** While the focus is on the `forward` dataset, ensure the changes remain idiomatic and maintainable within the `locus-core` architecture.
- **Performance Invariants:** Ensure modifications do not violate the "Zero-Allocation Hot Loop" principle defined in the project guidelines.

## Acceptance Criteria
- Running `cargo test --test regression_icra2020 test_regression_icra2020_forward --release` results in:
    - `mean_recall > 0.75`
    - `mean_rmse < 0.3`
- All other tests in the suite pass.
- `insta` snapshots for the `forward` dataset are updated and committed.

## Out of Scope
- Optimization of other ICRA 2020 datasets (`rotation`, `random`, `circle`) beyond ensuring they don't catastrophically regress.
- Major architectural changes to the `TagDecoder` trait or `Detector` interface.
