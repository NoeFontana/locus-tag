# Summary of Changes (bench/36h11-640x480 vs main)

This document outlines the significant changes, fixes, and optimizations introduced in this branch compared to the `main` branch.

## 1. Core Geometric Pipeline & Sub-Pixel Precision
* **Coordinate Convention Alignment**: Corrected the coordinate mapping across the pipeline to strictly follow the modern OpenCV and Blender convention where pixel centers are at `(x + 0.5, y + 0.5)`. 
* **Bias Eradication**: Fixed a systematic 0.5-pixel bias in corner detections. The `trace_boundary` algorithm now yields pixel centers directly, and the scaling of decimated coordinates is mathematically exact.
* **Robust Radial Inflation**: Replaced a flawed linear-intersection expansion hack with a robust radial geometric expansion (`inflation_dist = 0.5 * decimation * sqrt(2)`). This ensures the physical outer boundary is correctly mapped without degrading performance on severely distorted lenses.

## 2. Corner Refinement Optimizations
* **Center-Parameterized Gauss-Newton (Erf Mode)**: The intensity-based sub-pixel refinement (`CornerRefinementMode::Erf`) was completely rewritten to parameterize the angle around the *center* of the edge segment rather than the origin. This avoids mathematical ill-conditioning.
* **Dynamic Sampling & Robust Percentiles (Erf Mode)**: The sampling window for the error function optimization now dynamically scales with the edge length. The background (dark) and foreground (light) asymptotes are robustly estimated using the 10th and 90th intensity percentiles.
* **Edge Refinement Mode Improvements**: The fallback gradient peak algorithm (`CornerRefinementMode::Edge`) now correctly handles flat gradient plateaus (caused by bilinear interpolation on sharp step edges) by averaging the start and end of the plateau. It was reverted to an offset-only refinement to preserve high robustness against lens distortion on real-world datasets like ICRA 2020.
* **Default Mode Update**: Changed the default pipeline refinement mode to `Edge` for maximum stability and performance across diverse camera systems.

## 3. Regression & Testing Updates
* **Python Regression Suite**: Added `tests/regression/test_coordinate_alignment.py` to strictly enforce and measure sub-pixel and decimation accuracy.
* **Snapshot Improvements**: Updated Rust `insta` snapshots for `regression_icra2020.rs` and `regression_render_tag.rs`. 
  * The ICRA 2020 dataset showed a drop in RMSE to **0.5269 px** with improved recall.
  * The synthetic `regression_render_tag` dataset saw its Rotation Error drop to **0.9088 degrees** and Reprojection RMSE to **1.8496 px**.
* Added fallback mechanisms for dataset loading and config discovery within the Python benchmarking/sync utilities.

## 4. Code Quality & Tooling
* **Python Static Analysis**: Resolved all `mypy` and `ruff` type-checking and linting errors. Added missing type stubs (`types-jsonschema`, `types-requests`, `types-PyYAML`, `types-setuptools`) to `pyproject.toml`.
* **Rust Bindings (`locus.pyi`)**: Updated the Python stub file to accurately reflect the internal Pyo3 bindings (`PyDetectorConfig`, `PyPose`, etc.) ensuring strict typing for end-users.
* **Rerun SDK Update**: Updated telemetry visualization tools (`tools/cli.py`) to align with the new Rerun v0.30.0 API (`rr.set_time`).
* **Rust Linters**: Applied comprehensive formatting and resolved all `cargo clippy` warnings in the core codebase.