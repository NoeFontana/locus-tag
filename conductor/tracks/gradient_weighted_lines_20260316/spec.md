# Specification: Gradient-Weighted Line Fitting (GWLF)

## Overview
Implement Gradient-Weighted Line Fitting (GWLF) as an optional alternative to the existing Edge Response Function (ERF) for corner refinement. This method uses weighted orthogonal distance regression (PCA on gradients) along the four edges of a coarse quad to find high-precision line intersections.

## Functional Requirements
1. **Edge Normal & Rasterization:** Implement a stepping vector (DDA/Bresenham) along the 4 edges of the coarse quad.
2. **Moment Accumulation:** Perform single-pass accumulation of `sum_w`, `sum_wx`, `sum_wy`, `sum_wxx`, `sum_wyy`, `sum_wxy` using squared gradient magnitude as weights.
3. **Analytic Eigendecomposition:** Solve for the smallest eigenvalue and its corresponding eigenvector of the 2x2 covariance matrix analytically.
4. **Homogeneous Intersection:** Compute the cross-product of adjacent homogeneous lines to find refined corner coordinates.
5. **Sanity Gate & Fallback:** Implement a 3.0-pixel Euclidean distance check; fall back to coarse corners if exceeded.
6. **Configuration Toggle:** Allow users to choose between ERF and GWLF via the API/CLI.

## Non-Functional Requirements
- **Target Accuracy:** Achieve a Reprojection RMSE of < 0.8px.
- **Latency Budget:** Maximum overhead of 1.0ms per quad.
- **Performance:** Implementation must be SIMD-friendly and avoid heap allocations during the accumulation phase.

## Acceptance Criteria
- [ ] RMSE on standard datasets (e.g., ICRA 2020) is consistently < 0.8px when GWLF is enabled.
- [ ] Latency per quad is within the 1.0ms budget on benchmark hardware.
- [ ] Telemetry successfully reports Fallback Frequency and Refinement Delta.
- [ ] The algorithm is selectable via a new configuration parameter.

## Out of Scope
- Full SVD or iterative eigenvalue solvers.
- Multi-scale or pyramid-based refinement.
- Replacing ERF entirely (GWLF will co-exist).
