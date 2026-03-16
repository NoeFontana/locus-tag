# Specification: Gradient-Weighted Line Fitting (GWLF) Refinement

**Track ID:** `gradient_weighted_lines_20260316`
**Type:** Feature
**Status:** New

## 1. Overview
Implement Gradient-Weighted Line Fitting (GWLF) as a high-precision, sub-pixel corner refinement alternative to the existing Edge Response Function (ERF). This track focuses on formal error propagation from image gradients through geometric line intersection to Cartesian corner uncertainty.

## 2. Functional Requirements
- **Core Math:**
    - Perform gradient-weighted PCA on pixels along each of the four quad edges to determine line centroids $\bar{\mathbf{x}}$ and normals $\mathbf{n}$.
    - Compute $3 \times 3$ homogeneous line covariance $\Sigma_l$ via first-order error propagation.
    - Compute Cartesian corner covariance $\Sigma_c$ via projective line intersection and perspective division.
- **Solver Integration:**
    - Invert $\Sigma_c$ (with minimal Tikhonov regularization) to produce the precision matrix $W_i$.
    - Inject $W_i$ into the Weighted Levenberg-Marquardt solver in `pose_weighted.rs`.
- **API/Config:**
    - Add `CornerRefinementMode::Gwlf` to the detector configuration.
- **Visualization:**
    - Implement Rerun logging for:
        - **Error Ellipses:** 2D visualization of $\Sigma_c$.
        - **Refined Lines:** Infinite lines representing the four edges.
        - **Gradient Weights:** Heatmap or point-cloud of the weights used in PCA.

## 3. Performance & Quality Requirements
- **Accuracy:** Rotation P50 error < 0.1° on Hub (Synthetic) 480p dataset.
- **Efficiency:** Latency increase < 0.2ms per quad compared to current "Accurate" mode.
- **Precision:** 100% detection precision (zero false positives) on verified datasets.
- **Reliability:** The solver must handle nearly parallel lines via gain-scheduled regularization.

## 4. Acceptance Criteria
- [ ] GWLF mode is selectable via `CornerRefinementMode`.
- [ ] Cartesian covariance matrices are verified against numerical finite differences (where feasible).
- [ ] Benchmarks on Hub datasets show a significant improvement in rotation stability.
- [ ] Rerun visualizations correctly show the highly anisotropic nature of the refined corners.

## 5. Out of Scope
- Support for non-planar tags or alternative marker families (ArUco/STag) in this iteration.
- Optimization of the thresholding or segmentation stages.
- Changes to the "Fast" (IPPE) pose estimation mode.
