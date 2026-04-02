# Implementation Plan: charuco_support_20260401

## Phase 1: Core Mathematical Foundation (Subpixel Refinement) [checkpoint: 5a654a6]
- [x] **Task: Implement Bivariate Polynomial Surface Fitting** 56ede68
    - [x] Write failing unit tests for `BivariatePolynomial` fitting (Rust).
    - [x] Implement analytical 2nd-order Taylor expansion solver ($H^{-1}\mathbf{J}$).
    - [x] Verify subpixel accuracy against synthetic saddle point images.
- [x] **Task: Refinement API Integration** da72bf1
    - [x] Define `SubpixelRefinement` trait for future extensibility (e.g., Template Matching).
    - [x] Implement `PolynomialRefiner` with configurable search window ($N \times N$).
- [x] **Task: Conductor - User Manual Verification 'Core Mathematical Foundation' (Protocol in workflow.md)** 5a654a6

## Phase 2: ChArUco Board Logic & Pipeline Decoupling [checkpoint: 9862113]
- [x] **Task: Decouple Detection API** efc8797
    - [x] Refactor detection entry points to include `detect_aprilgrid` and `detect_charuco`.
    - [x] Define `CharucoBoard` configuration (tag IDs, checkerboard layout).
- [x] **Task: Coarse Detection & Corner Prediction** 67cbbd6
    - [x] Implement coarse tag detection stage and rough board homography.
    - [x] Implement geometric projection of checkerboard intersections from tag corners.
- [x] **Task: Conductor - User Manual Verification 'ChArUco Board Logic & Pipeline Decoupling' (Protocol in workflow.md)** 9862113

## Phase 3: Integration & Pose Estimation
- [ ] **Task: ChArUco PnP Solver Integration**
    - [ ] Map 2D refined corners to 3D board coordinates.
    - [ ] Integrate with existing `solvePnP` solver.
- [ ] **Task: Benchmarking & Regression**
    - [ ] Re-run pose estimation benchmarks to verify improvement.
    - [ ] Run full regression suite to ensure zero impact on AprilGrid detection.
- [ ] **Task: Conductor - User Manual Verification 'Integration & Pose Estimation' (Protocol in workflow.md)**

## Phase 4: Finalization & Documentation
- [ ] **Task: Documentation Update**
    - [ ] Update `docs/explanation/pipeline.md` with ChArUco-specific details.
    - [ ] Update API reference for new detection methods.
- [ ] **Task: Conductor - User Manual Verification 'Finalization & Documentation' (Protocol in workflow.md)**
