# Track Specification: Board Estimator (ChAruco/AprilGrid)

## Overview
Implement a pure-Rust, high-performance board detector for `locus-tag`. The goal is to integrate a ChAruco/AprilGrid detector into the pipeline, prioritizing theoretical purity and latency budgets, culminating in a rigorous OpenCV regression benchmark. Initially, the implementation will focus on ChAruco boards while establishing an interface that easily accommodates AprilGrid later.

## Functional Requirements
- **Data-Oriented Modeling (Phase 1):**
  - Implement a `BoardConfig` context (static, immutable) to store canonical 3D coordinates.
  - Implement a `WORKSPACE_ARENA` borrow pattern for the `BoardEstimator::estimate` method to avoid hot-path heap allocations.
- **Classification Engine (Phase 2 - LO-RANSAC):**
  - Implement Infinitesimal Plane-Based Pose Estimation (IPPE) for minimal planar pose generation from 4 decoded tags (16 corners).
  - Implement SIMD consensus evaluation supporting AVX2, AVX-512, and ARM NEON.
  - Apply a relaxed geometric threshold ($\tau \approx 2.0$ pixels).
  - Implement Local Optimization (LO) handoff using a fast, unweighted Gauss-Newton step.
- **Metrology Engine (Phase 3 - AW-LM):**
  - Implement covariance injection by reading the $2 \times 2$ matrix $\Sigma_i$ from `DetectionBatch`.
  - Stack the sparse $2M \times 6$ Jacobian matrix analytically.
  - Implement Huber-damped minimization iteratively solving $(J^T W J + \lambda I) \Delta \xi = -J^T W r$.
- **Code Reuse:** Ensure we reuse already existing optimization routines if they match the requirements of the board pose optimizers.

## Non-Functional Requirements
- **Performance:** Prioritize accuracy first in this phase, although the architecture must map cleanly to CPU cache lines and avoid hot-path heap allocations.
- **Compatibility:** SIMD targets must include AVX2, AVX-512, and ARM NEON.
- **Extensibility:** The API boundary must be rigidly defined so individual contributors can optimize parallel tracks without merge conflicts.

## Acceptance Criteria
- [ ] `BoardConfig` struct and `WORKSPACE_ARENA` are implemented and functional.
- [ ] LO-RANSAC classification separates true observations from aliases using IPPE and SIMD consensus.
- [ ] AW-LM metrology engine refines the pose using covariance-weighted, Huber-damped optimization.
- [ ] A rigorous OpenCV regression benchmark is implemented within the Rust test suite to validate accuracy.

## Out of Scope
- Initial implementation will strictly focus on ChAruco (AprilGrid is deferred to a future iteration, though the API will support it).
- Sub-millisecond latency budgets are not strictly enforced in this phase, provided accuracy is achieved.