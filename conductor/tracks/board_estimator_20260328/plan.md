# Implementation Plan: Board Estimator (ChAruco/AprilGrid)

## Phase 1: Data-Oriented Modeling & Context Contract
- [ ] Task: Implement `BoardConfig`
    - [ ] Write tests for `BoardConfig` struct initialization and coordinate computation.
    - [ ] Implement `BoardConfig` (static, immutable) for canonical 3D coordinates.
- [ ] Task: Implement Workspace Arena Borrow Pattern
    - [ ] Write tests for `WORKSPACE_ARENA` allocation and `BoardEstimator::estimate` API interface.
    - [ ] Implement thread-local `WORKSPACE_ARENA` and `BoardEstimator::estimate` method signature taking a read-only `DetectionBatch`.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Data-Oriented Modeling & Context Contract' (Protocol in workflow.md)

## Phase 2: Classification Engine (LO-RANSAC)
- [ ] Task: Minimal Sample Generator (IPPE)
    - [ ] Write tests for minimal planar pose generation from 4 decoded tags using existing IPPE.
    - [ ] Implement logic to extract 16 corners and fetch corresponding 3D coordinates to solve pose.
- [ ] Task: SIMD Consensus Evaluation
    - [ ] Write tests for projecting 3D board corners and computing $L_2$ reprojection errors.
    - [ ] Implement SIMD consensus (AVX2, AVX-512, ARM NEON) applying relaxed geometric threshold ($\tau \approx 2.0$ pixels).
- [ ] Task: Local Optimization (LO) Handoff
    - [ ] Write tests for fast, unweighted Gauss-Newton step and early termination.
    - [ ] Implement LO handoff to settle pose and finalize the boolean mask, reusing existing optimization routines where appropriate.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Classification Engine (LO-RANSAC)' (Protocol in workflow.md)

## Phase 3: Metrology Engine (AW-LM)
- [ ] Task: Covariance Injection & Sparse Jacobian Stacking
    - [ ] Write tests for reading covariance $\Sigma_i$ and stacking the $2M \times 6$ Jacobian matrix.
    - [ ] Implement covariance injection from `DetectionBatch` and compute analytical derivatives.
- [ ] Task: Huber-Damped Minimization
    - [ ] Write tests for the Huber loss function and iterative solver.
    - [ ] Implement Huber-damped minimization normal equations solver and pose updates, reusing existing optimization routines.
- [ ] Task: OpenCV Regression Benchmark
    - [ ] Implement rigorous OpenCV regression benchmark within the Rust test suite to validate accuracy against OpenCV.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Metrology Engine (AW-LM)' (Protocol in workflow.md)