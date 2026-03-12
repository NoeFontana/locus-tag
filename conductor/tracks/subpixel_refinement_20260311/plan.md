# Implementation Plan: Sub-pixel Edge Refinement Testing

## Phase 1: Mathematical Verification (erf_approx) [checkpoint: 064f829]
- [x] Task: Implementation & Verification of erf_approx (064f829)
    - [x] Create unit tests for erf_approx (Symmetry, Asymptotic, Zero-crossing).
    - [x] Implement high-precision integration tests for erf_approx vs ground truth.
    - [x] Verify maximum absolute error < 1.5e-7.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Mathematical Verification' (Protocol in workflow.md)

## Phase 2: Synthetic Test Harness (Image Generator) [checkpoint: ad89ccc]
- [x] Task: Implementation of Sub-pixel Edge Renderer (e3b2b95)
    - [x] Build the image generator (sampling at i+0.5, j+0.5).
    - [x] Implement the ERF model I(d) = (A+B)/2 + (B-A)/2 * erf(d / (sigma * sqrt(2))).
    - [x] Write meta-tests to verify the generator's output against manual hand-calculated values.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Synthetic Test Harness' (Protocol in workflow.md) (ad89ccc)

## Phase 3: Edge Recovery & Accuracy Testing (b13b147) [checkpoint: ccd6428]
- [x] Task: Zero-noise Edge Recovery Tests (b13b147)
    - [x] Implement axis-aligned edge recovery tests using (x_dec - 0.5) * K + 0.5 scaling.
    - [x] Implement arbitrary-angle edge recovery tests (accuracy < 0.001 pixels).
    - [x] Verify scale invariance for varying edge lengths.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Edge Recovery & Accuracy Testing' (Protocol in workflow.md) (ccd6428)

## Phase 4: Robustness & Stress Testing (cc5d793)
- [x] Task: Solver Robustness Tests (cc5d793)
    - [x] Implement low-contrast (SNR limit) tests.
    - [x] Implement high Gaussian noise tests.
    - [x] Implement image boundary (clipping) tests.
    - [x] Implement displaced seed (convergence basin) tests.
- [~] Task: Conductor - User Manual Verification 'Phase 4: Robustness & Stress Testing' (Protocol in workflow.md)