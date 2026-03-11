# Implementation Plan: Sub-pixel Edge Refinement Testing

## Phase 1: Mathematical Verification (erf_approx)
- [x] Task: Implementation & Verification of erf_approx (064f829)
    - [x] Create unit tests for erf_approx (Symmetry, Asymptotic, Zero-crossing).
    - [x] Implement high-precision integration tests for erf_approx vs ground truth.
    - [x] Verify maximum absolute error < 1.5e-7.
- [~] Task: Conductor - User Manual Verification 'Phase 1: Mathematical Verification' (Protocol in workflow.md)

## Phase 2: Synthetic Test Harness (Image Generator)
- [ ] Task: Implementation of Sub-pixel Edge Renderer
    - [ ] Build the image generator (sampling at i+0.5, j+0.5).
    - [ ] Implement the ERF model I(d) = (A+B)/2 + (B-A)/2 * erf(d / (sigma * sqrt(2))).
    - [ ] Write meta-tests to verify the generator's output against manual hand-calculated values.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Synthetic Test Harness' (Protocol in workflow.md)

## Phase 3: Edge Recovery & Accuracy Testing
- [ ] Task: Zero-noise Edge Recovery Tests
    - [ ] Implement axis-aligned edge recovery tests using (x_dec - 0.5) * K + 0.5 scaling.
    - [ ] Implement arbitrary-angle edge recovery tests (accuracy < 0.001 pixels).
    - [ ] Verify scale invariance for varying edge lengths.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Edge Recovery & Accuracy Testing' (Protocol in workflow.md)

## Phase 4: Robustness & Stress Testing
- [ ] Task: Solver Robustness Tests
    - [ ] Implement low-contrast (SNR limit) tests.
    - [ ] Implement high Gaussian noise tests.
    - [ ] Implement image boundary (clipping) tests.
    - [ ] Implement displaced seed (convergence basin) tests.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Robustness & Stress Testing' (Protocol in workflow.md)