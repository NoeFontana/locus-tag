# Implementation Plan: Sub-pixel Edge Refinement Testing

## Phase 1: Mathematical Verification (erf_approx)
- [ ] Task: Implementation & Verification of erf_approx
    - [ ] Create unit tests for erf_approx (Symmetry, Asymptotic, Zero-crossing).
    - [ ] Implement high-precision integration tests for erf_approx vs ground truth.
    - [ ] Verify maximum absolute error < 1.5e-7.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Mathematical Verification' (Protocol in workflow.md)

## Phase 2: Synthetic Test Harness (Image Generator)
- [ ] Task: Implementation of Sub-pixel Edge Renderer
    - [ ] Build the image generator based on PSF model and erf_approx.
    - [ ] Write meta-tests to verify the generator's output against manual hand-calculated values.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Synthetic Test Harness' (Protocol in workflow.md)

## Phase 3: Edge Recovery & Accuracy Testing
- [ ] Task: Zero-noise Edge Recovery Tests
    - [ ] Implement axis-aligned edge recovery tests (accuracy < 0.001 pixels).
    - [ ] Implement arbitrary-angle edge recovery tests.
    - [ ] Verify scale invariance for varying edge lengths.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Edge Recovery & Accuracy Testing' (Protocol in workflow.md)

## Phase 4: Robustness & Stress Testing
- [ ] Task: Solver Robustness Tests
    - [ ] Implement low-contrast (SNR limit) tests.
    - [ ] Implement high Gaussian noise tests.
    - [ ] Implement image boundary (clipping) tests.
    - [ ] Implement displaced seed (convergence basin) tests.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Robustness & Stress Testing' (Protocol in workflow.md)