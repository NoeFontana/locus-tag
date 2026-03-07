# Implementation Plan: Decentralized Robustness Testing Architecture

## Phase 1: Setup and Architecture [checkpoint: 1cc875c]
- [x] Task: Establish directories and remove monolith 190daa0
    - [ ] Remove `crates/locus-core/tests/robustness_testing.rs` stub.
    - [ ] Create `crates/locus-core/tests/robustness/` directory.
    - [ ] Archive old regression seeds in `crates/locus-core/proptest-regressions/`.
- [x] Task: Conductor - User Manual Verification 'Setup and Architecture' (Protocol in workflow.md) 1cc875c

## Phase 2: Domain-Specific Modules (Frontend Zone) [checkpoint: 60e6ed0]
- [x] Task: Implement `pipeline.rs` tests 505ae48
    - [ ] Write failing unit tests for end-to-end fuzzing (feeding pure noise).
    - [ ] Map property testing regressions to `proptest-regressions/pipeline.txt`.
    - [ ] Implement robust handling to make tests pass.
- [x] Task: Implement `threshold.rs` tests 66942c5
    - [ ] Write failing unit tests for SoA memory bounds and adaptive threshold limits.
    - [ ] Map property testing regressions to `proptest-regressions/threshold.txt`.
    - [ ] Implement robust handling to make tests pass.
- [x] Task: Conductor - User Manual Verification 'Domain-Specific Modules (Frontend Zone)' (Protocol in workflow.md) 60e6ed0

## Phase 3: Domain-Specific Modules (Backend Zone) [checkpoint: 8ddeea3]
- [x] Task: Implement `geometry.rs` tests acdff88
    - [ ] Write failing unit tests for Homography and PnP solvers (degenerate quads).
    - [ ] Map property testing regressions to `proptest-regressions/geometry.txt`.
    - [ ] Implement robust handling to make tests pass.
- [x] Task: Implement `decoder.rs` tests c204928
    - [ ] Write failing unit tests for tag family decoders and bit error correction.
    - [ ] Map property testing regressions to `proptest-regressions/decoder.txt`.
    - [ ] Implement robust handling to make tests pass.
- [x] Task: Conductor - User Manual Verification 'Domain-Specific Modules (Backend Zone)' (Protocol in workflow.md) 8ddeea3

## Phase 4: Additional Blast Zones
- [~] Task: Implement `concurrency.rs` tests
    - [ ] Write failing unit tests for thread safety and race conditions.
    - [ ] Map property testing regressions to `proptest-regressions/concurrency.txt`.
    - [ ] Implement robust handling to make tests pass.
- [ ] Task: Implement `memory.rs` tests
    - [ ] Write failing unit tests for memory bounds and OOM handling.
    - [ ] Map property testing regressions to `proptest-regressions/memory.txt`.
    - [ ] Implement robust handling to make tests pass.
- [ ] Task: Implement `api.rs` tests
    - [ ] Write failing unit tests for API fuzzing.
    - [ ] Map property testing regressions to `proptest-regressions/api.txt`.
    - [ ] Implement robust handling to make tests pass.
- [ ] Task: Conductor - User Manual Verification 'Additional Blast Zones' (Protocol in workflow.md)

## Phase 5: CI/CD Quality Gates
- [ ] Task: Configure `.config/nextest.toml`
    - [ ] Write test to verify nextest behavior (if applicable).
    - [ ] Exclude `tests/robustness/` directory from default execution.
- [ ] Task: Update `.github/workflows/ci.yml`
    - [ ] Implement non-blocking warning check for robustness suite in CI pipeline.
- [ ] Task: Conductor - User Manual Verification 'CI/CD Quality Gates' (Protocol in workflow.md)