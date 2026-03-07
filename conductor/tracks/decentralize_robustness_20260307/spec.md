# Specification: Decentralized Robustness Testing Architecture

## Overview
Modularize the property-based testing and regression storage to scale with team growth and isolate subsystem failures, mitigating merge conflicts in the monolithic test file.

## Functional Requirements
- **Test Architecture:** Create a dedicated directory `crates/locus-core/tests/robustness/`.
- **Domain-Specific Modules:** Implement test suites mapped to architectural choke points:
  - `pipeline.rs`: End-to-end fuzzing (feeding pure noise into the Detector facade).
  - `geometry.rs`: Homography and PnP solvers (defending against ML-generated degenerate quads).
  - `decoder.rs`: Tag family decoders and bit error correction.
  - `threshold.rs`: Structure of Arrays (SoA) memory bounds and adaptive thresholding limits.
  - Additional 'Blast Zones': `concurrency.rs` (thread safety/race conditions), `memory.rs` (explicit allocation bounds/OOM handling), and `api.rs` (fuzzing Python/C API boundaries).
- **Regression Seed Storage:**
  - Create standard files in `crates/locus-core/proptest-regressions/` that mirror the test directory (e.g., `geometry.txt`).
  - Configure property testing to read/write strictly from these mapped files.
  - Archive all old existing seeds and start fresh with the new structure.
- **CI/CD Quality Gates:**
  - Update `.config/nextest.toml` to exclude the `tests/robustness/` directory from the default local execution.
  - Update `.github/workflows/ci.yml` to run the robustness suite. Initially, it will run as a non-blocking warning (allowing merging during the rollout phase).

## Non-Functional Requirements
- Maintainability: Strict mapping between domains and failure seeds, ensuring developers can immediately identify the subsystem failure.
- Version Control: Enforce strict check-in policy for new regression files to maintain an immutable history of mathematical panics.
- Performance: Isolate computationally heavy property tests from the fast development path.

## Acceptance Criteria
- [ ] Monolithic `robustness_testing.rs` stub is removed.
- [ ] `tests/robustness/` directory exists with the defined domain-specific modules.
- [ ] `proptest-regressions/` is reorganized and correctly mapped.
- [ ] Old regression seeds are archived.
- [ ] `nextest.toml` is configured to skip the robustness suite by default.
- [ ] `ci.yml` includes the robustness suite as a non-blocking warning.