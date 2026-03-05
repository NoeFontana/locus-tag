# Implementation Plan: Performance Alignment and API Expansion

## Phase 1: Research and Audit
Identify discrepancies in configuration and verify coordinate mapping.

- - [x] Task: Compare `DetectorConfig::default()` in `crates/locus-core/src/config.rs` with historical main branch values.
- - [x] Task: Audit `crates/locus-core/src/detector.rs` detect loop for any missing `+ 0.5` offsets or incorrect scaling in the decimation path.
- [ ] Task: Run current benchmark with manual parameter overrides to isolate the regression cause.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Research and Audit' (Protocol in workflow.md)

## Phase 2: Python API Expansion
Expose all relevant pipeline parameters through the Python bridge.

- -  - [x] Task: Update `locus-py/src/lib.rs` to accept additional keyword arguments in `Detector::new`.
-  - [x] Task: Update `locus-py/locus/__init__.py` to pass these arguments from the Python constructor.
-  - [x] Task: Verify that parameters like `threshold_tile_size` and `quad_min_fill_ratio` are correctly propagated to the Rust core.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Python API Expansion' (Protocol in workflow.md)

## Phase 3: Final Verification and Tuning
Restore baseline performance and update documentation.

- - [x] Task: Set optimal defaults in `DetectorConfig` to match pre-refactor performance.
- - [x] Task: Run full ICRA benchmark suite using the unified CLI and verify metrics.
- - [x] Task: Update `README.md` performance table with the latest verified results.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Final Verification and Tuning' (Protocol in workflow.md)
