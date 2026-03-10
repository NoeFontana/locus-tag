# Implementation Plan: OpenCV Parity Migration

## Phase 1: Consolidate Python Data Generation Layer
- [x] Task: Expand `examples/dictionary_generation/extract_opencv.py` with AprilTag 36h11, 41h12.
- [x] Task: Deprecate `extract_umich.py` and standalone ArUco extractors.
- [x] Task: Regenerate all IR JSON files in `data/dictionaries/` using updated `extract_opencv.py`.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Consolidate Python Data Generation Layer' (Protocol in workflow.md)

## Phase 2: Update Rust Core Code Generation
- [x] Task: Simplify `crates/locus-core/templates/dictionaries.rs.j2` (remove coordinate inversions).
- [x] Task: Regenerate `crates/locus-core/src/dictionaries.rs` using `update_generator.py`.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Update Rust Core Code Generation' (Protocol in workflow.md)

## Phase 3: Realign Testing Suite
- [x] Task: Rewrite dictionary parity tests to use `cv2.aruco` as oracle.
- [x] Task: Run test suite and accept new baselines for snapshot tests using `cargo insta accept`.
- [x] Task: Verify pose estimation outputs match OpenCV (`benches/pose_bench.rs`).
- [x] Task: Conductor - User Manual Verification 'Phase 3: Realign Testing Suite' (Protocol in workflow.md)

## Phase 4: Documentation & Cleanup
- [x] Task: Update `docs/explanation/coordinates.md` with OpenCV alignment.
- [x] Task: Add migration warnings to documentation.
- [x] Task: Final project-wide lint and format.
- [x] Task: Conductor - User Manual Verification 'Phase 4: Documentation & Cleanup' (Protocol in workflow.md)

## Phase: Review Fixes
- [x] Task: Apply review suggestions 346412a
