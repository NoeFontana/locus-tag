# Plan: Restore Forward Dataset Performance

This plan follows the TDD workflow to reach >75% recall and <0.3px RMSE on the ICRA 2020 `forward` dataset.

## Phase 1: Diagnostics and Baseline
- [x] Task: Establish baseline performance by running `cargo test --test regression_icra2020 test_regression_icra2020_forward --release`. 1ebbb10
- [x] Task: Identify failure modes using `rerun` visualization (e.g., missed quads, decoding failures, poor sub-pixel fit).
- [x] Task: Select 5 "critical failure" images from the `forward` sequence to serve as TDD anchors. (Selected sparsely: 0001, 0012, 0022, 0030, 0040)
- [x] Task: Conductor - User Manual Verification 'Diagnostics and Baseline' (Protocol in workflow.md)

## Phase 2: Recall Recovery (Thresholding & Segmentation)
- [ ] Task: Write failing unit tests in `crates/locus-core/tests/repro_failure.rs` for images where tags are currently missed.
- [ ] Task: Optimize adaptive thresholding parameters (tiling, local stats) to improve initial segmentation.
- [ ] Task: Refine quad candidate filtering to retain more valid candidates in challenging conditions (motion blur, low contrast).
- [ ] Task: Verify that the 5 anchor images now meet recall targets.
- [ ] Task: Conductor - User Manual Verification 'Recall Recovery' (Protocol in workflow.md)

## Phase 3: RMSE Optimization (Sub-pixel Refinement)
- [ ] Task: Write unit tests that assert corner accuracy <0.3px for the anchor images.
- [ ] Task: Improve gradient-based corner refinement logic (e.g., better edge interpolation or weighted fitting).
- [ ] Task: Ensure the homography sampling for bit extraction utilizes the refined corners for higher decoding confidence.
- [ ] Task: Verify that average RMSE on the anchor images falls below 0.3px.
- [ ] Task: Conductor - User Manual Verification 'RMSE Optimization' (Protocol in workflow.md)

## Phase 4: Final Integration and Validation
- [ ] Task: Run the full `forward` regression test and verify: Recall > 75%, RMSE < 0.3px.
- [ ] Task: Update and commit `insta` snapshots for `icra2020_forward_standard`.
- [ ] Task: Perform a sanity check on `rotation`, `random`, and `circle` datasets to ensure no major regressions.
- [ ] Task: Conductor - User Manual Verification 'Final Integration and Validation' (Protocol in workflow.md)