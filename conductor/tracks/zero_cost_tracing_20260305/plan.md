# Implementation Plan: Zero-Cost Tracing

## Phase 1: Infrastructure & Configuration [checkpoint: f9daba8]
- [x] Task: Configure `tracing` compile-time erasure in `crates/locus-core/Cargo.toml`. 460270c
    - Add `features = ["max_level_error"]` (or `off`) to the `tracing` dependency for production.
- [x] Task: Verify `tracy` feature correctly gates `tracing-tracy` and its subscriber initialization.
- [x] Task: Conductor - User Manual Verification 'Infrastructure & Configuration' (Protocol in workflow.md)

## Phase 2: Core Pipeline Instrumentation [checkpoint: 0ee8ea6]
- [x] Task: Instrument `thresholding` stage in `crates/locus-core/src/threshold.rs`. 8b4114a
    - Use `#[tracing::instrument(skip_all, name = "pipeline::thresholding")]`.
- [x] Task: Instrument `segmentation` stage in `crates/locus-core/src/segmentation.rs`. 8b4114a
    - Use `#[tracing::instrument(skip_all, name = "pipeline::segmentation")]`.
- [x] Task: Instrument `quad_extraction` stage in `crates/locus-core/src/quad.rs`. 8b4114a
    - Use `#[tracing::instrument(skip_all, name = "pipeline::quad_extraction")]`.
- [x] Task: Instrument `homography_pass` stage in `crates/locus-core/src/decoder.rs`. 8b4114a
    - Use `#[tracing::instrument(skip_all, name = "pipeline::homography_pass")]`.
- [x] Task: Instrument `decoding_pass` stage in `crates/locus-core/src/decoder.rs`. 8b4114a
    - Use `#[tracing::instrument(skip_all, name = "pipeline::decoding_pass")]`.
- [x] Task: Instrument `pose_refinement` stage in `crates/locus-core/src/pose.rs`. 8b4114a
    - Use `#[tracing::instrument(skip_all, name = "pipeline::pose_refinement")]`.
- [x] Task: Conductor - User Manual Verification 'Core Pipeline Instrumentation' (Protocol in workflow.md)

## Phase 3: Hot-Path Optimization & Validation
- [~] Task: Audit and convert/remove dynamic logs (`info!`, `debug!`) in instrumented stages.
    - Ensure no `format!` or string manipulation remains in the hot loop.
- [ ] Task: Verify zero-cost erasure by inspecting assembly or comparing benchmark results.
- [ ] Task: Confirm Tracy visual profiling works as expected with `--features tracy`.
- [ ] Task: Conductor - User Manual Verification 'Hot-Path Optimization & Validation' (Protocol in workflow.md)
