# Implementation Plan: Zero-Cost Tracing

## Phase 1: Infrastructure & Configuration
- [ ] Task: Configure `tracing` compile-time erasure in `crates/locus-core/Cargo.toml`.
    - Add `features = ["max_level_error"]` (or `off`) to the `tracing` dependency for production.
- [ ] Task: Verify `tracy` feature correctly gates `tracing-tracy` and its subscriber initialization.
- [ ] Task: Conductor - User Manual Verification 'Infrastructure & Configuration' (Protocol in workflow.md)

## Phase 2: Core Pipeline Instrumentation
- [ ] Task: Instrument `thresholding` stage in `crates/locus-core/src/threshold.rs`.
    - Use `#[tracing::instrument(skip_all, name = "pipeline::thresholding")]`.
- [ ] Task: Instrument `segmentation` stage in `crates/locus-core/src/segmentation.rs`.
    - Use `#[tracing::instrument(skip_all, name = "pipeline::segmentation")]`.
- [ ] Task: Instrument `quad_extraction` stage in `crates/locus-core/src/quad.rs`.
    - Use `#[tracing::instrument(skip_all, name = "pipeline::quad_extraction")]`.
- [ ] Task: Instrument `homography_pass` stage in `crates/locus-core/src/decoder.rs`.
    - Use `#[tracing::instrument(skip_all, name = "pipeline::homography_pass")]`.
- [ ] Task: Instrument `decoding_pass` stage in `crates/locus-core/src/decoder.rs`.
    - Use `#[tracing::instrument(skip_all, name = "pipeline::decoding_pass")]`.
- [ ] Task: Instrument `pose_refinement` stage in `crates/locus-core/src/pose.rs`.
    - Use `#[tracing::instrument(skip_all, name = "pipeline::pose_refinement")]`.
- [ ] Task: Conductor - User Manual Verification 'Core Pipeline Instrumentation' (Protocol in workflow.md)

## Phase 3: Hot-Path Optimization & Validation
- [ ] Task: Audit and convert/remove dynamic logs (`info!`, `debug!`) in instrumented stages.
    - Ensure no `format!` or string manipulation remains in the hot loop.
- [ ] Task: Verify zero-cost erasure by inspecting assembly or comparing benchmark results.
- [ ] Task: Confirm Tracy visual profiling works as expected with `--features tracy`.
- [ ] Task: Conductor - User Manual Verification 'Hot-Path Optimization & Validation' (Protocol in workflow.md)
