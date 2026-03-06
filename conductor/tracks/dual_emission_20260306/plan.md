# Implementation Plan: Dual-Emission Observability

## Phase 1: Setup Telemetry Dependencies
- [ ] Task: Update `dev-dependencies` in `crates/locus-core/Cargo.toml`
    - [ ] Add `tracing-appender` crate.
    - [ ] Update `tracing-subscriber` to include `json` and `env-filter` features.
- [ ] Task: Conductor - User Manual Verification 'Setup Telemetry Dependencies' (Protocol in workflow.md)

## Phase 2: Architect Dual-Layer Test Harness
- [ ] Task: Create Test Harness `tests/common/telemetry.rs`
    - [ ] Write failing test `tests/test_telemetry.rs` to verify log file creation on trace.
    - [ ] Implement `init()` function configuring a non-blocking JSON writer to `target/profiling/regression_events.json` (overwrite mode).
    - [ ] Build a `tracing_subscriber::Registry` with Layer 1 (JSON) and Layer 2 (TracyLayer via `#[cfg(feature = "tracy")]`).
    - [ ] Register as global default subscriber.
    - [ ] Ensure failing test passes.
- [ ] Task: Conductor - User Manual Verification 'Architect Dual-Layer Test Harness' (Protocol in workflow.md)

## Phase 3: Instrument Core Pipeline
- [ ] Task: Add Zero-Cost Instrumentation to Pipeline Stages
    - [ ] Instrument Thresholding functions (`adaptive_threshold_with_map` and related) in `crates/locus-core/src/`.
    - [ ] Instrument Segmentation functions (`extract_quads_soa` and related).
    - [ ] Instrument Decoding functions (`decode_batch_soa` and related).
    - [ ] Instrument Pose Estimation functions.
- [ ] Task: Verify Zero-Cost Compilation
    - [ ] Run `cargo check` and `cargo test` to ensure macros compile correctly without errors.
- [ ] Task: Conductor - User Manual Verification 'Instrument Core Pipeline' (Protocol in workflow.md)

## Phase 4: Wire Regression Tests
- [ ] Task: Integrate Telemetry into Integration Tests
    - [ ] Wire `common::telemetry::init()` at the start of tests in `tests/regression_icra2020.rs`.
    - [ ] Ensure the asynchronous writer `_guard` is held until the end of each test scope.
- [ ] Task: Integrate Telemetry into Render Tag Tests
    - [ ] Wire `common::telemetry::init()` at the start of tests in `tests/regression_render_tag.rs`.
    - [ ] Ensure the asynchronous writer `_guard` is held until the end of each test scope.
- [ ] Task: Execute Test Suite and Validate Output
    - [ ] Run `cargo test --test regression_icra2020` and verify `target/profiling/regression_events.json` is populated with tracing spans.
- [ ] Task: Conductor - User Manual Verification 'Wire Regression Tests' (Protocol in workflow.md)

## Phase 5: Automated Workflow Support
- [ ] Task: Verify Dual-Emission Output
    - [ ] Run a test with `cargo integration-test --features tracy` (or equivalent) while `tracy-capture` is running, to ensure both `regression.tracy` and JSON are correctly generated.
- [ ] Task: Conductor - User Manual Verification 'Automated Workflow Support' (Protocol in workflow.md)