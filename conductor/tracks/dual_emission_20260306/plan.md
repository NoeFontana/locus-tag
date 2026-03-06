# Implementation Plan: Dual-Emission Observability

## Phase 1: Setup Telemetry Dependencies [checkpoint: 06e7850]
- [x] Task: Update `dev-dependencies` in `crates/locus-core/Cargo.toml` 6a4d838
    - [x] Add `tracing-appender` crate.
    - [x] Update `tracing-subscriber` to include `json` and `env-filter` features.
- [x] Task: Conductor - User Manual Verification 'Setup Telemetry Dependencies' (Protocol in workflow.md) 06e7850

## Phase 2: Architect Dual-Layer Test Harness [checkpoint: 723b979]
- [x] Task: Create Test Harness `tests/common/telemetry.rs` 4ceda98
    - [x] Write failing test `tests/test_telemetry.rs` to verify log file creation on trace.
    - [x] Implement `init()` function configuring a non-blocking JSON writer to `target/profiling/regression_events.json` (overwrite mode).
    - [x] Build a `tracing_subscriber::Registry` with Layer 1 (JSON) and Layer 2 (TracyLayer via `#[cfg(feature = "tracy")]`).
    - [x] Register as global default subscriber.
    - [x] Ensure failing test passes.
- [x] Task: Conductor - User Manual Verification 'Architect Dual-Layer Test Harness' (Protocol in workflow.md) 723b979

## Phase 3: Instrument Core Pipeline
- [x] Task: Add Zero-Cost Instrumentation to Pipeline Stages 3a56943
    - [x] Instrument Thresholding functions (`adaptive_threshold_with_map` and related) in `crates/locus-core/src/`.
    - [x] Instrument Segmentation functions (`extract_quads_soa` and related).
    - [x] Instrument Decoding functions (`decode_batch_soa` and related).
    - [x] Instrument Pose Estimation functions.
- [x] Task: Verify Zero-Cost Compilation 3a56943
    - [x] Run `cargo check` and `cargo test` to ensure macros compile correctly without errors.
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