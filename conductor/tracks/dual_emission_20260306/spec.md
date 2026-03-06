# Specification: Dual-Emission Observability

## Overview
Implement a dual-emission observability pipeline that sends high-fidelity binary data to the Tracy GUI for human analysis while simultaneously dumping structured JSON to `target/profiling/regression_events.json` for AI/CI pipelines. This instrumentation must be strictly isolated to regression tests, guaranteeing zero-cost execution in production (via the Python wheel).

## Functional Requirements
1. **Telemetry Dependencies Isolation**: 
   - Add `tracing-appender` to `dev-dependencies` in `crates/locus-core/Cargo.toml`.
   - Update `tracing-subscriber` in `dev-dependencies` to include `json` and `env-filter` features.
2. **Dual-Layer Test Harness**:
   - Create `tests/common/telemetry.rs` with an initialization function.
   - Configure a non-blocking JSON writer using `tracing_appender::non_blocking` pointing to `target/profiling/regression_events.json` (overwrite on each run, not rolling).
   - Build a `tracing_subscriber::Registry` with two layers:
     - Layer 1 (Agent): JSON formatter attached to the non-blocking writer.
     - Layer 2 (GUI): `tracing_tracy::TracyLayer` enabled conditionally via `#[cfg(feature = "tracy")]`.
   - Register as the global default subscriber.
3. **Core Pipeline Instrumentation (Zero-Cost)**:
   - Add `#[tracing::instrument(skip_all, name = "...")]` to major pipeline stages in `locus-core/src/`, explicitly including:
     - Thresholding (`adaptive_threshold_with_map` and related)
     - Segmentation (`extract_quads_soa` and related)
     - Decoding (`decode_batch_soa` and related)
     - Pose Estimation
4. **Regression Test Integration**:
   - Wire the telemetry initialization at the start of all tests in `regression_icra2020` and `regression_render_tag`.
   - Ensure the asynchronous JSON writer `_guard` is held until the end of each test to flush remaining events to disk.
5. **Automated Workflow Support**:
   - The CI/agent flow can run headless Tracy capture (`tracy-capture -o regression.tracy &`) and execute tests (`cargo integration-test --features tracy`).
   - Output must seamlessly provide both `regression.tracy` for GUI analysis and `target/profiling/regression_events.json` for semantic analysis.

## Out of Scope
- Instrumenting non-core pipeline functions.
- Deploying telemetry to production `locus-py` or `locus-core` releases.
- Setting up log rotation (file will be overwritten per test run).