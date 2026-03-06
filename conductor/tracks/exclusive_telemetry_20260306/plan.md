# Implementation Plan: Mutually Exclusive Telemetry

## Phase 1: Refactor Telemetry Initializer
- [ ] Task: Update `common::telemetry::init()` for Mutually Exclusive Modes
    - [ ] Add `test_id: &str` parameter to `init()`.
    - [ ] Read `TELEMETRY_MODE` environment variable.
    - [ ] Implement conditional subscriber registry (JSON vs. Tracy vs. Silent).
- [ ] Task: Test-Driven Development (TDD) for Switch Logic
    - [ ] Write a test `tests/test_exclusive_telemetry.rs` to verify that when `TELEMETRY_MODE=json`, a JSON file is created, and when silent, no file is created.
- [ ] Task: Conductor - User Manual Verification 'Refactor Telemetry Initializer' (Protocol in workflow.md)

## Phase 2: Implement Context-Aware Output Paths
- [ ] Task: Dynamic JSON Filename Generation
    - [ ] Use `test_id` to generate `target/profiling/{test_id}_events.json`.
    - [ ] Ensure parent directories exist.
- [ ] Task: TDD for Filename Context
    - [ ] Update `tests/test_exclusive_telemetry.rs` to verify unique file names based on `test_id`.
- [ ] Task: Conductor - User Manual Verification 'Implement Context-Aware Output Paths' (Protocol in workflow.md)

## Phase 3: Wire Regression Suites
- [ ] Task: Update ICRA 2020 Regression Suite
    - [ ] Update `tests/regression_icra2020.rs` calls to `init()` with an appropriate `test_id` (e.g., "regression_icra2020").
- [ ] Task: Update Render Tag Regression Suite
    - [ ] Update `tests/regression_render_tag.rs` calls to `init()` with an appropriate `test_id`.
- [ ] Task: Conductor - User Manual Verification 'Wire Regression Suites' (Protocol in workflow.md)

## Phase 4: Document CI Matrix Strategy
- [ ] Task: Documentation Update
    - [ ] Update `docs/benchmarking.md` to include the Parallel Matrix Workflow instructions using `TELEMETRY_MODE`.
- [ ] Task: Conductor - User Manual Verification 'Document CI Matrix Strategy' (Protocol in workflow.md)
