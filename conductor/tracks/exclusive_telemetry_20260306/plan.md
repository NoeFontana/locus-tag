# Implementation Plan: Mutually Exclusive Telemetry

## Phase 1: Refactor Telemetry Initializer [checkpoint: 7e11994]
- [x] Task: Update `common::telemetry::init()` for Mutually Exclusive Modes 89d9b52
    - [x] Add `test_id: &str` parameter to `init()`.
    - [x] Read `TELEMETRY_MODE` environment variable.
    - [x] Implement conditional subscriber registry (JSON vs. Tracy vs. Silent).
- [x] Task: Test-Driven Development (TDD) for Switch Logic 89d9b52
    - [x] Write a test `tests/test_exclusive_telemetry.rs` to verify that when `TELEMETRY_MODE=json`, a JSON file is created, and when silent, no file is created.
- [x] Task: Conductor - User Manual Verification 'Refactor Telemetry Initializer' (Protocol in workflow.md) 7e11994

## Phase 2: Implement Context-Aware Output Paths [checkpoint: ab517c8]
- [x] Task: Dynamic JSON Filename Generation 89d9b52
    - [x] Use `test_id` to generate `target/profiling/{test_id}_events.json`.
    - [x] Ensure parent directories exist.
- [x] Task: TDD for Filename Context 89d9b52
    - [x] Update `tests/test_exclusive_telemetry.rs` to verify unique file names based on `test_id`.
- [x] Task: Conductor - User Manual Verification 'Implement Context-Aware Output Paths' (Protocol in workflow.md) ab517c8

## Phase 3: Wire Regression Suites
- [x] Task: Update ICRA 2020 Regression Suite 89d9b52
    - [x] Update `tests/regression_icra2020.rs` calls to `init()` with an appropriate `test_id` (e.g., "regression_icra2020").
- [x] Task: Update Render Tag Regression Suite 89d9b52
    - [x] Update `tests/regression_render_tag.rs` calls to `init()` with an appropriate `test_id`.
- [ ] Task: Conductor - User Manual Verification 'Wire Regression Suites' (Protocol in workflow.md)

## Phase 4: Document CI Matrix Strategy
- [ ] Task: Documentation Update
    - [ ] Update `docs/benchmarking.md` to include the Parallel Matrix Workflow instructions using `TELEMETRY_MODE`.
- [ ] Task: Conductor - User Manual Verification 'Document CI Matrix Strategy' (Protocol in workflow.md)
