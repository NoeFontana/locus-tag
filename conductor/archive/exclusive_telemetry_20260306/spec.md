# Specification: Mutually Exclusive Telemetry Matrix

## Overview
Implement a mutually exclusive telemetry system for the `locus-tag` regression suite to eliminate the "Observer Effect" during profiling. By decoupling high-fidelity binary traces (Tracy) from structured JSON logging (AI/CI payloads), we ensure that nano-second level performance benchmarks remain accurate and unpolluted by JSON serialization overhead.

## Functional Requirements
1. **Mutually Exclusive Subscriber Selection**:
   - Refactor `tests/common/telemetry.rs` to allow only one subscriber type to be active at a time.
   - Use the `TELEMETRY_MODE` environment variable to select the mode:
     - `json`: Enables the non-blocking JSON subscriber (via `tracing-appender`).
     - `tracy`: Enables the high-fidelity `TracyLayer` (only if the `tracy` feature is enabled).
     - Unset/Other: Telemetry remains silent to ensure maximum performance for general test runs.
2. **Context-Aware Output Filenames**:
   - In `json` mode, the output path should follow the pattern `target/profiling/{test_id}_events.json`.
   - The telemetry initialization function must accept an identifier (e.g., the test name or filename) to ensure parallel test runs do not overwrite each other's data.
3. **Latency-Obsessed Subscriber Registry**:
   - Ensure the `Registry` construction is optimized and only layers the selected subscriber.
   - The `tracing-tracy` layer must remain conditionally compiled behind the `tracy` feature flag.
4. **CI Matrix Documentation**:
   - Provide a technical guide (or update `docs/benchmarking.md`) describing how to implement the Parallel Matrix Workflow in GitHub Actions using the new `TELEMETRY_MODE` toggle.

## Non-Functional Requirements
- **Zero Observer Effect in Tracy Mode**: When `TELEMETRY_MODE=tracy`, the JSON subscriber must not even be initialized, ensuring no competing allocations on the main perception thread.
- **Zero Runtime Overhead (Silent Mode)**: When `TELEMETRY_MODE` is unset, the telemetry system must incur exactly zero cost beyond the baseline `tracing` macro overhead.

## Out of Scope
- Automatic modification of `.github/workflows/ci.yml`.
- Implementing log rotation for the JSON files (files will be overwritten per test suite execution).

## Acceptance Criteria
- [ ] Running `TELEMETRY_MODE=json cargo test --test regression_icra2020` generates a JSON file at the expected path.
- [ ] Running `TELEMETRY_MODE=tracy cargo test --test regression_icra2020 --features tracy` connects to a Tracy instance with zero JSON serialization logs detected.
- [ ] Running tests without `TELEMETRY_MODE` produces no profiling artifacts and maintains baseline performance.
