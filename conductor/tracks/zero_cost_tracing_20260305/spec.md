# Specification: Zero-Cost Tracing (Performance Observability)

## Overview
This track implements nanosecond-level visibility into the `locus-tag` pipeline using static `tracing` spans. The goal is to provide high-resolution profiling capabilities (via Tracy) while ensuring zero runtime overhead in production builds through compile-time erasure.

## Functional Requirements
- **Pipeline Instrumentation:** Add `#[tracing::instrument(skip_all, name = "...")]` to the boundaries of the 6 major pipeline stages in `crates/locus-core/src/detector.rs` and related modules:
    1. `thresholding`
    2. `segmentation`
    3. `quad_extraction`
    4. `homography_pass`
    5. `decoding_pass`
    6. `pose_refinement`
- **Static Spans:** Ensure that hot-loop spans do not perform dynamic string formatting. Use logical names and static field metadata.
- **Log Conversion:** Replace existing dynamic logs (`info!`, `debug!`) within these stages with structured spans or static fields where appropriate to avoid overhead.
- **Tracy Integration:** Ensure the `tracy` feature in `crates/locus-core/Cargo.toml` correctly enables `tracing-tracy` for profiling.

## Non-Functional Requirements
- **Zero Runtime Cost:** Production builds (release wheels) must be compiled with `tracing` level features (`max_level_off` or `max_level_error`) to physically remove instrumentation from the binary.
- **Nanosecond Visibility:** Spans must be placed as close as possible to the actual work boundaries to minimize measurement noise.

## Acceptance Criteria
- [ ] 6 major pipeline stages are instrumented with `#[tracing::instrument(skip_all)]`.
- [ ] No string formatting occurs within the instrumented hot-path spans.
- [ ] Compiling with `--features tracy` allows visual profiling in the Tracy client.
- [ ] Compiling for production (with level erasure) results in the complete removal of tracing instructions from the hot loop.
- [ ] The `tracy = ["dep:tracing-tracy"]` feature gate is preserved and functional.

## Out of Scope
- Adding tracing to utility functions or low-level SIMD kernels.
- Implementing new profiling tools beyond the existing Tracy integration.
