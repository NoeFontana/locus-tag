# Specification: Performance Regression Investigation and API Alignment

## Overview
Investigate and resolve the performance gap (Recall and RMSE) observed in the `locus-py` vectorized API compared to the pre-refactor baseline. Ensure the Python API provides comprehensive access to pipeline parameters to allow fine-tuning and matching of historical performance.

## Functional Requirements
- **Configuration Audit**: Compare current `DetectorConfig` defaults in Rust with the historical values in the `main` branch (pre-refactor).
- **API Extension**: Update the Python `Detector` class and the underlying Rust bridge to expose critical pipeline parameters, including:
    - `threshold_tile_size`
    - `adaptive_threshold_constant`
    - `quad_min_fill_ratio`
    - `refinement_mode` (ensure `Erf` is correctly selectable and default)
- **Math Verification**: Perform a targeted audit of the `DetectionBatch::reassemble` and coordinate scaling logic to ensure no systematic 0.5px shifts or precision loss were introduced.

## Acceptance Criteria
- **Recall Target**: Match or exceed pre-refactor baseline (~94.35% on ICRA 2020 "forward" scenario).
- **RMSE Target**: Match or exceed pre-refactor baseline (~0.26 px on ICRA 2020 "forward" scenario).
- **Configurability**: Demonstrate that changing parameters via the Python `Detector` constructor correctly impacts detection results.
- **Verification**: `uv run --group dev --group bench tools/cli.py bench real` passes with metrics aligned to baseline.

## Out of Scope
- Implementation of new SIMD kernels.
- Adding support for entirely new tag families.
- Refactoring the core Rust detection algorithm beyond parameter tuning and bug fixes.
