# Implementation Plan: Expose Presets to the Python Facade

## Objective
Expose the semantic `DetectorPreset` to Python consumers, abstracting complex topological and algorithmic choices. This simplifies the API for three standard, highly optimized pipelines (`sota_metrology_default`, `sota_pure_tags_default`, `sota_checkerboard_default`) and the `production_default`.

## Key Files & Context
- `crates/locus-core/src/config.rs`: Houses the core Rust factory methods. (Existing methods verify correctly).
- `crates/locus-py/src/lib.rs`: The PyO3 bindings bridging Rust and Python. Needs `DetectorPreset` exposure.
- `crates/locus-py/locus/locus.pyi`: The type stubs documenting the API. Needs updates to include `DetectorPreset`.
- `crates/locus-py/locus/__init__.py`: The pure Python wrapper around the `Detector`. Needs logic to handle the preset and emit warnings if conflicting configs are applied.

## Implementation Steps

### 1. PyO3 Binding Layer (`crates/locus-py/src/lib.rs`)
- Add a new enum `DetectorPreset` with `#[pyclass(eq, eq_int)]` and `#[derive(Clone, Copy, Debug, PartialEq)]`.
- Add variants: `Metrology = 0`, `PureTags = 1`, `Checkerboard = 2`, `Production = 3`.
- Register the enum in the `locus` PyO3 module with `m.add_class::<DetectorPreset>()?`.
- Update `create_detector`'s signature to accept `preset: Option<DetectorPreset>`.
- In `create_detector`, map the preset (if provided) to the corresponding `locus_core::DetectorConfig` factory methods (e.g. `sota_metrology_default()`, `sota_pure_tags_default()`, `sota_checkerboard_default()`, `production_default()`). If no preset is given, default to `DetectorConfig::default()`. Then apply any additional `kwargs` on top of this base configuration via the builder.

### 2. Python API Wrapping (`crates/locus-py/locus/__init__.py` and `_config.py`)
- Export `DetectorPreset` from the underlying `.locus` native module into `locus/__init__.py` and update `__all__`.
- Modify `Detector.__init__` to accept `preset: DetectorPreset | None = None`.
- Add validation logic: If `preset == DetectorPreset.Metrology` and the user forces `DecodeMode.Soft` in kwargs, issue a `warnings.warn` explaining that soft decoding with metrology causes known precision collapse.
- If `preset == DetectorPreset.Checkerboard`, emit a warning or documentation note if they try to disable connectivity constraints (e.g. `segmentation_connectivity`).
- Pass the `preset` argument through to `_create_detector` alongside other kwargs.

### 3. Type Stubs and Documentation (`crates/locus-py/locus/locus.pyi`)
- Add `class DetectorPreset(enum.IntEnum)` with the 4 variants.
- Add descriptive docstrings to each variant reflecting its trade-offs (e.g., Checkerboard enforces 4-connectivity and reduces contrast gates; Metrology uses EdLines without sharpening; PureTags uses Soft decoding).
- Update the signatures for `create_detector` and `Detector.__init__` to accept the `preset` parameter.

## Verification & Testing
- Run Python tests (`pytest tests/`) to ensure the `Detector` initialization works correctly with presets.
- Specifically, test that the `Metrology` preset triggers the warning when `DecodeMode.Soft` is passed.
- Verify that `DetectorConfig` values (like `quad_extraction_mode` or `decode_mode`) match the expected outputs from the Rust core for each preset.