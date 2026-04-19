# Documentation Overhaul Plan

## Objective
Overhaul the documentation to be up-to-date, concise, and easy to use. This focuses on fixing outdated Python API patterns (such as passing arguments to `detect()` instead of `__init__()`, removing direct usage of `DetectorConfig` constructor) and documenting `DetectorBuilder` explicitly as the required workaround for configuring concurrent detection.

## Key Files & Context
- `docs/tutorials/guide.md`
- `docs/how-to/concurrent_detection.md`
- `docs/reference/api.md`

## Implementation Steps

### 1. Update `docs/tutorials/guide.md`
- **Fix decimation usage**: Remove `decimation=2` from `detect(img)` and correctly place it in the `Detector` initialization as `Detector(decimation=2, ...)`.
- **Fix Configuration API**: Replace instances of `DetectorConfig(decode_mode=...)` passed directly to `Detector` with inline kwargs: `Detector(decode_mode=...)`.
- **Fix Presets**: Replace `Detector.checkerboard()` with `Detector(preset=locus.DetectorPreset.Grid)`.

### 2. Update `docs/how-to/concurrent_detection.md`
- Add a clear note explaining that `DetectorBuilder` is required for concurrent detection because `max_concurrent_frames` is currently only exposed through the builder API, not the standard `Detector` wrapper.
- Ensure the code examples clearly reflect the usage of the Rust-backed `DetectorBuilder`.

### 3. Update `docs/reference/api.md`
- Update the **Core Interface** section to highlight the standard `Detector(**kwargs)` usage pattern.
- In the **Concurrent Detection** section, add a note specifying that `DetectorBuilder` must be used to set `max_concurrent_frames`.

## Verification & Testing
- Run `uv run --group docs mkdocs build` to ensure the static site compiles without warnings.
- Review markdown diffs to ensure explanations are concise and accurate.
