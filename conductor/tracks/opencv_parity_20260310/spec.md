# Specification: OpenCV Parity Migration (Dictionary Alignment)

## Overview
This track aims to align the Locus-tag dictionary generation, layout, and bit order with modern OpenCV (cv2.aruco). We will consolidate the Python data generation layer, update the Rust core code generation, and realign the testing suite to use OpenCV as the single source of truth.

## Functional Requirements
- **Consolidate Data Generation**: Expand the OpenCV extractor (`extract_opencv.py`) to include AprilTag 36h11, 41h12, and all standard ArUco families.
- **Deprecate Legacy Extractors**: Delete `extract_umich.py` and standalone ArUco extractors.
- **Regenerate IR JSONs**: Overwrite existing JSONs in `data/dictionaries/` with OpenCV-backed data.
- **Update Rust Code Generation**: Simplify the Jinja template (`dictionaries.rs.j2`) to assume canonical image-space format (Y-down).
- **Update Validation**: Rewrite dictionary parity tests to use `cv2.aruco` as the test oracle.
- **Accept New Snapshots**: Run the test suite and accept the new baseline for all snapshot tests.
- **Verify Pose Outputs**: Explicitly assert that $6 \times 6$ pose covariance and translation vectors match OpenCV's `estimatePoseSingleMarkers`.

## Non-Functional Requirements
- **Performance**: Ensure the simplified template doesn't introduce performance regressions in the decoding hot loop.
- **Zero-Copy**: Maintain zero-copy/GIL-free execution in the updated decoding logic.

## Acceptance Criteria
- [ ] `extract_opencv.py` successfully extracts all target families.
- [ ] `data/dictionaries/*.json` updated and verified.
- [ ] `crates/locus-core/src/dictionaries.rs` regenerated and compiles.
- [ ] Dictionary parity tests pass using `cv2.aruco` as oracle.
- [ ] Pose estimation benchmarks match OpenCV outputs within tolerance.
- [ ] Documentation updated to reflect OpenCV coordinate system and orientation shift.

## Out of Scope
- **Adding New Tag Types**: We are only aligning existing families (AprilTag, ArUco).
- **UMich Compatibility**: Backward compatibility with the UMich C library is explicitly deprecated.

## Breaking Changes
- **Orientation Shift**: 0-degree roll orientation shifts by 90 degrees (from bottom-left to top-left origin).
- **Bit Sampling Order**: Hex codes and sampling orders will change entirely.
- **UMich Deprecation**: Legacy UMich-specific spiral layouts are removed.
