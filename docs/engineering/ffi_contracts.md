# FFI Contract Inventory

> **Scope.** This document enumerates every invariant the Python↔Rust boundary
> enforces today, keyed to its enforcement point. It is the authoritative test
> matrix for Phase A1 hardening: each row here is a test to write.

Entry points covered:

- `Detector.detect` / `Detector.detect_concurrent`
- `BoardEstimator.estimate`
- `CharucoRefiner.estimate`
- `CameraIntrinsics.__new__`
- `create_detector` (implicit constructor behind `Detector.__init__`)

Each section is organised as `{Invariant | Enforcement point | Error type | Notes}`.
"Enforcement point" is the file and line where the check actually runs —
not where the argument is declared.

---

## 1. Image buffer invariants

All four detection entry points accept a 2-D `uint8` NumPy array and route it
through `prepare_image_view` at `crates/locus-py/src/lib.rs:1631`.

| Invariant | Enforcement point | Error type |
| --- | --- | --- |
| `dtype == np.uint8` (single-frame) | `crates/locus-py/locus/__init__.py:268` | `ValueError` (Python-side) |
| `dtype == np.uint8` (per frame in concurrent path) | `crates/locus-py/locus/__init__.py:325` | `ValueError` (Python-side) |
| Shape is 2-D `(H, W)` | `PyReadonlyArray2<'_, u8>` signature at `lib.rs:593` (`BoardEstimator::estimate`), `lib.rs:758` (`CharucoRefiner::estimate`), `lib.rs:1097` (`Detector::detect`) | `TypeError` (PyO3 built-in) |
| `stride_x == 1` (row-major, C-contiguous along the last axis) | `lib.rs:1639-1653` in `prepare_image_view` | `PyValueError` — `"Array must be C-contiguous. Call np.ascontiguousarray(image) first."` |
| `stride_y >= width` and buffer length fits `(height - 1) * stride + width` | `crates/locus-core/src/image.rs:26-46` (`ImageView::new`) | `PyRuntimeError` (wrapped from `String`) |

### SIMD padding (caller responsibility — not enforced)

`ImageView::has_simd_padding()` at `image.rs:55-67` returns whether the buffer
has ≥3 bytes past the last pixel — the slack that AVX2 `gather` instructions
may touch. **It is only called by SIMD kernels internally; it is not asserted
at the FFI boundary.** Callers must ensure their NumPy arrays have at least 3
bytes of trailing slack (typical with `np.ascontiguousarray` on any buffer
whose stride exceeds its row width, but not guaranteed for tightly packed
minimal buffers).

This is a **latent footgun**. A1 should either:
1. Assert `has_simd_padding()` at `prepare_image_view` and raise `PyValueError`.
2. Copy into a padded arena buffer at the FFI boundary when the check fails.

### Intrinsics ↔ image-shape coupling (enforced)

When `intrinsics` is passed to `detect()` or `detect_concurrent()`, the
principal point is validated against each image's dimensions:

| Invariant | Enforcement point | Error type |
| --- | --- | --- |
| `0 <= cx < width` | `lib.rs:validate_principal_point` (called before `detach`) | `PyValueError` |
| `0 <= cy < height` | same | `PyValueError` |

`fx, fy` plausibility relative to the image size is not gated — it is a
calibration concern, not a correctness concern.

---

## 2. `CameraIntrinsics` construction

Defined at `crates/locus-py/src/lib.rs:214-264`.

| Invariant | Enforcement point | Error type |
| --- | --- | --- |
| `fx, fy, cx, cy` all finite (no NaN/±inf) | `CameraIntrinsics::new` | `PyValueError` |
| `fx > 0` and `fy > 0` | `CameraIntrinsics::new` | `PyValueError` |
| `distortion_model == Pinhole` → any `dist_coeffs` length (including empty) accepted | `CameraIntrinsics::new` | n/a |
| `distortion_model == BrownConrady` → `len(dist_coeffs) == 5` | `CameraIntrinsics::new` (feature-gated on `non_rectified`) | `PyValueError` |
| `distortion_model == KannalaBrandt` → `len(dist_coeffs) == 4` | `CameraIntrinsics::new` (feature-gated on `non_rectified`) | `PyValueError` |
| `BrownConrady` / `KannalaBrandt` available | Compile-time `#[cfg(feature = "non_rectified")]` | `AttributeError` at import time if feature off |

Principal-point bounds `(cx, cy)` ∈ image are checked at
`detect()`/`detect_concurrent()` — see §1 "Intrinsics ↔ image-shape coupling".

---

## 3. `DetectorConfig` validation

All configuration validation happens in `DetectorConfig::validate()` at
`crates/locus-core/src/config.rs:234-272`, invoked by
`DetectorBuilder::validated_build()`. Every `ConfigError` is mapped to
`PyValueError` at `crates/locus-py/src/lib.rs:1623-1624`.

| Invariant | Enforcement point | `ConfigError` variant |
| --- | --- | --- |
| `threshold_tile_size >= 2` | `config.rs:237-239` | `TileSizeTooSmall` |
| `decimation >= 1` | `config.rs:240-242` | `InvalidDecimation` |
| `upscale_factor >= 1` | `config.rs:243-245` | `InvalidUpscaleFactor` |
| `0.0 <= quad_min_fill_ratio < quad_max_fill_ratio <= 1.0` | `config.rs:246-254` | `InvalidFillRatio { min, max }` |
| `quad_min_edge_length > 0.0` | `config.rs:255-257` | `InvalidEdgeLength` |
| `structure_tensor_radius <= 8` | `config.rs:258-262` | `InvalidStructureTensorRadius` |
| `quad_extraction_mode == EdLines` ⇒ `refinement_mode != Erf` | `config.rs:263-266` | `EdLinesIncompatibleWithErf` |
| `quad_extraction_mode == EdLines` ⇒ `decode_mode != Soft` | `config.rs:267-269` | `EdLinesIncompatibleWithSoftDecode` |

### Fields with **no** runtime validation

The following fields are passed straight through without range checks, even
though the Pydantic `DetectorConfig` in `crates/locus-py/locus/_config.py`
declares explicit ranges:

- `threshold_min_range`, `threshold_min_radius`, `threshold_max_radius`
- `adaptive_threshold_constant`, `adaptive_threshold_gradient_threshold`
- `quad_min_area`, `quad_max_aspect_ratio`, `quad_min_edge_score`
- `subpixel_refinement_sigma`, `segmentation_margin`
- `decoder_min_contrast`, `max_hamming_error`, `gwlf_transversal_alpha`
- `quad_max_elongation`, `quad_min_density`
- `huber_delta_px`, `tikhonov_alpha_max`, `sigma_n_sq`

Python-side Pydantic catches some of these (e.g. `threshold_min_range: ge=0,
le=255`), **but only when the user constructs a `DetectorConfig` directly**.
Values passed through `Detector(**kwargs)` skip the Pydantic model and go
straight to the Rust builder. A1 should either route all kwargs through
Pydantic or extend `validate()`.

---

## 4. `Detector` constructor (`create_detector`)

Rust entry at `crates/locus-py/src/lib.rs:1314-1422`. Python wrapper at
`crates/locus-py/locus/__init__.py:159-228`.

| Invariant | Enforcement point | Error type |
| --- | --- | --- |
| `families[i] in {0,1,2,3,4}` (valid `TagFamily` discriminant) | `lib.rs:1301-1312` (`tag_family_from_i32`) | `PyValueError` |
| `preset == Grid and segmentation_connectivity == Eight` | `crates/locus-py/locus/__init__.py:197-205` | **Soft `warnings.warn`, not a hard gate** |
| Final config passes `validate()` | `lib.rs:1622-1624` via `validated_build()` | `PyValueError` (wrapped from `ConfigError`) |

### Preset ↔ connectivity is advisory only

The Grid preset internally defaults to 4-connectivity (see
`DetectorConfig::grid_default()` at `config.rs:274-354`). If the user
overrides it to 8-connectivity, the code emits a warning and proceeds.
A1 should decide whether this escalates to a hard error.

---

## 5. `detect()` vs `detect_concurrent()` differences

Both share the image invariants above. Behavioural differences:

| Aspect | `detect()` | `detect_concurrent()` |
| --- | --- | --- |
| Frame pool gating | single `FrameContext` | `max_concurrent_frames` pool (`lib.rs:1610-1614`) |
| Telemetry returned | yes (`PipelineTelemetry`) | no (documented at `__init__.py:313`) |
| Rejected-corner data returned | yes | no — built from an owned path at `lib.rs:1663 build_detection_result_from_owned` |
| GIL released | yes, for the pipeline body | yes, for the entire parallel section |

A1 should document whether callers should rely on the presence of
`rejected_corners` as a signal for path selection, or if the two paths should
converge.

---

## 6. Entry-point reference index

| Python entry | Rust entry | Line |
| --- | --- | --- |
| `Detector.__init__` → `_create_detector` | `create_detector` | `lib.rs:1314` |
| `Detector.detect` | `Detector::detect` | `lib.rs:1097` |
| `Detector.detect_concurrent` | `Detector::detect_concurrent` | `lib.rs:1236` (doc), body below |
| `BoardEstimator.estimate` | `BoardEstimator::estimate` | `lib.rs:593` |
| `CharucoRefiner.estimate` | `CharucoRefiner::estimate` | `lib.rs:758` |
| `CameraIntrinsics.__new__` | `CameraIntrinsics::new` | `lib.rs:214` |
| (internal) stride/padding check | `prepare_image_view` | `lib.rs:1631` |
| (internal) image-view constructor | `ImageView::new` | `crates/locus-core/src/image.rs:26` |
| (internal) config validator | `DetectorConfig::validate` | `crates/locus-core/src/config.rs:234` |

---

## 7. Gaps and follow-ups

These feed the Phase A1 test matrix:

- **Image:** SIMD padding is caller-responsibility and silently unsafe. Assert
  at `prepare_image_view`. (Deferred from A1.2: hardening this either breaks
  every `np.zeros((H, W))` call site or requires an arena-copy path; tracked
  separately.)
- **Config:** ~18 Rust fields have no range validation; Pydantic validators in
  `_config.py` are bypassed when users pass kwargs directly to `Detector()`.
- **Preset ↔ connectivity:** Grid + 8-connectivity is a soft warning. Decide
  hard-gate vs documentation.
- **`detect()` vs `detect_concurrent()`:** Telemetry + rejected-corner asymmetry
  is load-bearing for callers; document or converge.
