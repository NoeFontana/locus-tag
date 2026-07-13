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
| `stride_x == 1` (row-major, C-contiguous along the last axis) | `lib.rs` in `prepare_image_view` | `PyValueError` — `"Array must be C-contiguous. Call np.ascontiguousarray(image) first."` |
| `strides[0] >= width` (rejects reverse-axis views with negative row stride) | `lib.rs` in `prepare_image_view` | `PyValueError` — `"Array row stride (<value>) must be >= width (<value>); negative strides…"` |
| `stride_y >= width` and buffer length fits `(height - 1) * stride + width` | `crates/locus-core/src/image.rs:26-46` (`ImageView::new`) | `PyRuntimeError` (wrapped from `String`) |
| `has_simd_padding()` — ≥3 trailing bytes past the last logical pixel | `lib.rs` in `prepare_image_view` (silently satisfied via internal copy fallback when missing) | n/a — accepted via copy into padded scratch |

### SIMD padding (A1.2 — enforced via internal copy fallback)

`ImageView::has_simd_padding()` at `image.rs:55-67` returns whether the buffer
has ≥3 bytes past the last logical pixel — the slack that AVX2 `gather`
instructions (e.g. `sample_bilinear_v8` in `decoder.rs`) may touch when
loading 32-bit words on 8-bit data. **`prepare_image_view` is responsible for
this gate at every entry path**, but it satisfies it transparently rather
than rejecting under-padded inputs:

* If the incoming NumPy buffer already exposes ≥3 trailing bytes per row
  (`stride_y >= width + 3`, e.g. a `parent[:, :W]` slice from a wider
  parent allocation), the function returns a zero-copy `FfiImageBuffer::Borrowed`
  variant — the `ImageView` borrows directly from the NumPy data.
* Otherwise (e.g. a tightly-packed `np.zeros((H, W), dtype=np.uint8)`),
  the function copies the image into an over-allocated scratch
  `Vec<u8>` of length `H * W + 3` (the trailing 3 bytes are zero-padded
  guard bytes) and returns an `FfiImageBuffer::Padded` variant. The
  scratch buffer lives for the lifetime of the `detect()` call and is
  dropped afterwards.

The fallback violates the otherwise-strict "no copies at the FFI" rule in
`docs/engineering/constraints.md` §2. It is a deliberate trade-off —
PR #287 PR-A initially rejected under-padded inputs and broke every
`np.zeros((H, W))` call site in the public API; the redesign restored
backwards-compatibility at the cost of one `H × W`-byte allocation + copy
per `detect()` call for the tightly-packed path. PRs that touch this
fallback should re-evaluate the trade-off if frame copy becomes a
measurable cost.

To take the zero-copy fast path explicitly, allocate a wider parent and
view a column prefix:

```python
parent = np.pad(img, ((0, 0), (0, 3)))  # or `np.zeros((H, W + 3), ...)`
view = parent[:, : img.shape[1]]
detector.detect(view)  # FfiImageBuffer::Borrowed — zero-copy
```

Note that `view` is non-C-contiguous (`stride_y > width`) by design — the
`stride_x == 1` gate is what defines "C-contiguous along the last axis" at
this boundary, not NumPy's stricter `flags['C_CONTIGUOUS']`.

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

The config crosses the FFI as a **JSON string**, not a typed struct: Python
passes `config.model_dump_json()` to `_create_detector_from_config(config_json=…)`,
and Rust parses it with `DetectorConfig::from_profile_json`.

| Invariant | Enforcement point | Error type |
| --- | --- | --- |
| `families[i] in {0,1,2,3,4}` (valid `TagFamily` discriminant) | `tag_family_from_i32` in `crates/locus-py/src/lib.rs` | `PyValueError` |
| Nested config invariants (radius ordering, fill-ratio ordering, cross-group compatibility) | `locus._config.DetectorConfig` model validators | `pydantic.ValidationError` |
| `config_json` parses (known keys only, valid enum-variant strings) | `from_profile_json` serde deserialize (`deny_unknown_fields`) | `PyValueError` (wrapped from `ConfigError::ProfileParse`) |
| Final config passes Rust `DetectorConfig::validate()` | `_create_detector_from_config` → `from_profile_json` → `validated_build()` | `PyValueError` (wrapped from `ConfigError`) |

### Profile-level connectivity is an invariant, not advice

The `grid` profile JSON sets `segmentation.connectivity = "Four"`. Because
detector settings now come from a profile file rather than from kwargs,
mixing the `grid` profile with 8-connectivity is inexpressible — the user
would have to hand-edit the profile first, and at that point the setting is
theirs to own.

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

- **Image:** SIMD padding is now satisfied transparently at
  `prepare_image_view` via a copy-into-padded-scratch fallback (A1.2 —
  shipped 2026-05-30 alongside the negative-stride and lifetime-tie fixes).
  Tightly packed `np.zeros((H, W))` buffers are accepted at the FFI: the
  function returns a `Borrowed` zero-copy view when the input already has
  ≥3 trailing bytes per row, or a `Padded` owned scratch otherwise (see §1
  SIMD-padding block). The public Python API is unchanged from pre-PR #287.
  Test fixtures in `tests/` and `crates/locus-py/tests/` were briefly
  flipped to the column-prefix-view pattern by PR #287 PR-A's reject
  design and then reverted to the original tight-buffer pattern when the
  copy fallback shipped.
- **Config:** ~18 Rust fields have no range validation. With the JSON-profile
  refactor, Pydantic `_config.py` is the first-line gate and the
  `Detector(**kwargs)` escape hatch is gone, so these reach Rust only through
  user-authored `from_profile_json` payloads — lift range checks into the
  Pydantic model where they are missing.
- **`detect()` vs `detect_concurrent()`:** Telemetry + rejected-corner asymmetry
  is load-bearing for callers; document or converge.
