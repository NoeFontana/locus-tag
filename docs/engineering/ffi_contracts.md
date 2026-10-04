# FFI Contract Inventory

> **Scope.** This document enumerates every invariant the Python↔Rust boundary
> enforces today, keyed to its enforcement point. It is the authoritative test
> matrix for Phase A1 hardening: each row here is a test to write.

Entry points covered:

- `Detector.detect` / `Detector.detect_concurrent`
- `BoardEstimator.estimate`
- `CharucoRefiner.estimate`
- `CameraIntrinsics.__new__`
- `_create_detector_from_config` (implicit constructor behind `Detector.__init__`)

Each section is organised as `{Invariant | Enforcement point | Error type | Notes}`.
"Enforcement point" is the function where the check actually runs — not where
the argument is declared. Functions are named rather than line-numbered so the
table does not go stale; `crates/locus-py/src/lib.rs` is abbreviated `lib.rs`.

---

## 1. Image buffer invariants

All four detection entry points accept a 2-D `uint8` NumPy array and route it
through `prepare_image_view` in `crates/locus-py/src/lib.rs`.

| Invariant | Enforcement point | Error type |
| --- | --- | --- |
| `dtype == np.uint8` (single-frame) | `Detector.detect` in `crates/locus-py/locus/__init__.py` | `ValueError` (Python-side) |
| `dtype == np.uint8` (per frame in concurrent path) | `Detector.detect_concurrent` in `crates/locus-py/locus/__init__.py` | `ValueError` (Python-side) |
| Shape is 2-D `(H, W)` | `PyReadonlyArray2<'_, u8>` signature of `BoardEstimator::estimate`, `CharucoRefiner::estimate`, `Detector::detect` and `Detector::detect_concurrent` (`lib.rs`) | `TypeError` (PyO3 built-in) |
| `stride_x == 1` (row-major, C-contiguous along the last axis) | `lib.rs` in `prepare_image_view` | `PyValueError` — `"Array must be C-contiguous. Call np.ascontiguousarray(image) first."` |
| `strides[0] >= width` (rejects reverse-axis views with negative row stride) | `lib.rs` in `prepare_image_view` | `PyValueError` — `"Array row stride (<value>) must be >= width (<value>); negative strides…"` |
| `stride_y >= width` and buffer length fits `(height - 1) * stride + width` | `ImageView::new` (`crates/locus-core/src/image.rs`) | `PyRuntimeError` (wrapped from `String`) |
| `has_simd_padding()` — ≥3 trailing bytes past the last logical pixel | `lib.rs` in `prepare_image_view` (silently satisfied via internal copy fallback when missing) | n/a — accepted via copy into padded scratch |

### SIMD padding (A1.2 — enforced via internal copy fallback)

`ImageView::has_simd_padding()` (`image.rs`) returns whether the buffer
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

This violates the otherwise-strict "no copies at the FFI" rule in
`constraints.md` §2, deliberately: PR #287 PR-A initially rejected
under-padded inputs and broke every `np.zeros((H, W))` call site in the
public API; the fallback restores compatibility at the cost of one
`H × W` copy per `detect()` call on the tightly-packed path.

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

Defined by `CameraIntrinsics::new` in `crates/locus-py/src/lib.rs`.

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

All configuration validation happens in `DetectorConfig::validate()`
(`crates/locus-core/src/config.rs`), invoked by
`DetectorBuilder::validated_build()`. `_create_detector_from_config` maps every
`ConfigError` to `PyValueError`.

| Invariant | Enforcement point | `ConfigError` variant |
| --- | --- | --- |
| `threshold_tile_size >= 2` | `DetectorConfig::validate` | `TileSizeTooSmall` |
| `1 <= threshold_local_mean_radius <= 127` (`threshold::MAX_LOCAL_MEAN_RADIUS`: keeps box sums `< 2²⁴`, exact in the local-mean arithmetic) | `DetectorConfig::validate` | `InvalidLocalMeanRadius` |
| `threshold_noise_k` finite and `> 0` | `DetectorConfig::validate` | `InvalidNoiseK` |
| `decoder_max_border_error_rate`, when set, lies in `[0, 1]` (`None` = the family's `max_hamming / bit_count`; `1.0` disables the ring check) | `DetectorConfig::validate` | `InvalidBorderErrorRate` |
| `decimation >= 1` | `DetectorConfig::validate` | `InvalidDecimation` |
| `upscale_factor >= 1` | `DetectorConfig::validate` | `InvalidUpscaleFactor` |
| `0.0 <= quad_min_fill_ratio < quad_max_fill_ratio <= 1.0` | `DetectorConfig::validate` | `InvalidFillRatio { min, max }` |
| `quad_min_edge_length > 0.0` | `DetectorConfig::validate` | `InvalidEdgeLength` |
| `1 <= structure_tensor_radius <= 8` | `DetectorConfig::validate` | `InvalidStructureTensorRadius` |
| `0 <= pose_consistency_fpr < 1` | `DetectorConfig::validate` | `InvalidPoseConsistencyFpr` |
| `outlier_drop_d2_threshold` finite and `>= 0` | `DetectorConfig::validate` | `InvalidOutlierDropD2Threshold` |
| `quad_extraction_mode == EdLines` ⇒ `refinement_mode != Erf` (also per `AdaptivePpb` route) | `DetectorConfig::validate` | `EdLinesIncompatibleWithErf` |
| `AdaptivePpb` low/high extraction modes differ | `DetectorConfig::validate` | `AdaptivePolicyDegenerate` |
| `AdaptivePpb` threshold in `(1.0, 5.0)` | `DetectorConfig::validate` | `AdaptivePolicyThresholdOutOfRange` |
| `quad_extraction_policy == AdaptivePpb` ⇒ `refinement_mode == None` (each route carries its own refinement) | `DetectorConfig::validate` | `AdaptivePolicyStaticRefinement` |

A further check runs per call: a `Static` `EdLines`
config with distorted intrinsics fails `detect()` with
`EdLinesUnsupportedWithDistortion`.

### Fields with **no** runtime validation

The following fields are passed straight through without range checks, even
though the Pydantic `DetectorConfig` in `crates/locus-py/locus/_config.py`
declares explicit ranges:

- `quad_min_area`, `quad_max_aspect_ratio`, `quad_min_edge_score`
- `subpixel_refinement_sigma`
- `decoder_min_contrast`, `max_hamming_error`
- `quad_max_elongation`, `quad_min_density`
- `huber_delta_px`, `tikhonov_alpha_max`, `sigma_n_sq`

The Pydantic model declares ranges for most of these. `Detector(profile=…)`,
`Detector(config=…)` and `locus.DetectorBuilder.build()` all serialize a validated
`DetectorConfig`. A field assigned on an existing model after construction is not
re-validated (the model has no `validate_assignment`), so Rust's `validate()` is the
backstop for that path and for user-authored JSON handed to the Rust API.

### Field scope: live, telemetry-only, inert

A config field that is accepted, stored and echoed back by `Detector.config()`
but never read is a contract violation in its own right — the API promises an
effect it does not deliver. `crates/locus-core/tests/contract_config_inertness.rs`
is the tripwire: it mutates **every** `DetectorConfig` field on a synthetic
frame set and requires the observable output (detections, rejected candidates,
poses, covariances *and* the `binarized` / `threshold_map` telemetry images) to
change. The field list comes from an exhaustive destructuring of the struct, so
a new field fails the build until a case is written for it.

Two categories are exempt, each with a written reason in that file's allowlist.
The allowlist is two-way — an allowlisted field that *starts* changing output
also fails the test — so neither category can drift silently.

| Field | Scope |
| --- | --- |
| `nthreads` | **Live but output-invariant** by design: it picks the scoped Rayon pool (`LocusEngine::run_scoped`). Allowlisted as inert *for output*; `nthreads_selects_the_pipeline_pool` separately proves the pool is installed. |
| `threshold_local_mean_radius`, `threshold_noise_k` | **Live under `threshold_mode = LocalMean` only** (the local-mean window and its noise-calibrated offset); ignored by `TileMidExtreme`. Not allowlisted: their cases switch the mode on first. |

The tile-contrast floor that used to be the `threshold_min_range` field is now an internal
constant (`TILE_MIN_RANGE = 10` in `threshold.rs`). It is telemetry-only: the tile-validity
mask it drives is applied only while writing `telemetry.binarized`, never to the
`threshold_map` segmentation consumes.

---

## 4. `Detector` constructor (`_create_detector_from_config`)

Rust entry: `_create_detector_from_config` in `crates/locus-py/src/lib.rs`.
Python wrapper: `Detector.__init__` in `crates/locus-py/locus/__init__.py`.

The config crosses the FFI as a **JSON string**, not a typed struct: Python
passes `config.model_dump_json()` to `_create_detector_from_config(config_json=…)`,
and Rust parses it with `DetectorConfig::from_profile_json`.

| Invariant | Enforcement point | Error type |
| --- | --- | --- |
| `families[i] in {0,…,5}` (valid `TagFamily` discriminant: 16h5, 36h11, 4x4_50, 4x4_100, 6x6_250, ArUcoMip36h12) | `tag_family_from_i32` in `lib.rs` (also `set_families`) | `PyValueError` |
| Nested config invariants (field ranges, fill-ratio ordering, cross-group compatibility) | `locus._config.DetectorConfig` model validators | `pydantic.ValidationError` |
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
| Frame pool gating | single `FrameContext` | `max_concurrent_frames` pool of `FrameContext`s |
| Telemetry returned | yes (`PipelineTelemetry`) | no (documented on `Detector.detect_concurrent`) |
| Rejected-corner data returned | yes | no — empty arrays, built from owned detections by `build_detection_result_from_owned` (`lib.rs`) |
| GIL released | yes, for the pipeline body | yes, for the entire parallel section |

A1 should document whether callers should rely on the presence of
`rejected_corners` as a signal for path selection, or if the two paths should
converge.

---

## 6. Entry-point reference index

| Python entry | Rust entry | File |
| --- | --- | --- |
| `Detector.__init__` | `_create_detector_from_config` | `lib.rs` |
| `Detector.detect` | `Detector::detect` | `lib.rs` |
| `Detector.detect_concurrent` | `Detector::detect_concurrent` | `lib.rs` |
| `BoardEstimator.estimate` | `BoardEstimator::estimate` | `lib.rs` |
| `CharucoRefiner.estimate` | `CharucoRefiner::estimate` | `lib.rs` |
| `CameraIntrinsics.__new__` | `CameraIntrinsics::new` | `lib.rs` |
| (internal) stride/padding check | `prepare_image_view` | `lib.rs` |
| (internal) principal-point check | `validate_principal_point` | `lib.rs` |
| (internal) image-view constructor | `ImageView::new` | `crates/locus-core/src/image.rs` |
| (internal) config validator | `DetectorConfig::validate` | `crates/locus-core/src/config.rs` |

---

## 7. Gaps and follow-ups

These feed the Phase A1 test matrix:

- **Image:** SIMD padding is satisfied transparently at
  `prepare_image_view` (A1.2, shipped 2026-05-30): `Borrowed` zero-copy
  when the input already has ≥3 trailing bytes/row, else a `Padded`
  owned scratch (see §1). Public API unchanged from pre-PR #287.
- **Config:** the fields listed in §3 have no Rust range validation. Pydantic
  `_config.py` is the first-line gate for every Python construction path, so
  they reach Rust unchecked only through user-authored `from_profile_json`
  payloads — lift range checks into the Pydantic model where they are missing.
- **`detect()` vs `detect_concurrent()`:** Telemetry + rejected-corner asymmetry
  is load-bearing for callers; document or converge.
