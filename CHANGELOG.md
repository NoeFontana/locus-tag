# Changelog

All notable changes to this project will be documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Added

- **Opt-in gradient-orthogonality corner refinement (`decoder.corner_subpix_half_window`,
  default `0` = off).**
  - **Model:** the `cv::cornerSubPix` model, run on every decoded marker after the configured
    `refinement_mode`. Each corner moves to the least-squares point that every gradient in a
    `(2h + 1)²` Gaussian-weighted window is orthogonal to, iterated (12 iterations, 0.005 px).
  - **Homography:** recomputed from the moved corners, so board refiners and pose see one
    consistent quad.
  - **Parity:** matches OpenCV's `cornerSubPix` to 0.007 px on synthetic corners
    (`refinement::subpix_tests`). The implementation is allocation-free, on a stack patch.
  - **Scope:** undistorted cameras.
  - **Why:** with each detector's mean radial bias removed, the 1-DOF ERF corners scatter
    about 2× more than `cornerSubPix` (render-tag 1080p 0.41 vs 0.21 px; ICRA random 0.30 vs
    0.12 px). The 1-DOF fit keeps the seed edge direction.
  - **Results** with `h = 4` (`cargo xtask sota`, main `a6199d4`):
    - Corner cells won: `standard` 31 → 39 of 88, the decode-first candidate 37 → 40.
    - EuRoC LOO median 0.70 → 0.26 px (OpenCV 2-bit 0.53).
    - Liu4K corner median 0.72 → 0.47 px.
    - AprilGrid corner mean 0.94 → 0.06 px.
    - ICRA circle/random corner mean 0.25/0.32 → 0.18 px.
    - Latency: within 4 % at 1 thread.
    - No cell is lost.
- **Opt-in `threshold.mode = "LocalMean"` foreground thresholder** (supersedes #383; root
  cause RC1 of #409). Segmentation marks a pixel foreground when `pixel < threshold_map[pixel]`,
  and that map was only ever the midpoint of the min/max over a 3×3 tile neighbourhood, with no
  validity gate: it follows the local *extremes*, so flat tiles speckle with sensor noise and a
  dark background between a marker border and a nearby highlight fuses with the marker.
  `LocalMean` thresholds each pixel against the mean of a `(2·local_mean_radius + 1)²` window
  minus an offset, from a sliding column-sum accumulator (≈ 0.4 MB scratch at 4K, independent
  of strip size and worker count).
- **Decode-first ordering (`quad.refine_before_decode = false`, opt-in; L1–L3 of #409).**
  Refine-first ordering (the default) sub-pixel refines every candidate quad before decoding.
  Real images produce hundreds of candidates per marker, so most of the quad stage refines
  texture that is then rejected. Decode-first decodes from the contour corners and refines
  only the candidates that decode, or are near misses within the recovery window (see Changed):
  - The edge-contrast gate scores unrefined seeds over ±1 px across the edge.
  - Decoded candidates run the skipped quad-stage refinement and its edge gate, then the
    decoder's ERF pass.
  - A match stands only if the refined quad decodes the same id within budget (and passes
    `decoder.max_border_error_rate`) at the scale that matched.
  - A marker that decodes under both orders gets the same corners (`regression_decode_first`).

  Scope: the ERF route on undistorted cameras; other refinement modes and the
  distortion-aware path still refine first. Measured with `cargo xtask sota` (setup as below;
  accuracy runs on full datasets; latency is a serial 1-thread run, mean ms/image over a
  stride-8 Liu4K and stride-2 ICRA/render-tag subset). Front end = LocalMean r = 7, k = 4,
  sharpening off, quad gates off, 4-connectivity, ring rate 0:

  | Benchmark | Front end | + decode-first | `standard` | + decode-first | Best reference |
  | :-- | --: | --: | --: | --: | :-- |
  | Liu4K recall / precision % | 66.74 / 100 | 66.48 / 99.98 | 29.11 / 99.58 | 29.02 / 99.85 | aruco_nano 66.27 / 100 |
  | Liu4K corner mean / p99 px | 0.768 / 1.893 | 0.768 / 1.879 | 0.827 / 3.287 | 0.836 / 3.467 | OpenCV APRILTAG 0.526 / 0.839 |
  | Liu4K, 1 thread, ms | 308.3 | **123.8** | 194.6 | 158.6 | aruco_nano 53.7 |
  | ICRA forward recall % | 71.56 | 73.65 | 73.75 | 74.88 | aruco_nano 53.71 |
  | ICRA forward, 1 thread, ms | 142.2 | 82.1 | 98.0 | 80.8 | aruco_nano 10.3 |
  | EuRoC recall / precision % | 86.93 / 97.98 | 87.07 / 98.16 | 20.43 / 97.93 | 20.47 / 98.24 | OpenCV APRILTAG 44.57 / 99.69 |
  | render-tag 4K, 1 thread, ms | 313.3 | 165.3 | 153.7 | 122.8 | aruco_nano 31.5 |
  | render-tag tag16h5 precision % | 97.09 | **90.91** | 83.33 | **76.92** | 100 |

  Corner error is unchanged or lower on every benchmark. The known cost is tag16h5
  precision: more tiny candidates reach the decoder, and a 16-bit code decodes texture at 2 px
  per cell by coincidence. Those false positives pass every check refine-first applies.
  Bit-bimodality evidence is the planned fix. The `standard` latency figures come from an
  earlier serial run in the same session.

- **Border-ring marker evidence (`decoder.max_border_error_rate`, opt-in).** A decoded
  candidate can be required to show the one-cell black border around its payload: the
  `4·(d + 1)` ring cells are sampled through the candidate homography, and a cell is an error
  when it reaches 90 % of the way from the payload's dark to its bright class mean (a midpoint
  cut is fooled by blur on small markers). The rate is the error budget (`0` = every ring cell
  dark; default `1.0` = off). Evidence is checked on the *reported* quad, so an outer
  quiet-zone contour that decodes only through the 0.9 scale retry is rejected; it applies to
  the pinhole and the distortion-aware decode paths, and evidence that cannot be evaluated
  (samples outside the image, a single-class payload) never rejects. Measured with
  `cargo xtask sota` at rate 0 (setup as above):

  | Benchmark | `standard` | `standard` + ring | LocalMean front end | + ring | Best reference |
  | :-- | --: | --: | --: | --: | --: |
  | render-tag tag16h5, precision % | 83.33 | **99.01** | 47.17 | **97.09** | 100 |
  | Liu4K, recall / precision % | 29.11 / 99.58 | 28.84 / **100** | 67.07 / 99.42 | **66.74 / 100** | aruco_nano 66.27 / 100 |
  | ICRA forward, recall % | 73.75 | 72.17 | 73.39 | 71.56 | aruco_nano 53.71 |
  | AprilGrid board, recall / precision % | 97.01 / 99.40 | 96.97 / 99.44 | — | 99.52 / 99.61 | aruco_nano 92.17 / 99.73 |

  Opt-in: shipped output is byte-identical; the default is decided with the profile switch.

- **Noise-calibrated local-mean offset (`threshold.noise_k`, RC3).** With `noise_k = k > 0` the
  offset is `clamp(round(k · σ̂ₙ), 2, 20)` grey levels per frame, `σ̂ₙ` the noise of the
  thresholded image: the Immerkær estimate on the raw frame (`gradient::estimate_noise_sigma`,
  allocation-free, exact median, ~2¹⁸ strided samples) times the pre-filters' white-noise gain
  (`threshold::prefilter_noise_gain`: area decimation, bilinear upscale, sharpening √29). A flat
  pixel then turns foreground with probability ≈ Φ(−k) on any sensor instead of a per-camera
  grey-level constant; with sharpening on the honest offset saturates the 20-level ceiling, so
  run `LocalMean` unsharpened. `compute_image_noise_floor` shares the estimator and no longer
  allocates. The local mean is exact (`⌊box_sum / area⌋`, radius ≤ 127) and vectorised: 1080p,
  1 thread, fastest of 100, 9.09 → 3.03 ms (tile thresholder 2.69 ms).
- Measured with `cargo xtask sota` (accuracy runs, full datasets; latency not judged here) on
  an AMD EPYC-Milan KVM guest (8 vCPU, AVX2), Linux 6.8, rustc 1.92, `maturin develop
  --release`, each detector on 1 thread (`RAYON_NUM_THREADS=1`); setup and baseline in
  `docs/engineering/benchmarking/sota_scoreboard_20261002.md`. `standard` +
  `LocalMean` r = 7, k = 4, with sharpening off, the filled-blob quad gates off and
  4-connectivity (the M3 front end, still opt-in):

  | Benchmark | Recall % | Precision % | Reference |
  | :-- | --: | --: | :-- |
  | Liu4K (924 images) | 67.07 (`standard` 29.11) | 99.42 | aruco_nano 66.27 / 100 |
  | EuRoC (`grid` + LocalMean k = 5) | 88.37 (`grid` 70.07) | 97.50 | OpenCV APRILTAG 44.53 / 99.65 |
  | render-tag 1080p / low_key / raw_pipeline | 100 / 98 / 100 | 98.0 / 100 / 96.2 | aruco_nano 100 / 100 |
  | render-tag tag16h5 | 100 | **47.2** | aruco_nano 100 / 100 |

  With the filled-blob gates *on*, `LocalMean` r = 7 collapses render-tag recall (32 % at
  1080p): local-mean foreground is a hollow ring, which the fill/density/elongation gates reject.
  The precision loss on weak dictionaries is the gates' false-positive control moving out, to
  be replaced by decoder-side marker evidence. **Opt-in only: every shipped profile keeps
  `TileMidExtreme` and `noise_k = 0`; shipped output is byte-identical.**

- **`ArUcoMip36h12` tag family** (`cv2.aruco.DICT_ARUCO_MIP_36h12`: 250 codes, 6x6 bits,
  minimum Hamming distance 12). New dictionary JSON generated with
  `examples/dictionary_generation/extract_opencv.py`, `TagFamily` variant in `locus-core` and
  `locus-py` (discriminant 5), decoder, regenerated `locus.pyi`, dictionary parity snapshot.
- **Liu4K real-photo benchmark** (`bench real --dataset liu4k`, `tools/bench/liu4k.py`). Runtime
  download from Zenodo (10.5281/zenodo.18667018, CC BY 4.0) into gitignored `tests/data/liu4k/`
  with md5 verification; reports id-agnostic quad recall and, with `--family ArUcoMip36h12`,
  id-aware decode recall/precision. Corners/ids ground truth only, so no pose metrics. No dataset
  files are committed or packaged. Attribution in `docs/engineering/benchmarking.md`.
  The scorer mirrors aruco_nano's `testperf.cpp` (same id, centre distance `<= 10 px`, first-match
  TP/FP/FN); the 10 px radius is a Liu4K-specific constant and the repo-wide match threshold is
  unchanged. Adds `bench real --sharpening/--no-sharpening` and the
  [Liu4K report](docs/engineering/benchmarking/liu4k_euroc_sota_20261001.md) (config sweep, comparison against
  OpenCV 4.10 and aruco_nano, and open detector findings).

- **`cargo xtask sota` comparative benchmarking** (`xtask/`, `tools/bench/sota/`). Builds pinned,
  unpatched references (OpenCV 4.10.0 minimal build, aruco_nano `961b18b`) plus a C++ runner,
  fetches Liu4K / EuRoC, runs Locus, aruco_nano, OpenCV and AprilTag 3 under one timing protocol,
  and scores them (aruco_nano rule on Liu4K; ground-truth-free board-consistency protocol on
  EuRoC). Every run records verified hardware/thread metadata (`xtask/README.md`).

- **SOTA scoreboard over every compatible dataset** (`cargo xtask sota list | scoreboard`).
  Benchmarks are `[sota.*]` tables in `xtask/datasets.toml` (resolved by
  `tools/bench/sota/spec.py`) and now cover ICRA 2020 forward/circle/random, the render-tag Hub
  suites and the AprilGrid / ChArUco board renders besides Liu4K and EuRoC. New ground-truth
  scorers report recall/precision/F1, recall by marker side and order-preserving corner RMSE
  (one corner relabelling per detector; a common-tag variant compares detectors on the same
  tags); Liu4K gains corner error. `refrun` covers every dictionary Locus ships that OpenCV
  predefines and OpenCV's `CORNER_REFINE_APRILTAG` operating point (`opencv-apriltag`). Each
  report ends with a win table: Locus against the best reference operating point per metric.

- **`cargo xtask data`: pinned dataset provisioning** (`xtask/datasets.toml`,
  `tools/bench/dataset_registry.py`). One manifest declares every external dataset (Hub suites, ICRA
  2020 scenarios, EuRoC, Liu4K) with its Hugging Face revision or URL checksum, licence, citation
  and destination; `list` / `fetch` / `verify` replace four ad-hoc downloaders. Hugging Face sources
  are now pinned to a repository commit (previously the default branch); readiness markers appear
  only on success (staged extraction, Hub `annotations.jsonl` renamed in last); per-dataset /
  per-subset stamps let `verify` flag missing or stale-pin copies; `LOCUS_*_DATASET_DIR` overrides
  are honoured. `bench prepare`, `prepare_liu4k`,
  `DatasetLoader.prepare_icra`, `xtask sota fetch` and `sync_hub.py`'s CLI delegate to it.
  `sync_subset_to_local` gains a `revision` argument and now raises on image-write or auxiliary
  download failures (a file absent from the repo stays benign) instead of only logging them.

### Fixed

- **Multi-family decoding picked the last matching family, not the best.** Within a scale, a
  later decoder matching within its budget replaced an earlier one with a lower Hamming
  distance (`best_code.is_none() || …` was always true). The distortion-aware path accepted on
  the frame-wide budget even when the best distance came from a decoder whose own budget it
  exceeded. Both now keep the lowest-distance match within its own decoder's budget.
- **Resampling coordinate maps (`decimation > 1`, `upscale_factor > 1`).** Corners found on a
  decimated or upscaled grid were mapped back with the integer-pixel-centre formula
  `(v + 0.5)·d − 0.5` inside Locus's +0.5 convention: seeds landed 1 px off at `decimation = 2`
  and refined corners 0.25 px off at `upscale_factor = 2`. One pair of maps,
  `image::decimated_to_full` / `image::upscaled_to_full` (pure scalings), now serves quad
  extraction, the camera-aware path, `ScaledIntrinsics` and the upscale corner/covariance mapping,
  with resampler-consistency tests. `ImageView::decimate_to` now area-averages each `d × d`
  block (it point-sampled one pixel, aliasing fine texture), and `upscale_to` rounds instead of
  truncating (which darkened every pixel by 0.5 grey). Shipped profiles use neither, so their
  output is unchanged.
- **`sample_gradient_bilinear` border fallback** sampled 0.5 px off the requested point within
  ~1.5 px of the image border (only AprilGrid board snapshots move, by ~1e-6 relative).
- **AVX2 branch of the ERF gradient projection** sampled `px` instead of the pixel centre
  `px + 0.5` (compiled only with `target_feature = "avx2"`; regression test added).
- **Detector-level GWLF** no longer refines candidates the contrast funnel already rejected.
- `detection-batch-contract.md` documents the real phase execution order and the GWLF phase.

### Changed

- **Decoder near-miss recovery is budgeted by probability, not a fixed window.** Before
  rejecting a candidate, the decoder tries to recover it: corner nudging, and under
  decode-first, refine-and-redecode. Each attempt costs tens of decodes or a corner
  refinement. The trigger used to be a hard-coded Hamming window (≤ 10 for tag36h11, ≤ 4
  otherwise), and that window is reached by almost every texture candidate: tag36h11 texture
  lies within 10 bits of about 13 codes on average, and tag16h5 within 4 bits of about 5.
  Recovery now runs only when both hold:
  - the best match is within the family's *recovery window*: the largest Hamming distance that
    random bits reach with probability ≤ 0.1 (union bound over the 4 · N rotated codes). That
    gives 36h11 6, ArUcoMip36h12 and 6x6_250 6, tag16h5 and 4x4_50 1, and 4x4_100 0.
  - the seed quad shows a marker's dark border ring, with ≤ 20 % of ring cells bright. Every
    successful recovery but one on Liu4K, ICRA, EuRoC and render-tag had < 10 % bright.

  Shipped `standard` output changes:
  - render-tag tag16h5 1080p precision 93.4 → 97.0 % (tuned 93.6 → 97.5 %), recall and RMSE
    unchanged;
  - ICRA forward recall 73.74 → 73.69 % (checkerboard grid 70.12 → 70.03 %);
  - the main render-tag, board and distortion snapshots are byte-identical.
- **Cached initial-offset scan (output byte-identical).** The decoder-style ERF fit tries 13
  edge offsets in ±2.4 px and keeps the one with the strongest projected gradient. Each offset
  rescanned and re-sampled the gradient of every pixel within one pixel of its shifted line.
  The scan now samples each pixel within 3.5 px of the line once and sums the cached values
  per offset, in the same row-major order with the same distance expression. A proptest
  checks it picks the bit-identical offset. 1 thread: ICRA decode −26 % and pipeline −15 %;
  decode −21 % on Liu4K and −13 % on render-tag.
- **Band-restricted ERF scans (output byte-identical).** ERF sample collection and the
  decoder's initial-offset gradient scan used to visit every pixel of the edge's bounding box
  and keep those within the band `|n·p + d| ≤ w`. On a diagonal edge that box is mostly empty.
  Each row now visits only the column interval that can satisfy the band test, widened by two
  pixels and gated to rows where narrowing pays, with the per-pixel tests unchanged. A proptest
  checks it never drops an in-band pixel. 1 thread: decode −30 % on Liu4K, −21 % on render-tag
  1080p and −13 % on 4K; ICRA (short edges) unchanged.
- **Tracing without the full-frame label image (output byte-identical).**
  - Segmentation returns each surviving component's runs (`LabelResult::component_runs`).
  - Boundary tracing paints a component's runs into a one-pixel-padded mask of its bounding
    box and runs the same Moore walk, so it is property-tested point-for-point equal to tracing
    the label image.
  - The detector builds the 33 MB (at 4K) `u32` label image only when an EdLines route is
    possible (`DetectorConfig::may_use_edlines`).

  1 thread: segmentation −11 to −20 %, quad extraction up to −19 %; pipeline −11 % on Liu4K
  and −10 % on render-tag 4K.
- **Contour pre-rejects before vertex selection.** Every traced outline used to pay the
  O(n log n) dominant-vertex selection before the area and compactness tests. Two tests now run
  on the raw contour first:
  - *Decodability floor.* By Pick's theorem the outline encloses `polygon area + L/2 + 1`
    pixels. A marker needs at least one pixel per cell across its `d + 2` cells, less half a
    pixel of threshold erosion per side. So outlines enclosing fewer than
    `(min_outer_dim/decimation − 1)²` pixels are dropped: 49 px² for 36h11, MIP and 6x6, and
    25 px² for 16h5 and 4x4. The same bound on the bounding box skips the trace altogether.
    The smallest detections on the scoreboard are ICRA 36h11 markers at 8.0 px, which is
    exactly 1 px per cell.
  - *Compactness pre-reject* at half the quad compactness floor (`4π·A/L² < 0.05` on the
    outline). On Liu4K these outlines cost 3.4 ms per frame and never yielded a passing quad.

  Recall, precision and corner error are identical on all 12 scoreboard datasets for both
  `standard` and the decode-first front end, and every snapshot is unchanged. 1 thread, medians
  of interleaved runs: quad extraction −11 to −20 %; pipeline −5 % (Liu4K 98.7 → 93.6 ms),
  −6 % (render-tag 4K), −11 % (1080p).
||||||| parent of 4df78d5 (perf(refine): fit each quad edge once; share exp(-s^2) in the ERF loop)
- **Faster corner refinement (output byte-identical).**
  - The quad-stage refinement fits each of a quad's four edge lines once. Each edge had been
    fitted twice, once per adjacent corner, from the same endpoints.
  - The ERF Gauss–Newton loop computes `exp(−s²)` once per sample and shares it between the
    erf approximation and the Jacobian; a proptest pins the bit-identity.
  - Decoder-style A/B estimation reuses each sample's stored pixel value. Sampling bilinearly at
    a pixel centre returns exactly that value.

  These matter most on ICRA, where about 110 markers per frame are refined. 1 thread: ICRA
  decode −36 % and pipeline −24 %; render-tag 4K decode −25 %; Liu4K decode −12 %. Output
  hashes and every snapshot are unchanged.
- **Faster candidate generation (output byte-identical).**
  - The quad edge-contrast gate is now a decision with early exit: it stops at the first
    failing edge. On unrefined decode-first seeds it accepts an edge whose chord alone passes
    without resampling the ±1 px band, since the band mean dominates the chord mean term by
    term. Property-tested equal to thresholding the full score.
  - The boundary tracer preallocates its point buffer from the component's bounding-box
    perimeter.
  - Single-worker root resolution reuses the already-resolved slot of a parent that precedes
    a run, instead of walking the union-find forest.

  Quad extraction is 12–18 % faster at 1 thread. The whole pipeline is 4–9 % faster: Liu4K
  104.9 → 98.0 ms, render-tag 4K 73.7 → 67.0 ms, ICRA 53.1 → 50.8 ms. These are medians of 12
  interleaved runs of the span-instrumented pipeline on the decode-first front end. Output
  hashes and every snapshot are unchanged.
- **Faster codebook search.** `TagDictionary::decode` makes one branch-free pass over
  `hamming << 16 | index` keys, dispatched at runtime to `popcnt`/AVX2/NEON, so the default
  x86-64 wheel no longer runs software popcount. Results are identical (lowest index on ties,
  property-tested against exhaustive search). Texture query, full search, 1 thread:
  tag36h11 5.6 → 0.67 µs, ArUcoMip36h12 2.4 → 0.30 µs.

  Together, 1-thread latency, measured with `cargo xtask sota` interleaved against the previous
  build (A-B-B-A, mean ms per image). The machine was shared (load average 1–8), so read the
  ratios:

  | Benchmark | `standard` | Decode-first front end |
  | :-- | --: | --: |
  | render-tag 1080p | −21 % | −74 % (110.2 → 28.7 ms) |
  | render-tag 4K | −25 % | −58 % (206.2 → 86.2 ms) |
  | ICRA forward | −9 % | −35 % |
  | Liu4K | −6 % | −5 % |

- **Threshold knobs.** `threshold.min_radius`, `threshold.max_radius` and
  `threshold.gradient_threshold` are removed: they were read only by
  `adaptive_threshold_gradient_window`, whose sole callers are benchmarks, so no value changed a
  detection. `threshold.constant` is now the local-mean offset (default `15`) and is ignored by
  `TileMidExtreme`. **Breaking for hand-written profile JSON:** `deny_unknown_fields` rejects the
  removed keys. `threshold.mode`, `threshold.local_mean_radius` and `threshold.noise_k` have serde
  defaults, so a profile that omits them keeps the historical behaviour.

### Removed

- `scripts/fetch_euroc_calibration.sh` (and its `LOCUS_EUROC_HF_REPO` override): use
  `cargo xtask data fetch euroc`, which no longer needs the `hf` CLI or `unzip`.

### Documentation

- **SOTA scoreboard checkpoint (2026-10-03)** (`docs/engineering/benchmarking/sota_scoreboard_20261003.md`):
  - At `main` `a6199d4`, the opt-in decode-first LocalMean candidate wins 37 of 88 cells and
    `standard` wins 32.
  - At 8 threads the candidate is faster than aruco_nano on Liu4K, render-tag 1080p and 4K.
  - 1-thread latency, corner accuracy (the M6 prerequisite) and a ChArUco recall loss remain.
  - Latency negatives are recorded in Benchmarking Lessons §4.5.
- **SOTA scoreboard baseline (2026-10-02)** (`docs/engineering/benchmarking/sota_scoreboard_20261002.md`):
  `standard` wins 31 of 88 judged cells against the best OpenCV 4.10 / aruco_nano operating point
  on 15 benchmarks; every 1-thread latency cell is lost (aruco_nano 4–10× faster).
- **Real-image competitiveness root causes (Liu4K, EuRoC).** Dated `lessons/` subsections record why
  Locus trailed aruco_nano / OpenCV on real photos and how a Locus configuration closes the recall
  gap: the segmentation threshold model, filled-blob quad pre-gates, a grey-level (not noise-scaled)
  offset, Kalibr's 2-bit tag border, refine-before-decode latency, and photometric (sRGB gamma)
  corner bias, including an EdLines outward bias that the sRGB render-tag benchmark masks.
  [Real-image competitiveness snapshot](docs/engineering/benchmarking/liu4k_euroc_sota_20261001.md)
  supersedes the 2026-09-19 Liu4K report; controlled reproducer
  `tools/bench/photometric_corner_bias.py`; `coordinates.md` gains an OpenCV/Kalibr pixel-centre
  interop note.

## Released versions

Full per-release notes live under [`docs/changelogs/`](docs/changelogs/index.md).

- [0.8.0](docs/changelogs/v0.8.0.md) - 2026-09-25
- [0.7.1](docs/changelogs/v0.7.1.md) - 2026-07-19
- [0.7.0](docs/changelogs/v0.7.0.md) - 2026-07-19
- [0.6.0](docs/changelogs/v0.6.0.md) - 2026-06-14
- [0.5.0](docs/changelogs/v0.5.0.md) - 2026-05-20
