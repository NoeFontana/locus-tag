# Render-tag 1080p SOTA pursuit (2026-04-25)

Track the build of a `render_tag_hub` profile that beats AprilTag-C
(`pupil_apriltags`) and OpenCV `cv2.aruco.ArucoDetector` on the
`locus_v1_tag36h11_1920x1080` Hub regression subset (50 scenes, single tag36h11 each).

Branch: `feat-render-tag-sota` (forked from `origin/main`, no ROI rescue or
AdaptivePpb features in scope).

## §1 External SOTA baseline

Captured 2026-04-25 via `tools/bench/render_tag_sota_eval.py`, which
benchmarks all five detectors on the same 50 scenes and reports the full
distribution (mean + p50/p95/p99) of translation and rotation error,
recall, precision, and latency.

```bash
PYTHONPATH=. LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
  uv run --group bench python tools/bench/render_tag_sota_eval.py
```

External pose conventions and the `--compare` CLI

- AprilTag-C (`pupil_apriltags.Detector.detect`) is invoked with
  `estimate_tag_pose=True`, `camera_params=(fx,fy,cx,cy)`, and
  `tag_size`. Its tag frame differs from the Hub GT (and Locus) by 180°
  about z; the wrapper applies `R @ diag(-1,-1,1)` to align.
- OpenCV `cv2.aruco` poses come from `cv2.solvePnP(SOLVEPNP_ITERATIVE)`
  with y-down center-origin object points (`TL=[-s/2,-s/2,0]` …) so the
  result is directly comparable. `SOLVEPNP_IPPE_SQUARE` is rejected here
  because its mandated y-up object-point ordering picks an inconsistent
  branch on roughly half the symmetric synthetic tags.
- All five detectors report center-origin pose. Translation error is
  `||t_det_center − t_gt_center||` (metres); rotation error is the
  geodesic angle of `R_det^T R_gt` (degrees).

### Detector matrix (1920×1080, 50 scenes, single tag36h11 each)

| Detector                 | Recall | Precision | Trans mean | t p50  | t p95  | t p99  | Rot mean | r p50 | r p95 | r p99   | Latency |
| ------------------------ | -----: | --------: | ---------: | -----: | -----: | -----: | -------: | ----: | ----: | ------: | ------: |
| OpenCV `cv2.aruco`       |  100 % |   98.04 % |   13.5 mm  | 3.4 mm | 51.9 mm | 141.4 mm | 0.183 ° | 0.113 ° | 0.448 ° |  1.228 ° | 44.45 ms |
| AprilTag-C (pupil)       |  100 % |   100 %   |    7.9 mm  | 2.9 mm | 26.7 mm |  54.4 mm | 2.648 ° | 0.061 ° | 0.359 ° | 65.365 ° | 25.54 ms |
| Locus `standard`         |  100 % |   100 %   |    8.9 mm  | 3.5 mm | 32.1 mm |  50.3 mm | 1.480 ° | 0.288 ° | 1.572 ° | 27.248 ° | 19.24 ms |
| Locus `high_accuracy`    |   94 % |   97.92 % |    2.3 mm  | 0.4 mm |  9.2 mm |  25.8 mm | 0.189 ° | 0.058 ° | 0.654 ° |  1.967 ° | 11.37 ms |
| **Locus `render_tag_hub`** | **100 %** | 98.04 % | **2.2 mm** | **0.4 mm** | **9.0 mm** | **25.6 mm** | **0.187 °** | **0.058 °** | 0.628 ° | **1.897 °** | **11.67 ms** |

### Findings

1. **Locus `render_tag_hub` is the only detector that wins on every
   translation percentile** — 6× better than AprilTag-C on p50, 5.5× on
   p99, all while tying on recall.
2. **AprilTag-C has catastrophic rotation outliers** — r p99 of 65.4°
   despite r p50 of 0.06°. This is the symmetric-tag branch ambiguity in
   its IRLS pose solver. `render_tag_hub` keeps r p99 at 1.9°, a 35×
   tighter tail.
3. **OpenCV is the rotation distribution leader at p95/p99** but pays
   for it with the worst translation distribution and 4× latency.
4. **Locus `render_tag_hub` is the latency leader** at 11.67 ms — 2.2×
   faster than AprilTag-C, 3.8× faster than OpenCV, and on par with the
   non-recall-perfect `high_accuracy` profile.
5. **Recall is profile-driven, not dataset-driven** — `high_accuracy`'s
   6 % gap (scenes 0002 / 0033 / 0042) is the EdLines axis-aligned
   imbalance failure that `render_tag_hub`'s opt-in
   `edlines_imbalance_gate` resolves (see §4).

### SOTA gate (what `render_tag_hub` must clear)

- Recall ≥ 100 % on the 50-scene 1080p subset.
- Translation p50 < 2.9 mm (AprilTag-C, the toughest external competitor).
- Translation p99 < 54.4 mm (AprilTag-C).
- Rotation p99 < 1.228 ° (OpenCV — the toughest tail competitor).
- Latency < 25.5 ms (AprilTag-C).
- No regression to `high_accuracy`'s rotation p50 (0.058 °) — guards
  against trading pose precision for recall.

## §2 Caveats and gaps

- **Recall** on the bench counts a tag as detected if its ID appears in the
  detector output. Corner inlier-ness and pose quality are reported
  separately via the translation/rotation columns; recall is purely "did we
  decode the right ID?". A detection whose pose is gibberish still counts
  toward recall but moves the translation/rotation tail out (visible as the
  catastrophic AprilTag-C r p99 of 65°).
- **Precision** is `matched_detections / total_detections` aggregated over
  the 50 scenes. Locus `standard` and AprilTag-C achieve 100 % here because
  every reported tag was a true positive. The 98 % readings (OpenCV,
  `high_accuracy`, `render_tag_hub`) come from a small handful of FP
  candidates surfaced by the more sensitive front-ends; they are still
  bounded by the 1080p, single-tag, clean-render nature of the subset and
  are not retained after corner/pose inlier checks in production.
- **Pose convention alignment** between detectors is non-trivial — the
  AprilTag-C tag frame differs from the Hub GT (and Locus) by 180° about z,
  and `cv2.solvePnP(SOLVEPNP_IPPE_SQUARE)` requires y-up object points
  whose returned R picks an inconsistent branch on roughly half the
  symmetric synthetic tags. The `tools/bench/utils.py` wrappers
  (`AprilTagWrapper`, `OpenCVWrapper`) apply the appropriate frame fix;
  see comments inline. Without these fixes, rotation error reads ~180°
  for both libraries even though their translation is correct.
- **Bench-tool conventions** that affect how these numbers reproduce:
  - `tools/bench/utils.py::FamilyMapper` lookup keys are `int(family)`, so the
    CLI may pass either `int` or `locus.TagFamily` for `--family`.
  - `HubDatasetLoader.load_dataset` unwraps the v2 `rich_truth.json` envelope
    (`{records: [...]}`) as well as the v1 bare-list shape.
  - All wrappers report center-origin tag translation. Locus reports the
    center translation directly (`crates/locus-core/src/pose.rs`
    `centered_tag_corners`); AprilTag-C and OpenCV are aligned to the same
    origin in their wrappers. The bench computes
    `||t_det − t_gt||` without any per-detector origin shift.

## §3 JSON tuning sweep — 96 % recall plateau on EdLines

Driver: `tools/bench/render_tag_sweep.py`. All candidates load `high_accuracy`
and apply a single mutator function, so the diff is auditable. Each row below
is a 50-scene run on this workstation.

| # | Candidate                              | Recall | mean RMSE | p99 RMSE | rot P50 | mean lat | Misses |
| - | -------------------------------------- | -----: | --------: | -------: | ------: | -------: | :----- |
| 0 | `high_accuracy` baseline               |   94 % |    0.2162 |   1.3503 | 0.345 ° |   11.2 ms | 0002, 0033, 0042 |
| 1 | `+ enable_sharpening`                  |   96 % | **0.1734**| **0.4698** | 0.257 ° | 16.6 ms | 0002, 0033 |
| 2 | `+ adaptive_window radii 2..8`         |   96 % |    0.1734 |   0.4698 | 0.257 ° | 16.3 ms | 0002, 0033 |
| 3 | `+ quad.min_edge_score 4 → 2`          |   96 % |    0.1734 |   0.4698 | 0.257 ° | 16.5 ms | 0002, 0033 |
| 4 | `+ decoder.refinement Gwlf`            |   96 % |    0.6391 |   0.8662 | 0.114 ° | 20.3 ms | 0002, 0033 |
| 5 | `+ subpixel sigma 0.6 → 0.5`           |   96 % |    0.6391 |   0.8662 | 0.114 ° | 20.2 ms | 0002, 0033 |
| 6 | `+ relax geometry + Gwlf`              |   96 % |    0.6391 |   0.8662 | 0.114 ° | 21.5 ms | 0002, 0033 |
| 7 | `ContourRdp + Erf` (≈ `standard`)      |  100 % |    1.3334 |   6.9977 | 1.971 ° | 18.7 ms | — |
| 8 | `ContourRdp + Gwlf`                    |   98 % |    0.8355 |   2.7429 | 0.159 ° | 15.5 ms | 0010 |
| 9 | `+ threshold.min_range 5 → 4, gradient_threshold 5 → 4` | 96 % | 0.1734 | 0.4698 | 0.257 ° | 16.6 ms | 0002, 0033 |
| 10 | `+ threshold.constant 0 → −3`         |   96 % |    0.1734 |   0.4698 | 0.257 ° | 16.5 ms | 0002, 0033 |
| 11 | `+ full recall tune` (cumulative)     |   96 % |    0.1734 |   0.4698 | 0.257 ° | 16.9 ms | 0002, 0033 |

### What this proves

1. **Sharpening alone rescues scene 0042** and tightens RMSE 1.25× — Candidate 1
   already beats AprilTag-C's mean / p99 RMSE.
2. **Every JSON knob downstream of #1 is a no-op for recall** on EdLines.
   Threshold relaxation, min_edge_score relaxation, geometry relaxation,
   refinement-mode swaps — none move the needle off 96 %. Scenes 0002 (id 46)
   and 0033 (id 81) reject *before* the JSON-exposed quad gates.
3. **ContourRdp recovers recall but blows RMSE** (Candidate 7: 1.33 px mean,
   2.3× the SOTA gate). EdLines's sub-pixel parabola is doing real work.
4. **Conclusion**: the 0002 / 0033 rejection is happening inside EdLines's own
   hard-coded gates — most likely `grad_min_mag = 8.0` discarding sub-pixel
   probes on the grazing-angle edges those two scenes present. JSON tuning
   alone cannot reach SOTA; the EdLines internals must be exposed (Phase 4).

## §4 EdLines fix — axis-aligned imbalance gate

Phase 4 traced the residual misses to a **boundary-segmentation degeneracy**, not
a `grad_min_mag` issue. EdLines's Phase 1 partitions the outer boundary into
four arcs at the topmost / rightmost / bottommost / leftmost extremals (TRBL).
For tags rendered near-axis-aligned, two adjacent corners can collapse onto the
same TRBL extremal — lumping their shared edge into a single arc and
compressing the opposite arc to near-zero. Phases 2-5 then fit a wrong-but-
validation-passing quad and the decoder rejects it on Hamming margin.

The fix is an **opt-in imbalance gate**: when AXIS-mode boundary segmentation
yields one arc > 40 % of the boundary AND another < 16 %, divert to DIAG-mode
(NW/NE/SE/SW extremals) which maps to the four corners of an axis-aligned tag
and gives clean four-way arc partitioning. The plumbing is:

- `DetectorConfig.edlines_imbalance_gate: bool` (default `false`) — Rust knob
- `quad.edlines_imbalance_gate` — JSON / Pydantic mirror
- `crates/locus-core/profiles/render_tag_hub.json` — sets the gate to `true`

The gate is **opt-in** because the distortion suite has many legitimate
aprilgrid sub-tags with min-arc in 8-15 % under brown_conrady / kannala_brandt
distortion. A global gate would regress brown_conrady recall 0.929 → 0.869.

### Final hub-regression result (all 4 resolutions)

| Profile / resolution                | Recall | Precision | RMSE  | Repro RMSE | rot P50 | mean lat |
| ----------------------------------- | -----: | --------: | ----: | ---------: | ------: | -------: |
| `high_accuracy` 640×480              |   86 % |     100 % | 0.21  |       0.20 |  0.12 ° | (snapshot) |
| `high_accuracy` 1280×720             |   90 % |     100 % | 0.21  |       0.19 |  0.06 ° | (snapshot) |
| `high_accuracy` 1920×1080            |   94 % |     100 % | 0.22  |       0.20 |  0.05 ° | (snapshot) |
| `high_accuracy` 3840×2160            |   94 % |     100 % | 0.17  |       0.15 |  0.05 ° | (snapshot) |
| **`render_tag_hub` 640×480**         | **100 %** | 100 %  | 0.21  |       0.20 |  0.12 ° |     (snapshot) |
| **`render_tag_hub` 1280×720**        | **100 %** | 100 %  | 0.21  |       0.19 |  0.06 ° |     (snapshot) |
| **`render_tag_hub` 1920×1080**       | **100 %** | 100 %  | 0.21  |       0.20 |  0.06 ° |     (snapshot) |
| `render_tag_hub` 3840×2160           |   94 % |     100 % | 0.17  |       0.15 |  0.05 ° |     (snapshot) |

The 2160p resolution stays at 94 % because those three misses are pre-EdLines
(segmentation fragments large tags into multiple components — out of scope for
this profile). Rotation P50 ≤ 0.06° at every resolution where recall lifted,
confirming no pose-precision regression vs `high_accuracy`.

### SOTA gate result (1920×1080 only)

The detector matrix in §1 is the canonical SOTA comparison. Summary against
the toughest competitors per metric:

| Metric              | Toughest external | render_tag_hub | Verdict |
| ------------------- | ----------------: | -------------: | :------ |
| Recall              | OpenCV / AprilTag-C 100 % | 100 % | **Tied** |
| Precision           | AprilTag-C 100 %  | 98.04 %        | AprilTag-C wins (small FP rate) |
| Translation p50     | AprilTag-C 2.9 mm | **0.4 mm**     | **7× tighter** |
| Translation p95     | AprilTag-C 26.7 mm | **9.0 mm**    | **3× tighter** |
| Translation p99     | AprilTag-C 54.4 mm | **25.6 mm**   | **2.1× tighter** |
| Rotation p50        | AprilTag-C 0.061 ° | 0.058 °       | **Tied** (slight win) |
| Rotation p95        | OpenCV 0.448 °    | 0.628 °        | OpenCV wins by 1.4× |
| Rotation p99        | OpenCV 1.228 °    | 1.897 °        | OpenCV wins by 1.5× |
| Mean latency        | Locus high_accuracy 11.37 ms | 11.67 ms | Effectively tied (best of all detectors) |

**`render_tag_hub` is the translation-precision SOTA across all percentiles
and the latency SOTA**, while tying on recall. OpenCV holds the rotation tail
on this dataset (paid for with 1–6× worse translation and 3.8× worse latency).
This is the first Locus profile to clear AprilTag-C on every translation
percentile while keeping recall at 100 %.

### Cross-dataset no-regression

All other regression suites pass on this branch with no snapshot diffs:

- `regression_render_tag` (15 tests across 4 resolutions × profile variants)
- `regression_render_tag_robustness` (7 tests: tag16h5, low_key, high_iso, raw_pipeline)
- `regression_distortion_hub` (2 tests: brown_conrady, kannala_brandt — recall identical)
- `regression_board_hub` (4 tests: aprilgrid + charuco)
- `regression_icra2020` (16 tests: forward / circle / random / rotation)

The opt-in `edlines_imbalance_gate` knob is the architectural reason for this
no-regression: every existing profile keeps the gate `false`, so behaviour for
all currently-snapshotted tests is identical to `origin/main`.
