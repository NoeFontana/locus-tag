# Benchmarking Lessons (consolidated, 2026-05-02)

This doc captures the **durable** learnings from two months of detector
optimization (Mar–Apr 2026). It supersedes
`historical_evolution.md`, `sota_metrology_20260321.md`,
`edlines_gauss_newton_20260321.md`, and three point-in-time progress
reports — what survives below is the part that should still inform
future work even after the surrounding numbers go stale.

For current numbers, see `release_performance_20260418.md`,
`hub_regression_20260423.md`, `render_tag_sota_20260425.md`,
`render_tag_2160p_20260425.md`, and `quad_truncation_fix_20260426.md`.

## §1 Optimization timeline (Mar 2 → Apr 26)

```
Mar 02   55.94 ms/frame  ── Initial baseline (decoding = 70 % of frame)
Mar 03   14.55 ms/frame  ── SoA migration (3.8× total)
Mar 12     accuracy      ── GWLF: 7× rotation improvement vs ContourRdp+ERF
Mar 16   0.063 ms/1024   ── DDA-SIMD decode: decode is no longer the bottleneck
Mar 19     ~3× CCL       ── SIMD CCL Fusion (RLE + LSL): defeated the memory wall
Mar 21     accuracy      ── EdLines + Joint Gauss-Newton (planarity-constrained corners)
Mar 21    presets        ── Three SOTA profiles ship: high_accuracy / standard / grid
Apr 18      v0.3.1       ── All shipped profiles default to Hard decode (precision floor)
Apr 23   regressions     ── Hub-rename + robustness suite (tag16h5 / low_key / raw_pipeline)
Apr 25      SOTA         ── render_tag_hub profile beats AprilTag-C on every t-percentile
Apr 25     2160p fix     ── pixel_count_descending_order — MAX_CANDIDATES truncation bug
Apr 26    correction     ── Truncate post-filter, not pre-filter (distortion +6.5 pp)
```

Per-stage speedups that still hold (single-thread, 1080p):

| Stage | Before SIMD | After (Apr 18) |
|---|---:|---:|
| Thresholding | 4.26 ms | **1.51 ms** (2.8×) |
| Segmentation (CCL) | ~9 ms scalar | **2.57 ms** |
| Quad extraction | 34.35 ms | **23.79 ms** |
| Decoding (1024 cands) | ~200 ms | **0.063 ms** |

Decode is no longer the bottleneck; **preprocessing dominates** at every
resolution above 720p. Future micro-optimization cycles should target
thresholding and quad extraction first.

## §2 Architecture-shaping decisions

These are the decisions that constrain future work. Violating any of
them re-introduces a class of bug we've already paid for once.

### §2.1 SoA `DetectionBatch` with hard `MAX_CANDIDATES = 1024`

**Why it matters.** Identity = index. `corners[7]`, `homographies[7]`,
`ids[7]`, `poses[7]` belong to the same physical marker. Phase
isolation (Phase A/B/C/D) is enforced by
`contract_detection_batch.rs`. The L1-friendly layout is what made the
3.8× SoA migration win.

**The 1024 ceiling is real.** EdLines accepts more candidate quads
than ContourRdp (more permissive geometric gates → smaller noise
components survive). On dense scenes the candidate count exceeds
`MAX_CANDIDATES`. Two bugs already cost us 6 weeks of confusion:

1. *Pre-filter truncation by raster order* dropped tag-bearing
   candidates with high label indices when noise filled the first
   1024 slots (2160p recall stuck at 47/50). Fix: sort by
   `pixel_count` descending. (`render_tag_2160p_20260425.md`)
2. *Pre-filter truncation by pixel-count* kept giant background blobs
   that geometric gates would have rejected anyway, and starved the
   downstream survivor set of true tag candidates (Brown–Conrady
   recall stuck at 0.870). Fix: truncate **post-filter**; let Rayon's
   ordered `collect()` preserve descending order so the smallest
   *survivors* are the ones dropped. (`quad_truncation_fix_20260426.md`)

**Forward rule:** any change to candidate ordering or truncation
must be regression-tested across **all** suites — render-tag is
order-insensitive (clean PSF/Blender renders), but ICRA and
distortion are order- and content-sensitive and will silently lose
2–3 pp recall.

### §2.2 Zero-allocation hot loop

`bumpalo::Bump` per frame; reset (cursor → 0) at frame start. No
`Vec::new`, no `Box::new`, no `HashMap::new` in `detect()`. Stack
allocation (`SmallVec`, fixed `[T; N]`) only.

The `extract_quads_soa` API was over-engineered to take a `&Bump`
parameter that nobody read; the per-component allocator is
`WORKSPACE_ARENA.with(...)` per Rayon worker. Lesson: pass the
allocator *at the level it's used*, not "for symmetry".

### §2.3 GIL release across the entire detect pass

The Python wrapper releases the GIL for the whole pipeline. Concurrent
detection of N frames scales linearly. Adding any Python callback
inside `detect()` would re-acquire the GIL and serialize all workers —
don't.

### §2.4 Telemetry erasure

`tracing` deps must carry `features = ["release_max_level_info"]` in
both `locus-core` and `locus-py`. `debug!`/`trace!` spans are
compile-time erased. Without this, per-pixel debug spans cost
~5–10 % even when no subscriber is attached.

`TELEMETRY_MODE` env is mutually exclusive: `tracy` (high-fidelity GUI)
or `json` (structured dumps). Never both — the JSON serialization
pollutes Tracy's nanosecond ring buffers (the "Observer Effect").

## §3 Profile philosophy: four shipped JSON presets

Detector settings live in `crates/locus-core/profiles/*.json`,
`include_str!`'d into the Rust crate and re-exposed to Python via
`_shipped_profile_json`. **The JSON is authoritative**; if a Rust
constant disagrees, the JSON wins.

The four presets target mutually incompatible scenarios:

| Profile | Target | Quad extract | Refinement | Sharpen | Decode | Notes |
|---|---|---|---|---|---|---|
| `high_accuracy` | Pose-precision SOTA (synthetic + lab metrology) | AdaptivePpb (ContourRdp+Erf low / EdLines+None high) + axis-imbalance gate | None on EdLines route | off | Hard | Beats AprilTag-C on every translation percentile; subsumes the former `render_tag_hub` profile (PR #2xx) |
| `standard` | Dense multi-tag production tracking | ContourRdp | Erf | on | Hard | v0.3.1+ ships Hard for 100 % precision (was Soft Mar 21) |
| `grid` | Touching-tag checkerboard | ContourRdp + 4-conn | Erf | off | Hard | 4-connectivity separates touching borders; sharpening creates halos |
| `max_recall_adaptive` | Production tracking with per-candidate PPB routing | AdaptivePpb (ContourRdp+Erf low / EdLines+None high) | Erf on low / None on high | on | Hard | Adds `max_hamming_error=2` and `post_decode_refinement=true` for blurry / distant tags |

### §3.1 Why four and not one

- **HighAccuracy** needs the lowest corner RMSE. EdLines + GN gives
  it (0.16–0.29 px on hub vs 0.99–1.15 px for production); AdaptivePpb
  routing falls back to ContourRdp+Erf for tags below ~2.5 PPB so
  ICRA-class data doesn't collapse inside EdLines, and the
  axis-imbalance gate rescues near-axis-aligned synthetic tags.
- **Standard** needs maximum recall without losing precision.
  ContourRdp + sharpening keeps recall at 100 % on dense scenes.
  Soft decode would add +19 pp recall on ICRA but causes a 10–22 %
  precision collapse when paired with EdLines (which surfaces more
  background-edge candidates) — so Standard ships ContourRdp + Hard.
- **Grid** has non-negotiable topological constraints
  (4-connectivity, relaxed contrast / edge-score gates) that hurt
  isolated-tag performance.
- **MaxRecallAdaptive** trades the two specialty pose-tail knobs
  (χ² gate, σ_gate=0.5) for `max_hamming_error=2` and
  `post_decode_refinement=true` to recover blurry / distant tags
  in real-camera tracking.

### §3.2 Decode mode is profile-bound, not user-tunable

| Decode | Use with | Effect |
|---|---|---|
| Hard | Default everywhere | Standard threshold → bit majority |
| Soft | ContourRdp on **synthetic ICRA-like** data only | +19 pp recall, ~+15 % RMSE |

Soft + EdLines is forbidden by every shipped profile — the precision
collapse is well-characterized and will recur if anyone re-enables it.

## §4 Algorithm tradeoffs that should inform future tuning

### §4.1 Refinement modes (with EdLines)

GN corners from EdLines Phase 5 are already sub-pixel. The post-
refinement stage is *counter-productive* for raw RMSE:

| Refinement (720p hub) | Corner RMSE | Repro RMSE | Rot P50 | Use case |
|---|---:|---:|---:|---|
| `None` (GN direct) | **0.166 px** | **0.541** | 0.27° | Best for metrology |
| `Erf` | 0.592 px | 0.848 | 0.28° | (don't — degrades GN) |
| `Gwlf` | 0.621 px | 0.656 | **0.16°** | Best rotation stability |

**Rule:** with `QuadExtractionMode::EdLines`, default to
`CornerRefinementMode::None`. Use `Gwlf` only when angular stability
matters more than per-corner pixel error.

ERF was designed for integer ContourRdp corners — it expects to push
them to sub-pixel positions. Applied to already-sub-pixel GN
corners, it just adds noise.

### §4.2 GN trades corner RMSE for pose precision

Joint 8-DOF Gauss-Newton (Phase 5) optimizes all four corners
simultaneously under the planarity constraint. Result: ~1 % worse
corner RMSE, but **34–39 % better median rotation** and **26–43 %
better P90 rotation**. Geometrically consistent corners → better
poses, even if individual pixel positions move slightly from their
locally-optimal locations.

### §4.3 EdLines axis-aligned imbalance gate

EdLines's Phase 1 partitions the boundary at the topmost / rightmost
/ bottommost / leftmost extremals (TRBL). On a near-axis-aligned tag,
two adjacent corners can collapse onto the same TRBL extremal —
lumping their shared edge into one arc and compressing the opposite
to near-zero. Phases 2–5 fit a wrong-but-validation-passing quad and
the decoder rejects it on Hamming margin.

The fix:
`DetectorConfig.edlines_imbalance_gate: EdLinesImbalanceGatePolicy`
(default **Disabled**). When `Enabled`, AXIS-mode arcs that hit > 40 % / < 16 %
imbalance divert to DIAG-mode (NW/NE/SE/SW extremals) which gives
clean four-way partitioning on axis-aligned tags.

**Why the `high_accuracy` profile can leave the gate on.** The
upstream guard at `crates/locus-core/src/detector.rs:194-198`
rejects EdLines under any non-trivial distortion, routing distorted
inputs through ContourRdp. So the gate is unreachable on the
distortion-bearing inputs (Brown–Conrady aprilgrid sub-tags in the
8–15 % min-arc band) where it would have collapsed legitimate
detections. The previous "opt-in" reasoning predates that guard;
post-guard, the gate is safe wherever EdLines runs.

### §4.4 PPB (pixels per bit) sets the recall floor

| PPB | Symptom | Recommendation |
|---|---|---|
| > 5.0 | Clean | `high_accuracy` is fine |
| 1.5–5.0 | Marginal | Sharpening + EdLines OK |
| 1.2–1.5 | EdLines collapses to < 1 % recall | Use ContourRdp |
| < 1.2 | Sub-pixel bit boundaries | Hard limit — Soft decode + ContourRdp + sharpening best we can do |

ICRA `forward` has ~18 % of tags below 1.2 PPB; `circle` has
thousands below 1.0 PPB (some at 0.03 PPB). This is why
`high_accuracy` reports 46.3 % recall on ICRA but 95.6 % on hub
1080p. **Real-world robotics tracking should default to `standard`,
not `high_accuracy`.**

### §4.5 Pruned algorithms (don't reintroduce)

- **Bilateral filtering** distorts the PSF; the GN solver's weighted
  least-squares already gives better noise rejection without
  geometric distortion. Removed Apr 11.
- **GridFit corner refinement** assumes a rigid 2D grid template and
  is mathematically incompatible with edge-driven GN. Removed Apr 11.
- **Iterate-until-converged threshold-tile invalid propagation,
  larger threshold tile sizes, opt-in 2× decimation** were tried as
  fixes for the 2160p recall gap. None moved the needle on detector
  recall — the gap was the SoA truncation bug, not segmentation.
  Documented in `render_tag_2160p_20260425.md` §2.

### §4.6 AdaptivePpb gracefully falls back on distortion

EdLines is geometrically incompatible with distorted intrinsics
(Huber IRLS line fit + GN solver assume Euclidean pixel geometry). Two
guards work together to keep this safe:

1. **Static `EdLines` + distortion**: the run-pipeline guard at
   `detector.rs:194-198` errors with `EdLinesUnsupportedWithDistortion`.
   This is the user-explicit-misconfiguration signal — they asked for
   EdLines on a distorted frame, we tell them.
2. **AdaptivePpb with `EdLines` high route + distortion**: the per-tag
   distortion extraction path (`extract_single_quad_with_camera`)
   silently degrades to ContourRdp, paired with the **low-route's
   refinement** (typically Erf) rather than the high-route's
   refinement. The high-PPB route's `None` refinement would skip
   sub-pixel refinement entirely on aprilgrid sub-tags, costing
   ~5 pp recall on the Brown-Conrady regression suite.

The graceful-fallback for AdaptivePpb means a single profile (e.g.
`max_recall_adaptive`, currently bound to rectified-only tests) can
load on either rectified or distorted frames without per-camera
profile switching. Static EdLines stays explicit-error so the
misconfiguration doesn't silently degrade.

## §5 Benchmark methodology traps

Things that look like algorithmic regressions but aren't:

### §5.1 Pose convention misalignment between detectors

| Detector | Convention | Fix in `tools/bench/utils.py` |
|---|---|---|
| AprilTag-C (`pupil_apriltags`) | Tag frame 180° rotated about z vs Hub GT and Locus | `R @ diag(-1, -1, 1)` |
| OpenCV `cv2.aruco` | `solvePnP(SOLVEPNP_ITERATIVE)` with y-down centered object points | Wrap with explicit object points; **never** use `IPPE_SQUARE` (picks an inconsistent branch on ~half of symmetric synthetic tags) |
| Locus | Center-origin pose (`centered_tag_corners`) | — |

Without these wrapper fixes, rotation error reads ~180° for AprilTag-C
and OpenCV even when translation is correct.

### §5.2 Cargo integration test CWD

Cargo runs each integration test with CWD = package root, **not**
workspace root. Always resolve dataset paths via `CARGO_MANIFEST_DIR`,
never bare `canonicalize`. The 2026-04-26 ICRA "byte-identical"
errata was caused by `tests/common::resolve_dataset_root` falling
through to a 1-frame stub when `LOCUS_ICRA_DATASET_DIR` resolved
relative to a wrong CWD — masking a real 2–3 pp regression.

### §5.3 Dataset priority (when results conflict)

1. **render-tag** (Hub, PSF/Blender) — production-relevant, treat as
   ground truth.
2. **distortion** (Brown–Conrady, Kannala–Brandt) — production-relevant
   for non-pinhole optics.
3. **ICRA forward / checkerboard** — community research benchmark;
   useful regression signal, but its synthetic imaging differs from
   real cameras (no PSF, hard pixel edges), so **never** trade
   render-tag pose tail or mean RMSE for ICRA recall gains.
4. **robustness** (tag16h5, low_key, high_iso, raw_pipeline) — KPI
   watchlist, not a regression bar.

### §5.4 Hardware metadata is mandatory

Every performance report must include `lscpu` / `/proc/cpuinfo`
output captured in the same session as the benchmark. State the
build profile (`--release`), thread count, and `RAYON_NUM_THREADS`.
Placeholder hardware specs (e.g. "Intel CPU" without verification)
are treated as a quality-gate failure.

### §5.5 Soft decode is a footgun outside specific scenarios

| Combo | Result |
|---|---|
| ContourRdp + Soft + ICRA | +19 pp recall, precision holds at 100 % |
| EdLines + Soft + hub | Precision collapses to 78–90 % (more candidates surfaced from background edges) |
| Anything + Soft + render-tag PSF | Marginal at best; EdLines+sharp+Hard is already at 100 % recall |

## §6 KPI watchlist (set by `regression_render_tag_robustness`)

These four 1080p single-tag subsets stress the detector under
conditions the golden render-tag suite doesn't exercise. Treat each
as a forward-looking watchlist, not a regression bar.

| Subset | Family | Baseline | Tuned ceiling | Knob |
|---|---|---:|---:|---|
| `tag16h5` | AprilTag16h5 | R 100 % / **P 27.6 %** | P 68.9 % | `decoder.max_hamming_error: 2→1` is the dominant lever |
| `high_iso` | AprilTag36h11 | R 100 % | — | No regression — high-noise demosaic is fine |
| `low_key` | AprilTag36h11 | **R 10.0 %** | R 22.0 % | `threshold.tile_size: 8→2` + neg `constant` — **22 % is the config ceiling**, future contrast-robust threshold work needed |
| `raw_pipeline` | AprilTag36h11 | **R 58.0 %** | R 60.0 % | Largely config-bound; remaining 40 % is noise-fragmented contours |

Movement on `tag16h5` precision (up = win, down = regression) and
`low_key` recall (up = win, down = regression) is interesting in
either direction. Movement on the others toward the baseline is a
regression.

## §7 ICRA 2020 community-benchmark fixture

The ICRA 2020 fiducial-tag benchmark is a long-standing research
dataset used widely in the AprilTag / fiducial-marker literature. Its
synthesis pipeline produces hard pixel edges and no PSF (different
imaging characteristics from real-camera and Blender-rendered data),
which is fine for the role it plays — head-to-head detector
comparison on a shared, reproducible reference. Its
`forward/pure_tags_images` subset has ~18 % of tags below 1.2 PPB
(sub-pixel bit boundaries), so the production-default recall on this
subset is ~74 % — the rest are structurally unrecoverable on Hard
decode without changing the decoder's noise model.

Soft decode bridges the gap on this dataset's imaging characteristics
specifically: it treats each bit as an LLR and accepts ambiguous
decodes that Hard would reject. The trade-off (precision falls on
real-camera data with PSF) is documented in §3.2 / §5.5, so the
tuning lives in a test-only fixture at
`crates/locus-core/tests/fixtures/icra_synthetic.json` rather than as
a shipped profile. Production users get Hard decode by default;
researchers reproducing literature numbers on ICRA 2020 can opt into
the fixture-driven Soft pathway.

The `regression_icra_forward_synthetic` test records the published
recall using the winning combination from the original sweep:
**Soft decode + threshold `tile_size=4`**. Recall = 0.9403, with
~6 fewer false positives per worst-case scene than Soft alone.

**Sweep findings (from the original 4-variant exploration, kept here as
documentation rather than as a permanent matrix of fixtures):**

1. **Soft decode is the only recall lever.** ContourRdp + Hard sits at
   0.738 on `forward/pure_tags_images`; Soft lifts it to 0.940 (+20.2
   pp). The remaining ~6 pp miss is structural — those tags are below
   1.2 PPB and decode-failed at the bit level.
2. **`min_area=20` has zero ICRA effect.** Neither recall nor FP rate
   changed in the sweep. Negative result: this knob is not a useful
   ICRA lever.
3. **`tile_size=4` is a precision win on top of Soft.** Same recall as
   Soft alone, but ~6 fewer FPs per worst-case scene. Finer threshold
   tiles let the gate reject background-edge candidates that Soft's
   probabilistic decode would otherwise call. **This is what the
   shipped fixture uses.**
4. **The published number for community comparison** is recall = 0.9403
   from `regression_icra_forward_synthetic`.

**Why this fixture is not a shipped profile.** Soft decode causes a
10-22 % precision collapse on data with PSF (real cameras, Blender-
rendered hub data) per §5.5. ICRA 2020's imaging characteristics don't
trigger that collapse because there's no PSF to amplify the LLR's
spurious-decode candidates — but production users almost always have
PSF and would lose precision. Keep the fixture in `tests/fixtures/`
so researchers can opt in for community comparison; never promote it
to `crates/locus-core/profiles/`.

## §8 Where to look next

- **Preprocessing dominates above 720p.** Thresholding + integral
  image is the next micro-optimization target.
- **`low_key` recall ceiling at 22 %.** Future contrast-robust
  threshold work — anything beyond histogram-equalization-style
  preprocessing.
- **`tag16h5` precision floor at 27 %.** Codebook-aware FP rejection
  (the remaining FPs are dense structural ambiguities in the 16h5
  dictionary itself, not detector bugs).
- **`raw_pipeline` recall at 58 %.** Pre-filtering pass for noise-
  fragmented contours.
- **Rotation-tail residual on 1080p hub.** Post PR #216 we lead OpenCV
  on rotation p99 (0.77° vs 1.23°). The remaining residual is EdLines
  pose ambiguity on grazing-angle synthetic renders; further headroom
  comes from the diagnostic-driven Phase 1 work (corner outlier /
  branch-flip dominance analysis).
