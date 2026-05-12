# EdLines #4 — Phase 1 boundary segmentation redesign (2026-05-03)

Design memo for the next leg of the scene_0008 investigation. Prior leg
(`edlines_phase5_decoupling_2026-05-03.md`) falsified F4 (chord-locking)
as the dominant cause and concluded the bug is upstream of Phase 5, in
Phases 1–4. This memo proposes two interventions on Phase 1, ordered
by cost: **S1** (corner-region exclusion) first, **S3** (ED-Lines
anchor walk) on falsification of S1.

S2 (convex-hull partition) is documented for completeness but is not on
the execution path — its expected lift is a strict subset of S3's, and
both cost more than S1.

## §1 What is empirically established

From `scene_0008_root_cause_2026-05-03.md` and the two negative-result
memos that followed:

1. The image data is clean. Linear regression of actual sub-pixel
   transitions on scene_0008's left and top edges agrees with GT pose
   projection to **0.06 px RMS** over 143 rows. Geometric intersection
   of the two image-edge fits sits at (903.16, 456.27), 0.24 px from
   GT. The detector reports (907.25, 456.84) — **4 px off** image-edge
   data, in a direction the pixel data does not support.
2. The 4 px error originates **before Phase 5**. The line-GN decoupling
   experiment (`edlines_phase5_decoupling_*.md`) showed that bypassing
   Phase 5 entirely (intersect Phase 4 lines directly) recovers only
   4 % of the error. F4 is real but not dominant on this scene.
3. Phase 4's intersected sub-pixel lines themselves intersect at
   ≈ (907.0, 457.0) — i.e. the bias is already present in `fl[3]` and
   `fl[0]` after Phase 4's IRLS re-fit on sub-pixel points. So either
   Phase 3's sub-pixel points are biased, or Phase 2's binary-line
   slope was biased and Phase 3 inherited it.
4. The 5 % exclusion zone in Phase 5 (F5) recovers ≈ 0.18 px when
   shrunk to 1 % — small contribution. Confirmed by experiment.

## §2 The hypothesis

scene_0008's tag is rotated ≈ 90° relative to canonical, but **axis-
aligned in image space**. For an axis-aligned tag, the four TRBL
extremals (T=topmost, R=rightmost, B=bottommost, L=leftmost) land at
the **corners** of the quad, not at edge midpoints — exactly the
geometry the angular-arc partition is least robust to.

Concretely, near corner 1 (image top-left of the tag, where T and L
extremals both land near the corner pixel):

- The four arcs (T→R, R→B, B→L, L→T) pivot at the four extremals.
- Boundary pixels in a small column-neighborhood of corner-1 have
  `atan2(p − centroid)` values almost identical to the T extremal's.
- Integer-pixel rounding of column-scan topmost-pixel positions
  introduces ±0.5 px jitter in the y-coordinate; the `<` comparator
  in `extract_boundary_segments` (`edlines.rs:852-862`) deterministically
  bins each point but **a few are inevitably on the wrong side of the
  L↔T arc boundary**.
- Wrong-arc points sit at the **end** of an edge — maximum lever arm.
  Three misassigned pixels at the end of a 150-px edge produce a
  slope error of ≈ 1.5°, i.e. ≈ 4 px end-displacement on the opposite
  end. **This is the entire scene_0008 budget.**

The hypothesis is testable in a single line of code: drop boundary
points within Δ pixels of the two extremals bounding each arc.

## §3 S1 — corner-region exclusion in Phase 1

### §3.1 Specification

After `extract_boundary_segments` returns the four edge point arrays,
filter each arc k's points by Euclidean distance to the two extremals
that defined the arc:

```rust
// Pseudocode — arc k spans extremal e_from[k] → e_to[k] (CW).
for k in 0..4 {
    edges[k].retain(|p| {
        dist(p, e_from[k]) > delta && dist(p, e_to[k]) > delta
    });
}
```

The two extremals per arc come from whichever system was selected
(TRBL or Diagonal). They are already known at the call site —
`extract_boundary_segments` would need to return them alongside the
edge arrays, or the filter applied inside the function.

### §3.2 Configuration surface

```rust
// EdLinesConfig
pub corner_exclusion_px: f64,  // default 0.0 (off, byte-identical)
```

Routed through `high_accuracy.json` only. `standard.json` and
`max_recall_adaptive.json` keep `corner_exclusion_px = 0.0` so existing
snapshots are byte-identical off the opt-in path.

### §3.3 Experiment

Sweep Δ ∈ {1, 2, 3, 5} px on:

1. **scene_0008 only** (cheap probe): pose-cov audit single-scene mode,
   capture corner 1 ‖r‖ for each Δ.
2. **Full corpus** (50 scenes): re-run pose-cov audit; capture mean
   ‖r‖, p99 ‖r‖, KL.

Add a `bench-internals` telemetry hook on `run_pipeline_with_mode` that
dumps per-edge `(slope, intercept, point_count, point_count_after_filter)`
when `LOCUS_DUMP_EDLINES_PHASE1=1`. This converts the experiment into a
diagnostic regardless of outcome — we learn *which* edge has the slope
error and *how many* points the filter removed.

### §3.4 Falsification criteria

S1 is **falsified** (and we move to S3) iff:

- No Δ ∈ {1,2,3,5} reduces scene_0008 corner 1 ‖r‖ by ≥ 1 px, AND
- The telemetry shows < 5 boundary points removed per arc near corner 1.

The first criterion alone says "the fix doesn't work"; the second says
"the hypothesis was wrong" (the corner-region pixels weren't enough to
move the slope, so the slope error is from somewhere else).

S1 is **non-shippable** (even if scene_0008 improves) iff:

- Best Δ regresses corpus mean ‖r‖ by ≥ 0.03 px, OR
- Best Δ regresses corpus p99 ‖r‖ by ≥ 0.05 px, OR
- Any baseline-passing scene starts failing snapshot validation.

### §3.5 Acceptance

Ship if best Δ:

- Reduces scene_0008 corner 1 ‖r‖ by ≥ 2 px (from 3.83 to ≤ 1.83), AND
- Corpus mean ‖r‖ within ±0.005 px of baseline (0.191 px), AND
- Corpus p99 ‖r‖ within ±0.05 px of baseline (0.697 px).

The recommended ship-default would be `corner_exclusion_px = best_Δ`
on `high_accuracy` only.

### §3.6 Estimated cost

Implementation + telemetry hook: **half day**.
Sweep + write-up: **half day**.
**Total: 1 day.**

## §4 S3 — ED-Lines anchor walk (executed on S1 falsification)

### §4.1 Why skip S2 (convex-hull partition)

S2 keeps the architecture of "binary-boundary tracing → 4 line fits →
intersect → sub-pixel re-fit". Its only change is the point-to-edge
assignment heuristic (perpendicular-to-hull-edge instead of
angular-arc). Failure modes S2 fixes that S1 doesn't:

1. The angular-arc partition is wrong globally (not just near corners).

This is a narrow improvement. If S1 falsifies — meaning the corner-
neighborhood pixels are not the issue — then either the binary-boundary
tracing itself is biased (rasterization rounding propagating along the
edge), or Phase 2's IRLS is not robust to systemic boundary noise.
Neither is fixed by S2.

S3 attacks both: by driving line extraction from **gray-image
gradients** rather than binary-boundary points, it sidesteps the
integer-pixel rounding floor entirely.

### §4.2 Specification (sketch)

Replace `extract_boundary_segments + fit_line_irls` (Phases 1–2) with
the canonical Akinlar–Topal ED-Lines algorithm operating on the gray
image:

1. **Anchor extraction**: for each pixel inside the component bbox,
   compute `|∇I|` (Sobel or central-difference). Mark as anchor if
   `|∇I| > τ_anchor` AND it is a local maximum along its gradient
   direction.
2. **Edge chaining**: from each anchor, walk along the gradient
   direction (Smart Routing: try ±tangent, pick the higher-gradient
   neighbor; stop at low-gradient pixel or chain length limit).
   Produce ordered chains of pixels.
3. **Line segment extraction**: split each chain into linear segments
   via least-squares with a residual threshold (typical 1.0 px). Apply
   Helmholtz a-contrario validation: NFA(segment) < 1 (Desolneux et al.
   2000), where NFA penalizes alignment by chance.
4. **Top-4 dominant segments**: rank validated segments by length;
   select top 4 with mutual-perpendicularity gate (each pair of
   adjacent segments at ≥ 70° from parallel). Reject component if no
   valid 4-segment configuration.
5. **Pair adjacent segments → corners**: intersect each pair of
   adjacent (CW) segments to get the 4 corners. Phase 5 (joint GN)
   continues to operate on these as before.

### §4.3 Why this would fix scene_0008

The 4 px error in Phase 4's lines is, by construction, an **integer-
pixel boundary tracing artefact**: the binary boundary tracer emits
the topmost foreground pixel per column with ±0.5 px y-jitter, and the
column-direction monotone scan picks pixels in column order. The
binary tracer commits to a slope before any sub-pixel information is
seen. Phase 3 then walks orthogonal to *that* slope.

ED-Lines' gradient-anchor approach is **never tied to an integer-pixel
slope**. It always operates on continuous gradient information and
its extracted line segments have sub-pixel slope/intercept from the
first iteration. The floor is the gradient noise, not the rasterization
floor.

### §4.4 Risks

1. **Corpus regression risk**: ED-Lines anchor walks behave very
   differently from the current monotone-scan + angular-partition
   approach on small / blurred / partially-occluded tags. We will
   need to bless 50+ snapshots and may inherit failure modes the
   current scheme doesn't have.
2. **Phase C.5 negative-result memory**: Phase C.5
   (`post_decode_refinement.rs`) is on a binary-line foundation and has
   a documented ~0.6 px floor on Blender PSF. ED-Lines has its own
   gradient-thresholding floor on synthetic PSF; our parabolic-vertex
   path already approximates this and showed corpus regression in
   experiment-1 of `edlines_sota_experiments_2026-05-03.md`. **The
   hypothesis is that S3's full chain-and-validate framework is more
   robust than per-probe parabolic vertex** — but this is a hypothesis,
   not a guarantee.
3. **Anchor parameter sensitivity**: τ_anchor and the chain-length
   threshold need tuning per resolution. Risk of the corpus-vs-
   scene_0008 trade reappearing.

### §4.5 Estimated cost

Reference implementation + integration: **3 days**.
Snapshot rebless + corpus sweep: **1 day**.
Memo + cleanup: **1 day**.
**Total: 5 days.**

### §4.6 Falsification criteria

S3 is **non-shippable** iff:

- Corpus mean ‖r‖ regresses by ≥ 0.03 px (same threshold as S1), OR
- Any baseline-passing snapshot fails to be reblessable (i.e. detection
  count drops or a tag is missed entirely on a baseline scene).

If S3 fixes scene_0008 but corpus regresses, the result becomes a
**conditional ship** — opt-in on `high_accuracy` only, with the corpus
trade documented. We make that decision based on the magnitude of the
regression vs. the magnitude of the scene_0008 fix.

### §4.7 If S3 also fails

If both S1 and S3 fail to recover ≥ 2 px on scene_0008, the failure
mode is not in Phase 1–2 architecture and we have systematically
exhausted the segmentation surface. Remaining candidates would be:

- Phase 3 sub-pixel parabolic vertex (improvement #3 in the SOTA
  design memo — ERF model). Phase C.5 has a negative-result memo here,
  so risk is high.
- A fundamental rasterization-vs-pose mismatch in the Blender render
  pipeline that no detector can resolve, in which case the runtime
  gate (Track A's `corner_geometry_outlier`) is the only defensible
  path on synthetic data.

At that point the recommendation hardens: **defer further architectural
work until real-camera evidence proves scene_0008-class failures occur
on real cameras at non-trivial rates** (Track C, hardware-blocked).

## §5 Out of scope

- Phase 3 (sub-pixel) and Phase 4 (sub-pixel IRLS) changes. Both
  consume Phase 1–2's output; if Phase 1–2 are fixed, Phase 3–4
  inherit the fix.
- Phase 5 (joint GN) changes. Falsified as dominant in
  `edlines_phase5_decoupling_*.md`.
- The runtime gate work (Track A's `corner_geometry_outlier`). It
  remains the practical fix on synthetic data; this memo's work is
  upstream of it and orthogonal.
- S2 (convex-hull partition). Documented in §4.1 for the record;
  not on the execution path.

## §6 Execution plan (closed 2026-05-04)

1. **S1 (1 day)**: ✅ done. See
   `docs/engineering/edlines_s1_corner_exclusion_2026-05-04.md` for
   sweep results. Hypothesis confirmed (monotonic improvement; Phase-2
   `y = 456.5` rounding-floor signature). Acceptance not met
   (best -1.19 px on scene_0008, target -2 px). d² regresses +12-23 %.
2. **Decision (principal-engineer call, 2026-05-04)**: do not ship S1,
   do not proceed to S3. The d² regression points the top-line metric
   in the wrong direction; S3 inherits the Phase C.5 PSF-floor risk
   and is multi-day on synthetic data alone; three negative-result
   memos converge on the same answer (defer architectural work, ship
   the runtime gate). See `edlines_s1_corner_exclusion_2026-05-04.md`
   §6 for full reasoning.
3. **S3**: deferred. Design sketch (§4 above) preserved for revival
   *if and only if* real-camera evidence emerges of scene_0008-class
   failures at non-trivial rates (Track C — hardware-blocked). The
   Phase C.5 PSF-floor risk applies; revival should plan for
   measurement-of-rotation-tail directly, not just ‖r‖ proxies.
4. **Action items**: revert S1 plumbing (~25 LoC; done in same change
   as this update), keep `dump_phase1` telemetry hook for future use,
   pursue runtime gate (Track A's `corner_geometry_outlier`) as the
   practical fix on synthetic data alone (~4 h, separate small track).

## §7 Reproducing

Same harness as prior EdLines memos:

```bash
uv run maturin develop --release \
    --manifest-path crates/locus-py/Cargo.toml --features bench-internals

LOCUS_DUMP_EDLINES_PHASE1=1 \
RAYON_NUM_THREADS=8 PYTHONPATH=. uv run --group bench \
    tools/bench/pose_cov_audit.py \
    --hub-config locus_v1_tag36h11_1920x1080 \
    --scene-id scene_0008_cam_0000 \
    --output-dir diagnostics/edlines_s1_delta_<n>
```

Hardware metadata per `constraints.md §6` to be captured in each
follow-up memo.
