# EdLines algorithmic deep-dive + SOTA improvements (2026-05-03)

This memo audits the EdLines quad extractor's five-phase pipeline as
shipped, identifies the architectural couplings that propagate errors
phase-to-phase, and proposes three ranked improvements grounded in
classical CV literature. The motivating empirical case is
`scene_0008_cam_0000` on `locus_v1_tag36h11_1920x1080`, documented in
`scene_0008_root_cause_2026-05-03.md`.

## §1 Verified hardware

AMD EPYC-Milan KVM, 8 logical CPUs, AVX2/FMA/F16C. `--release` build
with `bench-internals`, `RAYON_NUM_THREADS=8`, CPython 3.14.3, rustc
1.92.0. Same box as V1.

## §2 EdLines pipeline as shipped

`crates/locus-core/src/edlines.rs`, called from `quad.rs:280`.

### §2.1 Phase 1 — angular-arc boundary segmentation

Two stages.

**Stage 1 (monotone scan):** for each column record only the topmost
and bottommost foreground pixel; for each row, leftmost and rightmost.
This collects ~`2·(W + H)` outer-boundary pixels and **filters interior
data-bit pixels** that would otherwise contaminate edge fits.

**Stage 2 (angular partition):** classify each boundary point by its
angle from the component centroid into one of four arcs (CW order T→R,
R→B, B→L, L→T). The "extremal" pixels that demarcate the arcs are
chosen from one of two systems:

| System | Extremals | Maps to |
|---|---|---|
| AxisAligned (TRBL) | min y / max x / max y / min x | Edge midpoints of an axis-aligned tag |
| Diagonal (NW/NE/SE/SW) | min(x+y) / min(y−x) / max(x+y) / max(y−x) | Corners of an axis-aligned tag |

Selection: try AxisAligned first; fall back to Diagonal if any
TRBL arc < 5°. Tags rotated near 0° prefer TRBL; tags near 45° prefer
Diagonal. The `imbalance_gate` (`edlines.rs:1088`) is a separate
post-Stage-2 guard that rejects axis-aligned partitions where one arc
holds > 40 % of points and another holds < 16 % — empirically the
collapse mode where two adjacent corners share the same TRBL extremal.

### §2.2 Phase 2 — IRLS line fit on binary boundary

For each of the four arcs, fit a homogeneous line `n·x + d = 0` by
total-least-squares with Huber re-weighting. δ = 1.5 px, 3 iterations,
points at integer pixel coordinates. The resulting line is in
**binary-image space** with ±0.5 px sub-pixel ambiguity by construction.

### §2.3 Phase 3 — micro-ray parabolic sub-pixel refinement

For each Phase-2 line, walk in `sample_step` increments (default 1.5 px)
along the tangent. At each probe centre, sample 5 grayscale intensities
along the line's normal at integer offsets `k ∈ {−2,−1,0,+1,+2}`,
compute three central-difference gradients
`(g_{−1}, g_0, g_{+1})`, gate on `|g_0| ≥ 8`, and fit a 3-point
parabola whose vertex `k* = −b/(2a)` is **clamped to [−1.5, +1.5]**.
The sub-pixel edge point is at `(probe + k*·n̂)`.

The clamp is the safety mechanism: if the parabola extrapolates outside
the sample window, accept the boundary instead of the extrapolation.

### §2.4 Phase 4 — sub-pixel IRLS re-fit

Re-fit a homogeneous line per arc to the Phase-3 sub-pixel points.
Tighter Huber (δ = 0.5), 2 iterations. Falls back to the scaled
Phase-2 line if `< min_edge_pts = 5` sub-pixel points were collected.

### §2.5 Phase 5 — joint 8-DoF Gauss-Newton

State vector `θ = (x_0, y_0, x_1, y_1, x_2, y_2, x_3, y_3)` (4 corners).
Cost is the sum of squared perpendicular distances from each Phase-3
sub-pixel point to the **chord** between the two corners that bound its
edge:

    r_{k,i} = (q_x − x_k) · n̂_x(θ) + (q_y − y_k) · n̂_y(θ)

where `n̂(θ)` is the normal of the chord `(x_{k+1} − x_k, y_{k+1} − y_k)`
— **a function of the optimisation variables themselves.** The
sparse 4-non-zero-per-row Jacobian is assembled and an unrolled 8×8
Cholesky solver (`cholesky_solve_8x8`) computes the GN step. Tikhonov
λ = 1e-6 added to the diagonal. Three iterations, with a ±5 px global
divergence guard against initial-condition pathologies.

A **5 % exclusion zone** at each chord end (`α ∈ [0.05, 0.95]` only)
rejects sub-pixel observations near corners — these are the most
likely to be contaminated by the perpendicular edge's PSF tail.

## §3 The architectural coupling

The five phases share an implicit constraint: **every step's notion of
"line direction" is inherited from a prior step.**

- Phase 3 walks along the **binary** line's tangent. Sub-pixel
  positions `(probe + k*·n̂_{bin})` lie on a curve that is parallel to
  the binary line; only the perpendicular offset is solved for.
- Phase 4 fits a line to those parallel-perpendicular-only points;
  the **slope of the resulting sub-pixel line is statistically
  determined by the same point cloud whose tangent direction was
  inherited from Phase 2.**
- Phase 5's cost uses `n̂(θ)` from chord direction, which on iteration 0
  is exactly the chord between Phase-4 corner intersections. As θ
  evolves, the chord rotates — but the rotation is bounded by what
  the GN cost gradient pulls (which itself depends on the points'
  perpendicular distances, which depend on the chord direction…
  which is the loop).

In particular: **if Phase 2's binary line has the wrong slope, no
downstream phase can correct it independently.** Phase 3 walks
along the wrong tangent; Phase 4's points lie along the wrong slope;
Phase 5's chord is initialised from wrong corners.

This is the architectural reason the post-EdLines `refine_corner` (at
`quad.rs:1286`) cannot help either — it inherits `p_prev` and `p_next`
from EdLines's wrong corners, then constrains its own line fit to the
chord between them (`fit_edge_line`, `quad.rs:1340`).

## §4 Failure modes

| # | Mechanism | Phase | Triggering geometry |
|---|---|---|---|
| F1 | Phase-3 parabolic vertex extrapolates outside [−1.5, +1.5]; clamping introduces directional bias proportional to (true edge offset − 1.5). | 3 | Binary line offset > 1 px from gray edge |
| F2 | Phase-3 5-sample window fixed at ±2 px around binary line; a true edge > 2 px away cannot be located in one shot. | 3 | Long thin sub-pixel offsets, e.g. tags very close to perfect-tangent |
| F3 | Phase-4 IRLS slope is a least-squares fit to points whose y-spread is small (axis-aligned edge); slope is dominated by points' x-coordinate noise → high slope variance. | 4 | Near-axis-aligned edges with short edge length |
| F4 | Phase-5 chord-direction couples slope and offset; gradient flow cannot un-rotate a wrong initial chord without a perpendicular-edge constraint that's strong enough. | 5 | Wrong Phase-4 corners |
| F5 | Phase-5 5 % exclusion zone is computed against the *current* chord; when chord is wrong, "near-corner" points are misidentified, removing exactly the points that would constrain corner position. | 5 | Wrong-chord initial conditions |
| F6 | Imbalance gate (`edlines.rs:1088`) fires only on AxisAligned mode and uses a static 16 % / 40 % threshold. Marginal cases below the threshold still produce slightly-off Diagonal partitions. | 1 | Tags rotated near 0° but not exactly axis-aligned |

scene_0008 is consistent with **F1 + F4**: a near-axis-aligned tag where
the LEFT edge has ~0.3 px binary-vs-gray offset (the "PSF inward shift"
quantified in V1), Phase 3 places sub-pixel points correctly enough that
Phase 4's slope is right, but the corner intersection from Phase 4 is
biased by ~4 px **along the top edge**. Phase 5's chord-locked GN
inherits the bias and the divergence guard does not fire (4 px < 5 px).

## §5 Three ranked SOTA improvements

### §5.1 Improvement #1 — iterate Phase 3 / Phase 4 with refined-line trajectory

**Mechanism.** Replace the single-shot Phase 3 / Phase 4 with two
passes:

```text
Phase 3 walks along bin_line[k]                  → sp_pts_v1[k]
Phase 4 fits sub-pixel line to sp_pts_v1[k]      → sp_line_v1[k]
Phase 3 walks along sp_line_v1[k]                → sp_pts_v2[k]   ← NEW
Phase 4 fits sub-pixel line to sp_pts_v2[k]      → sp_line_v2[k]   ← NEW
Phase 5 starts from intersect(sp_line_v2[*])
```

After the first pass the sub-pixel line tracks the gray edge to ~0.3 px;
re-running Phase 3 on this trajectory means the parabolic-fit window is
already centred on the edge → `k* ∈ [−0.3, +0.3]`, well within the
clamp, so the bias from F1 vanishes. Phase 4's slope on the second
pass is constrained by sub-pixel points (not binary boundary points),
so F3 also drops by an order of magnitude.

**Why this addresses F1, F2, F3.** F1 and F3 disappear because the
probe trajectory is now in gray space. F2 disappears because the
trajectory is *centred* on the edge after the first pass.

**Implementation sketch** (~50 LoC in `edlines.rs:run_pipeline_with_mode`):

```rust
// (existing) Phase 3 v1: walk binary lines.
let sp_v1: [BumpVec<(f64, f64)>; 4] = [
    refine_edge_subpixel(arena, gray, &bin_lines[0], dec, …),
    // …
];

// (existing) Phase 4 v1: fit sub-pixel lines.
let fl_v1: [Line; 4] = [
    try_refit(sp_v1[0].as_slice(), to_gray(&bin_lines[0])),
    // …
];

// === NEW: Phase 3 v2 walks the sub-pixel line ===
// fl_v1 is in gray space; pass dec=1.0 so refine_edge_subpixel does
// no decimation scaling.  Bbox is gray-space (multiply by dec once).
let sp_v2: [BumpVec<(f64, f64)>; 4] = [
    refine_edge_subpixel(
        arena, gray, &fl_v1[0], 1.0,
        min_x_bin * dec, max_x_bin * dec,
        min_y_bin * dec, max_y_bin * dec,
        cfg.sample_step, cfg.grad_min_mag,
    ),
    // …
];

// === NEW: Phase 4 v2 ===
// Fall back to fl_v1 when too few v2 points; never fall back to bin_lines.
let fl: [Line; 4] = [
    try_refit(sp_v2[0].as_slice(), fl_v1[0]),
    // …
];

// (existing Phase 5 unchanged, but uses fl[] = fl_v2)
```

**Acceptance criteria.**

- scene_0008 corner 1 ‖r‖ ≤ 0.5 px (currently 3.83 px).
- Render-tag mean RMSE on `high_accuracy + Accurate` not regressing
  by more than 1 % across `{640, 720, 1080, 2160}`.
- 1080p p99 rotation not regressing more than 5 %.
- V1 audit KL not increasing.
- All 49 typical scenes' per-corner ‖r‖ unchanged within 0.05 px
  (we expect a *tiny* improvement on most, but nothing should regress).

**Cost estimate.** ~1 day implementation + bench validation.

**Risk.** Low. The change is additive; if the second pass produces
fewer than `min_edge_pts` valid sub-pixel points (e.g., the gradient
threshold filters them out at the refined-line probe positions), we
fall back to `fl_v1`, which is the current Phase-4 output.

### §5.2 Improvement #2 — decouple Phase 5 chord direction from corners

**Mechanism.** Replace the 8-DoF state vector `(x_0, y_0, …, x_3, y_3)`
with a 12-DoF state vector parameterising **lines** instead of corners:

    θ = (n̂_0, d_0, n̂_1, d_1, n̂_2, d_2, n̂_3, d_3)

where `n̂_k` is a unit normal (1 DoF after constraint, expressed as an
angle θ_k) and `d_k` is the homogeneous offset. So actually 8 DoFs:
`(θ_0, d_0, θ_1, d_1, θ_2, d_2, θ_3, d_3)` plus the constraint
`n̂_k = (cos θ_k, sin θ_k)`.

The cost is then the sum of squared perpendicular distances from each
Phase-3 sub-pixel point to its line — independent of corners:

    r_{k,i} = q_{k,i,x} · cos θ_k + q_{k,i,y} · sin θ_k + d_k

After GN convergence, corners are derived as line intersections:
`corner_k = intersect(line_{k−1}, line_k)`.

**Why this addresses F4 and F5.** The cost gradient now pulls each
*line* toward its own sub-pixel points, not toward an inferred
chord-direction average. Slope and offset are decoupled. The 5 %
exclusion zone (F5) is no longer needed because lines are
parameterised independently of corners.

**Implementation sketch** (~150 LoC; rewrites Phase 5):

```rust
// State vector (8 DoFs: 4 lines × (angle, offset))
let mut params = [
    bin_lines[0].angle(), bin_lines[0].d,
    bin_lines[1].angle(), bin_lines[1].d,
    bin_lines[2].angle(), bin_lines[2].d,
    bin_lines[3].angle(), bin_lines[3].d,
];

for _iter in 0..GN_ITERS {
    let mut h = [0.0_f64; 64];
    let mut g = [0.0_f64; 8];
    for k in 0..4 {
        let θ = params[2*k];
        let d = params[2*k+1];
        let (cos_θ, sin_θ) = (θ.cos(), θ.sin());
        for &(qx, qy) in sp[k] {
            let r = qx * cos_θ + qy * sin_θ + d;
            // Jacobian: dr/dθ = -qx·sin θ + qy·cos θ ; dr/dd = 1
            let jθ = -qx * sin_θ + qy * cos_θ;
            // Accumulate H[2k:2k+2, 2k:2k+2] (block-diagonal — corners couple
            // implicitly through the perpendicular constraint at intersections,
            // but the line-cost itself is per-edge separable).
            …
        }
    }
    // Solve, apply, check convergence.
}

// Derive corners from line intersections.
let corners = [
    intersect_lines(&lines_from(params, 3), &lines_from(params, 0))?,
    intersect_lines(&lines_from(params, 0), &lines_from(params, 1))?,
    …
];
```

**Acceptance criteria.** Same as #1.

**Cost estimate.** ~3 days. Most of the cost is rewriting the GN +
Cholesky kernel for the new state-vector layout, plus careful test
coverage of pathological cases (two lines nearly parallel, etc.).

**Risk.** Medium. The new parameterisation changes Phase 5's
convergence basin — we expect convergence on 100 % of scenes (the
problem is per-edge convex), but the specific corner positions may
shift sub-pixel-ly relative to the current implementation, potentially
requiring a snapshot rebless across all `EdLines`-using profiles.
Snapshot drift on the 49 typical scenes is the main acceptance gate.

### §5.3 Improvement #3 — ERF in Phase 3 (replace parabolic fit)

**Mechanism.** Replace the 3-point parabolic vertex finder with a full
**erf step model** Gauss-Newton fit per micro-ray:

    I(k) = I_low + (I_high − I_low) · 0.5 · (1 + erf((k − μ) / σ))

Five intensities at `k ∈ {−2, …, +2}` already collected. Solve for
`(μ, σ, I_low, I_high)` by Levenberg-Marquardt; the sub-pixel position
is `μ`. The model's PSF-aware fit is more accurate than parabolic peak
of the gradient when the PSF is not symmetric or its scale is different
from a single pixel — both common on Blender renders.

This is exactly what `refine_edge_erf` (at `edge_refinement.rs`)
already implements as a 2D edge fit. The proposal is to integrate it
into Phase 3 *as the sub-pixel finder*, not as post-EdLines refinement.

**Why this addresses F1, F2.** ERF model fit doesn't extrapolate; if
the edge is outside the sample window, the fit either converges
(within physical bounds) or fails cleanly. F1 disappears.

**Why this isn't already done.** The `EdLines + Erf` validator block
is at the *post-EdLines refinement* layer — it prevents stacking two
sub-pixel methods. ERF *replacing* parabolic in Phase 3 is a
different integration point.

**The Phase C.5 caveat.** The negative-result memo
(`project_phase_c5_render_tag_hub_negative.md`) reported ERF post-decode
refinement has a ~0.6 px corner-RMSE floor on Blender PSF. This was
measured for **post-decode** refinement — feeding final corners through
ERF and intersecting. Different mechanism than ERF as a sub-pixel-ray
finder *during* extraction. The C.5 floor result is suggestive but not
conclusive for #3; empirical re-test required.

**Implementation sketch** (~300 LoC):

```rust
fn refine_edge_subpixel_erf<'a>(
    arena: &'a Bump,
    gray: &ImageView,
    line: &Line,
    sample_step: f64,
    min_amplitude: f64,
) -> BumpVec<'a, (f64, f64)> {
    let mut result = BumpVec::new_in(arena);
    // … walk along line tangent as before …
    // For each probe:
    //   collect 5 intensities at k ∈ {−2,…,+2}
    //   fit ERF model by 4-DoF LM
    //   if fit converged AND fitted (I_high - I_low) > min_amplitude:
    //     push (probe + μ * n̂)
    result
}
```

**Acceptance criteria.**

- Same as #1 / #2.
- Additionally: bench latency on `pipeline::quad_extraction` not
  regressing more than 5 % at 1080p (ERF LM is ~3-4× the parabolic
  fit cost).

**Cost estimate.** ~5 days; significant new code surface.

**Risk.** High. Three reasons:
1. Phase C.5 negative result raises a real concern about ERF on
   synthetic PSFs — even a 0.3 px floor would dominate the 0.14 px
   noise floor on the 49 typical scenes.
2. Latency hit on the high-PPB EdLines path (currently ~46 ms p95 at
   2160p) could push render-tag bench latency past acceptable bounds.
3. New ERF integration is the most invasive of the three improvements;
   bug surface is largest.

## §6 Risk register summary

| # | Likelihood scene_0008 fixes | Risk to 49 typical scenes | Code surface |
|---|---|---|---|
| #1 (iterate 3/4) | **Medium-high** | **Low** | ~50 LoC |
| #2 (decouple chord) | High | Medium | ~150 LoC |
| #3 (ERF in Phase 3) | High | High | ~300 LoC + new module integration |

## §7 Validation strategy

For each improvement, in order:

1. **Unit test on scene_0008 alone.** Per-corner residual capture via
   the existing `pose_cov_audit.py` extension. Pass criterion: corner 1
   ‖r‖ ≤ 0.5 px.
2. **Corpus regression.** All 50 hub scenes; measure
   - Per-corner ‖r‖ distribution (mean, p99, max).
   - Pose error d² distribution (mean, p99, KL vs χ²(6)).
3. **Render-tag snapshot rebless.** Run all 4 hub resolutions
   (`{640, 720, 1080, 2160}`); verify mean RMSE / p99 rotation
   thresholds.
4. **ICRA forward.** Recall non-regressing within ±0.5 % at the
   weighted-pose threshold.
5. **Latency.** `pipeline::quad_extraction` p95 at 2160p — ≤ +5 % vs
   baseline.

If any improvement fails (1) on scene_0008 specifically, don't proceed
to the corpus test — go to the next improvement. If (1) passes but (2)
shows corpus regression, gate the improvement behind a profile flag
(`high_accuracy_v2`?) and ship as opt-in.

## §8 What is NOT proposed

**Multi-scale boundary segmentation.** Improves robustness but not
accuracy on this corpus; defer.

**RANSAC for Phase 2.** Marginal improvement; current IRLS handles the
clean-tag case adequately; defer.

**Direct corner detector (Harris / Förstner) priors.** Plausible but
adds a parallel detection pipeline; out-of-scope for this round.
Worth re-considering once #1 / #2 / #3 results are in.

**Per-tag adaptive sample_step / grad_min_mag.** Tuning knobs that
might help one scene but risk over-fitting the corpus.

**Removing the Phase 5 chord-cost entirely.** Tested implicitly by #2
(rewriting with line parameterisation); if #2 succeeds, the chord
cost is obsoleted.

## §9 Reference

`docs/engineering/scene_0008_root_cause_2026-05-03.md` — the empirical
investigation that motivated this design memo.
