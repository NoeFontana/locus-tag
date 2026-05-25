# EdLines SOTA follow-up postmortem (2026-05-25)

Two empirical falsifications of EdLines-architecture changes that
PR #281's "Out of scope" list deferred:

- *"EdLines axis-imbalance recall gap on `high_accuracy` — separate
  PR."*  Attempt A in this memo.
- *"Phase-5 8-DoF line parameterisation"* (design memo
  `edlines_sota_design_2026-05-03.md §5.2` Improvement #2).
  Attempt B.

Both attempts were implemented, run against the protected
`regression_render_tag` suite, and reverted on the same day after the
suite rejected the changes.  No code shipped.  Two negative results
worth capturing because both attempts looked principled on paper and
the failure mechanisms are non-obvious until run.

## §1 TL;DR

| Attempt | Falsifying observation | Root mechanism |
|---|---|---|
| A — competitive Phase-1 mode selector | 640×480 render-tag mean RMSE +15 %, scene_0049 became worst-offender (rmse 2.26 px) at every selector tuning tried | Phase 2–5 succeeds with bad-geometry corners; downstream decoder Hamming-margin rejection invisible to the selector → no recoverable signal |
| B — Phase-5 line-parameterisation | mean RMSE **+30–43 %** and p99 rotation **+30–100 %** at every render-tag resolution | Chord-coupling is a *variance-reducing regulariser*, not a hack; per-line decoupling amplifies per-edge noise through the intersection Jacobian |

scene_0008's 3.83-px corner-1 bias remains an open issue, bounded
upstream by PR #281's outlier-aware corner-drop LM and Σ_pose
inflation runtime gates.

## §2 Attempt A — competitive Phase-1 mode selector

### §2.1 Hypothesis

Failure mode F6 in `edlines_sota_design_2026-05-03.md §4`: the
static imbalance gate at `edlines.rs:1224` rejects AXIS-mode arcs
only when `max-arc > 40 % AND min-arc < 16 %` of the boundary
points.  Marginal cases (e.g. `scene_0008_cam_0000`: arc_pts
117 / 169 / 155 / 197 → max 30.9 %, min 18.3 %) fall *just below*
the static thresholds and produce biased AXIS partitions that propagate
through Phases 2–5 as wrong-but-validation-passing quads.

The proposed fix: replace the static threshold with a **continuous
competitive selector** — compute both AXIS and DIAG Phase-1
partitions and pick whichever has the lower `max_arc / min_arc`
ratio.  AXIS preferred on near-ties via a 0.90 strict-better margin.

### §2.2 Mechanism of failure

The selector picks DIAG for scene_0008-class cases.  Phase 2–5 then
runs on the DIAG partition and **succeeds** — returns
`PipelineOutcome::Success` with geometrically valid corners whose
positions pass every Phase-5 sanity check (bbox margin, shoelace area
non-degeneracy).  The downstream **decoder** then rejects those
corners on Hamming margin (the bit-pattern reads don't pass the
decoder's quality threshold).

The plan's fall-back-on-failure path catches `BoundaryRejected` and
`PipelineFailed`, but **not** "Phase 2-5 succeeded but corners are
geometrically wrong."  The decoder is downstream of EdLines and its
rejection is invisible to the selector's mode choice.

### §2.3 Falsification chain (each gate tightening regressed differently)

Six selector configurations attempted; render-tag effects at the
1080p / 640×480 resolutions:

| Configuration | 1080p | 640×480 |
|---|---|---|
| `axis_ratio ≥ 1.40` + `0.9` margin, no other gate | scene_0008 fixed; recall preserved | **mean RMSE +15 %**, scene_0049 worst (rmse 2.26) |
| + `min_arc ≥ 25` | recall preserved | mean RMSE +15 % (scene_0049's arcs all ≥ 25) |
| `min_arc ≥ 50` | **recall 1.0 → 0.96** | **recall 1.0 → 0.86** (5 missed tags) |
| `axis_ratio ≥ 1.55 / 1.65` | scene_0008 still fixed | scene_0049 still flips |
| `bbox_area ≥ 8000` / `16000` / `40000` | recall 0.96–0.98 | recall 0.84–0.96 |
| `max_diag_ratio ≤ 1.30` | recall 0.98 | recall 0.96 |

Every selector tuning that fixed scene_0008 caused some other scene
to either decode-fail (recall drop) or detect-with-bad-geometry
(mean RMSE blow-up).  The decoder-rejection failure mode dominates
the criterion space.

### §2.4 Re-attempt conditions

This bet should NOT be re-attempted unless one of:

1. **Selection-by-attempt-decode.**  Plumb the decoder back into
   EdLines so the candidate selector can try both AXIS and DIAG, run
   each through the full pipeline including decode, and return whichever
   decodes.  Cost: ~2× EdLines work + invasive decoder coupling.  Open
   question: even this doesn't help when *both* modes decode but with
   different geometric error magnitudes.

2. **Phase-1 telemetry exposing decoder-confidence vs. partition
   choice on the 49-scene corpus.**  Would tell us whether arc-balance
   is correlated with decode-success at all on marginal-imbalance
   cases — answer empirically required before more selector work.

3. **A fundamentally different selection criterion** that anticipates
   downstream decode failure without running it.

## §3 Attempt B — Phase-5 line parameterisation (design memo Improvement #2)

### §3.1 Hypothesis

Failure modes F4 + F5 in the design memo's taxonomy: Phase 5's
8-DoF corner-state Gauss-Newton uses a chord-cost where each edge's
normal direction is derived from the two corners that bound it.
Wrong initial corners cannot be un-rotated by gradient flow because
the chord rotates with the corner moves (F4), and the 5 % corner-bleed
exclusion zone is computed against the wrong chord (F5).

The proposed fix from `edlines_sota_design_2026-05-03.md §5.2`:
replace the 8-DoF corner state with **8-DoF line state**
`(θ_k, d_k)` per edge.  Each line is fit to its own sub-pixel
observations independently; corners derived post-convergence as line
intersections.

The implementation collapses to 4 independent 2×2 closed-form GN
problems, deletes ~170 LoC of 8×8 Cholesky kernel + chord-cost
machinery, and removes the 5 % exclusion-zone hack.

### §3.2 Measured render-tag protected-suite deltas (`accuracy_baseline` profile = `high_accuracy`)

| Resolution | mean_rmse | mean_reproj | p50_rot | p99_rot |
|---:|---|---|---|---|
| 640×480 | 0.209 → **0.297 (+42 %)** | 0.179 → 0.256 | 0.123 → **0.244 (+98 %)** | 0.570 → 0.853 |
| 720p    | 0.208 → **0.282 (+36 %)** | 0.185 → 0.253 | 0.064 → **0.152 (+138 %)** | 0.652 → **1.130 (+73 %)** |
| 1080p   | 0.214 → **0.285 (+33 %)** | 0.173 → 0.236 | 0.070 → **0.117 (+67 %)** | 0.561 → **1.010 (+80 %)** |
| 2160p   | 0.180 → **0.257 (+43 %)** | 0.160 → 0.234 | 0.057 → **0.155 (+172 %)** | 1.387 → 1.226 |

scene_0008 was *partially* fixed (rmse 2.06 → 1.79) but the gain
was dwarfed by systematic regression on the 49 other scenes.  All
12 of 14 render-tag tests failed acceptance (the 2 unaffected were
the gwlf-refinement variants where Phase 5's covariance feeds a
downstream solver that compensates).

### §3.3 Mechanism of failure

The chord-cost formulation is **not** an architectural hack.  It
enforces a structural regulariser: *adjacent edges meet at a shared
corner*.  Properties:

- Each corner is constrained by **two** edges' worth of sub-pixel
  observations.  Per-edge noise averages across edges.
- The corner-bleed exclusion (`α ∈ [0.05, 0.95]`) is a downstream
  consequence, not the primary purpose.

Per-line decoupling removes the constraint.  Each line is fit to
its own ~50–200 sub-pixel observations.  The corner emerges as a
*derived* quantity from line intersection.  Per-edge noise no longer
averages across edges — it amplifies through the intersection
Jacobian (which can be large when adjacent edges meet at near-90°
but small variation in the angle gives sub-pixel corner shifts at
real-world tag sizes).

At sub-pixel edge-points-noise σ ≈ 0.1–0.3 px (the realistic regime
on Blender renders), the variance increase from decoupling dominates
the bias reduction.  Empirically: 0.05–0.15 px per-corner displacement,
accumulating to 30–100 % p99 rotation increase.

The design memo's claim *"the problem is per-edge convex; corner
positions may shift sub-pixel-ly"* underestimated the variance
amplification.  F4 (chord-coupling-locks-wrong-chord) is real, but
it dominates only on the few mode-mismatched cases (scene_0008-class).
For the 49 typical scenes, chord-coupling is doing useful regulariser
work that no clean replacement preserves.

### §3.4 Re-attempt conditions

This bet should NOT be re-attempted unless one of:

1. **Chord-cost as a soft regulariser, not a hard constraint.**  Add
   a corner-consistency penalty `λ · ‖intersect(line_a, line_b) −
   intersect_other_pair‖²` alongside the per-line residuals.  λ
   tuned to recover the chord-coupling regulariser strength
   asymptotically while letting individual lines deviate when
   evidence is strong.  Adds a per-corpus calibration knob —
   exactly the kind of magic constant the original plan claimed to
   remove.

2. **Joint state + chord-relaxed cost.**  Keep the 8-DoF corner
   state but replace the chord-cost with `(d − d_ref)²` where
   `d_ref` is the perpendicular distance from the moving corner to
   a *separately tracked* line slope.  Decouples slope from chord
   without breaking the shared-corner constraint.  ~200 LoC,
   requires careful Jacobian work.

3. **Per-line IRLS-with-covariance, not GN.**  Phase 4 already
   gives Huber-robust lines; the only thing missing is covariance
   extraction (derivable from the IRLS weights).  Skip Phase 5
   entirely.  Same variance-amplification risk applies — independent
   lines means noise amplifies through intersection.

4. **Real-camera evidence that the architectural F4 problem is
   prevalent.**  All synthetic-data evidence
   (`edlines_s1_corner_exclusion_2026-05-04.md`, attempt A above,
   attempt B above) consistently shows F4 is a tail-only failure
   mode.  Real-camera frames may have different noise correlations
   — but until we have such evidence, removing the chord-coupling
   on synthetic-data alone is poor ROI.

## §4 What this closes

- The "EdLines axis-imbalance recall gap" deferred from PR #281's
  Out-of-scope list is **closed without ship**.  F6 stays.  scene_0008's
  geometric bias stays.  Their pose-error contributions stay bounded
  by PR #281's covariance gates.
- Design memo Improvement #2 is **falsified** on synthetic corpora.
  Re-opening requires one of the four conditions in §3.4.
- Improvement #1 (iterate Phase 3/4 with refined-line trajectory) and
  Improvement #3 (ERF in Phase 3, already opt-in via
  `edlines_phase3_erf`) remain untested by this PR's work and stay
  open for separate investigation.
- F4 ↔ chord-coupling is now understood: it is a structural
  regulariser, not a hack.  Future Phase 5 work must preserve or
  replace its variance-reduction property, not just remove it.

## §5 What to pursue instead

For SOTA pose accuracy with EdLines as a constraint:

- **Per-corner ERF residual MSE plumbing.**  Phase 4 of the global-
  covariance sweep (`anti_patterns_global_covariance_multipliers.md`)
  was the only lever the sweep didn't rule out.  Open.
- **Improvement #1 from the design memo** (iterate Phase 3/4 with
  refined-line trajectory).  Targets F1/F2/F3 (Phase-3 clamping +
  Phase-4 slope variance).  Different code path from this PR's
  attempts.
- **Board-pose follow-ups.**  PR #282 shipped per-corner outlier-aware
  drop on `refine_aw_lm`; the per-tag pose covariance audit may
  identify further levers.
- **High-ISO upstream fix.**  Catastrophic-regime tracked in
  `rotation_tail_diagnostic_phase0_*.md`.
- **Phase 3 ERF replacement** (design memo Improvement #3).  Already
  opt-in (`edlines_phase3_erf`), orthogonal to Phase 5.

## §6 Reproducing

Both attempts were ~330 LoC of `edlines.rs` changes each, fully
reverted via `git checkout`.  To reproduce attempt A:

1. Refactor `extract_quad_edlines` (`edlines.rs:1119-1165`) to
   compute both `AxisAligned` and `Diagonal` arc sets when
   `cfg.imbalance_gate` is enabled.
2. Add `select_phase1_mode(&arcs_axis, &arcs_diag)` returning the
   mode with lower `max/min` arc ratio (with 0.90 strict-better
   margin and 1.40 `AXIS_IMBALANCE_TRIGGER`).
3. Run `LOCUS_HUB_DATASET_DIR=tests/data/hub_cache cargo test
   --release -p locus-core --features bench-internals --test
   regression_render_tag`.  Expect 640×480 mean RMSE +14.8 % with
   `scene_0049_cam_0000.png` rmse 2.26 in `worst_offenders`.

To reproduce attempt B:

1. Replace `refine_corners_gauss_newton` (`edlines.rs:321-470`) with
   `refine_lines_gauss_newton` taking `[Line; 4]` instead of corners
   and running 4 independent 2×2 GN problems with state `(θ_k, d_k)`.
2. Delete `cholesky_factor_8x8` / `cholesky_solve_8x8` /
   `cholesky_inverse_8x8` / `cholesky_solve_with_factor`.
3. Update the Phase 5 call site (`edlines.rs:1322`) to pass Phase-4
   lines `fl` instead of the intersected corners; fold the
   intersection step into the new function.
4. Per-corner covariance derived from per-line covariance via the
   intersection Jacobian (closed-form 2×4 matrix).
5. Run the same protected suite.  Expect mean RMSE +30–43 % and p99
   rotation +30–100 % at every render-tag resolution.

## §7 Cross-references

- `docs/engineering/edlines_sota_design_2026-05-03.md` — original
  design memo; §5.2 marked as falsified by attempt B.
- `docs/engineering/scene_0008_root_cause_2026-05-03.md` — the
  empirical motivation for both attempts.
- `docs/engineering/edlines_s1_corner_exclusion_2026-05-04.md` —
  prior EdLines architectural-change negative result; same
  synthetic-data-only-evidence caveat.
- PR #281 — outlier-aware corner-drop LM (the upstream gate that
  bounds scene_0008's pose-error contribution).
- PR #282 — board-pose outlier-aware drop on `refine_aw_lm`.
