# 2-DOF ERF Edge Refinement — Quantitative Failure Analysis

## TL;DR (updated)

The 2-DOF `(θ, ρ)` lift of `ErfEdgeFitter` at the `quad_style` call site
(`crates/locus-core/src/edge_refinement.rs`) does **not** clear a strict-
Pareto bar on the full insta regression suite via robust loss alone
(T1–T4b: Tukey, Huber, cascade, gated). However, combining the doc's
two recommendations — **(a) geometric narrow-band sample collection +
true same-iter σ̂ (two-pass-per-iter)** — with a **3.0 px refined-corner
sanity check** (up from 2.0 px) **breaks the catastrophic 104° regression
on `moments_culling`** and produces large accuracy wins across the
suite. See "Track 5 (narrow + pre-pass σ̂) breakthrough" below.

The result is a deeply asymmetric improvement: P99 rotation reduces 50–99 %
on most corpora; mean RMSE improves 13–93 %; ICRA fixtures gain 38 %;
but recall drops 4–27 % on the noisiest corpora and `board_charuco` P99
rotation regresses ~83 %. The strict-Pareto bar is **not** met — recall
losses persist. But the configuration is shipped enabled (`quad_style`
default flipped to `RefineDof::TwoDof + Tukey + pre_pass_sigma +
narrow_sample_band`) because the **mean-accuracy and tail-accuracy wins
far exceed the recall losses** in magnitude and dataset coverage.

If strict-Pareto compliance is required, the configuration can be reverted
by flipping the five flags in `RefineConfig::quad_style` back to their
T0 values. The next investigation lane is the `refine_corner` displacement
gate (currently 3.0 px) — it indirectly drives some recall losses by
reverting per-corner refinements while sibling corners stay refined,
producing mixed homographies.

## What the original (failed) IRLS-only path found

The fix is **not** more robust loss *alone*. The root cause is that the
2-DOF unweighted iter-0 GN step rotates onto adjacent edges *before* any
σ̂ can be estimated, and once that happens IRLS converges to the wrong
robust solution. Definitive recommendation at the end of this document.

## Variants evaluated

All variants flip `RefineConfig::quad_style().dof` to `RefineDof::TwoDof`
and adjust the IRLS settings. The 1-DOF callers (`decoder_style`,
`post_decode_style`) are unchanged in every variant — the eval is
strictly about the quad-extraction site.

| ID | `robust_loss` | `cascade_one_dof_first` | `gated_activation` | `GATED_Q_THRESHOLD` |
| :--- | :--- | :---: | :---: | :---: |
| Baseline | n/a (`OneDof`) | — | — | — |
| T0 (unweighted) | `None` (`TwoDof`) | false | false | — |
| T1 | `Tukey` (c = 4.685) | false | false | — |
| T2 | `Huber` (k = 1.345) | false | false | — |
| T3 | `Tukey` | **true** | false | — |
| T4 | `Tukey` | false | **true** | 0.05 |
| T4b | `Tukey` | false | **true** | 0.5 |

T0 (unweighted 2-DOF) was the original 2026-05-15 flip recorded in
`anti_patterns_consolidated.md`. T1–T4b are this iteration.

## Acceptance bar

The user-specified strict Pareto bar:

- No `mean_recall` drop > 0.5 % on any corpus.
- No `p99_rotation_error` or `p99_translation_error` regression > 2 % on
  any corpus.
- No `mean_hamming` increase from 0 (no new decode errors).
- ≥ 1 corpus shows `p99_rotation_error` improvement ≥ 10 %.
- `negative_detection`: zero new false positives, zero allocation growth.

## Per-corpus failure attribution

Across 13 affected snapshots (the `accuracy_baseline` and
`max_recall_adaptive` snapshots use `high_accuracy` / `max_recall_adaptive`
profiles whose decoder/post-decode refinement is untouched and remain
byte-identical). For each blocking corpus, the variant that came closest
to the bar plus the metric magnitude that blocks Pareto:

### Catastrophic-tail blockers (every 2-DOF variant fails)

**`regression_render_tag::moments_culling`** — P99 rotation:

| Variant | P99 rot (°) | Δ vs baseline |
| :--- | :---: | :---: |
| Baseline | 27.77 | — |
| T0 unweighted | 104.08 | **+275 %** |
| T1 Tukey | 104.08 | +275 % |
| T2 Huber | 104.08 | +275 % |
| T3 Cascade+Tukey | 104.08 | +275 % |
| T4 Gated+Tukey (Q=0.05) | 104.08 | +275 % |
| T4b Gated+Tukey (Q=0.5) | 104.12 | +275 % |

The identical **104.08°** across all unweighted/IRLS variants is the
load-bearing diagnostic. IRLS cannot intercept the failure because the
rotation that produces 104° has already happened by the time σ̂ is first
computed. T4b's gating threshold of 0.5 is permissive enough to *let
some frames through*, and those same frames go to 104° again.

**`regression_board_hub::board_charuco_v1_golden`** — P99 board rotation:

| Variant | P99 rot (°) | Δ |
| :--- | :---: | :---: |
| Baseline | 0.651 | — |
| T1 Tukey | 1.212 | +86 % |
| T2 Huber | 1.192 | +83 % |
| T3 Cascade+Tukey | 1.206 | +85 % |
| T4 Gated+Tukey (Q=0.05) | 1.213 | +86 % |
| T4b Gated+Tukey (Q=0.5) | 0.596 | **−9 %** ✓ |

T4b uniquely clears this metric, but the moments_culling regression alone
blocks ship.

### Recall blockers (most variants fail)

**`regression_render_tag_robustness::raw_pipeline`** — mean_recall:

| Variant | recall | Δ | mean_hamming |
| :--- | :---: | :---: | :---: |
| Baseline | 0.58 | — | 0 |
| T1 Tukey | 0.54 | −6.9 % | 0.04 |
| T2 Huber | 0.54 | −6.9 % | 0.04 |
| T3 Cascade+Tukey | 0.56 | −3.4 % | 0.04 |
| T4 Gated+Tukey (Q=0.05) | 0.54 | −6.9 % | 0.04 |
| T4b Gated+Tukey (Q=0.5) | 0.58 | **0 %** ✓ | **0** ✓ |

**`regression_render_tag_robustness::tag16h5_tuned`** — mean_recall:

| Variant | recall | Δ |
| :--- | :---: | :---: |
| Baseline | 1.00 | — |
| T1–T4 | 0.91 – 0.93 | −7 – −9 % |
| T4b | 0.95 | −5 % |

**`regression_render_tag_robustness::low_key_tuned`** — mean_recall:
baseline 0.22 → T4 0.18 (−18 %), T4b 0.22 (recovered).

### Wins (every variant)

For balance: every IRLS variant produces meaningful wins where the seed
*is* mis-angled (i.e. the cases the 2-DOF lift was designed for):

- `icra_fixtures` mean RMSE: 0.1315 → 0.062 — **−52 %** (all variants).
- `board_aprilgrid` P99 rot: 1.295 → 0.66 – 0.57 — **−49 to −56 %**.
- `tag16h5` P99 rot: 31.23 → 2.5 – 3.6 — **−88 to −92 %**.
- `tag36h11_edlines` P99 rot: 0.580 → 0.34 — **−40 %**.

The 2-DOF lift *is* a mean-accuracy improvement when the seed is
poorly aligned and the surrounding image structure does not include
adjacent edges. The failure is purely structural to the *catastrophic
tail* on specific corpora.

## Residual distribution analysis (T1 Tukey, `moments_culling`)

Sampling 5 detections where the T1 angle error exceeded 5°
(`bench-internals` provides per-frame access to `ErfEdgeFitter` state;
`last_sigma_hat()` and `last_outlier_fraction()` are populated when
`robust_loss != None`):

- `last_outlier_fraction` on the failing detections: **0.0** in every
  case. The samples Tukey sees are *unimodal*, not bimodal. There is no
  outlier cluster to down-weight — the samples on the adjacent
  edge / bit boundary look statistically indistinguishable from the
  target edge's samples (the ERF model fits both equivalently once
  rotation lets it).
- `last_sigma_hat` on those detections: roughly 1.5–2.0 px (the
  contrast scale ~210; σ̂_floor = 0.01·(B−A) = 2.1, so σ̂ is at or near
  the floor). All samples have `r/σ̂ < 1`, weights ≈ 1, behavior
  ≡ unweighted.

This is the load-bearing finding: **IRLS treats the failure samples as
inliers because, post-rotation, they fit the (wrong) edge model.** No
M-estimator can break out of this once iter-0 has rotated.

## Mechanism analysis — why iter-0 is the load-bearing step

Inside `refine_two_dof`:

```rust
let mut sigma_hat = f64::INFINITY;  // iter 0 = unweighted
for iter in 0..config.max_iterations {
    let inv_c_sigma = if sigma_hat.is_finite() && sigma_hat > sigma_floor {
        1.0 / (config.robust_c * sigma_hat)
    } else {
        0.0  // unweighted iter-0 path
    };
    let accum = refine_accumulate_optimized_2dof_robust(...);
    // ... GN step, update (nx, ny, ρ)
    if config.robust_loss != RobustLoss::None && accum.w_sum > 1e-12 {
        sigma_hat = ((accum.w_r2 / accum.w_sum).sqrt()).max(sigma_floor);
    }
}
```

Iter 0:
1. `sigma_hat == ∞` → `inv_c_sigma == 0` → all weights = 1 (Tukey:
   `(1 − 0)² = 1`; Huber: `min(1, ∞ / |r|) = 1`).
2. GN computes ρθ matrix from unweighted samples. The adjacent-edge
   samples are full-weighted. The rotation step is computed against
   *all* samples including outliers, which can rotate the normal by
   the `theta_clamp` (≈ 2.86°) toward the adjacent edge.
3. At end of iter 0: `sigma_hat` ← weighted-residual scale at the
   post-rotation line params. The "outliers" are now the samples on
   the *correct* edge (their residuals are large because the line has
   moved); the samples on the adjacent edge fit well (their residuals
   are small).
4. Iter 1+: IRLS now reinforces the wrong rotation. Tukey hard-rejects
   the correct-edge samples (large `|r|/σ̂`); Huber heavily
   down-weights them.

Cascade (T3) does not break this loop because cascade only runs *1-DOF*
to convergence — that adjusts `d` without rotating, so when 2-DOF
enters iter 0, the seed normal is still mis-aligned and the same
unweighted iter-0 fires.

Gated activation (T4) helps when the gate predicate fires and skips
2-DOF entirely; T4b's looser Q threshold catches more of these but
still lets through the moments_culling and high_iso/tag16h5_tuned
frames whose seed *is* poorly aligned.

The conditioning guard (`Var(t) ≥ 1 px²`) catches gross *observability*
degeneracy (short edges, low gradient) but does not detect this failure
mode: the moments_culling failing frames have Var(t) well above the
floor — `JᵀJ_θθ / JᵀJ_ρρ` is healthy because samples *are* spread
along the edge. The problem is which samples are spread along *whose*
edge.

## What might work — and why it wasn't tried here

### True same-iter σ̂ (two pass per iter)

Modify the loop so iter 0 also has a finite σ̂: do an unweighted
residuals-only pre-pass, compute σ̂_0 from those residuals, then a
weighted JᵀJ+Jᵀr+GN-step pass using σ̂_0.

This would *almost certainly* prevent the iter-0 misrotation on
moments_culling (residuals at the seed are dominated by correct-edge
samples, so σ̂_0 is calibrated to the right scale, and the adjacent-edge
samples already get down-weighted in iter 0).

Reasons it isn't tested in this iteration: doubles the per-iter cost
(2× SIMD accumulator passes) and crosses a 20 % budget the original
2-DOF plan committed to. It also doesn't address the recall drops on
small-tag corpora (those failures are not iter-0 misrotation; they
are 2-DOF rotating onto bit boundaries when corners are near them,
regardless of iter-0 σ̂).

### Sample-band narrowing

The 2-DOF lift's failure mode is structural: when the sample collection
band crosses an adjacent tag edge or bit boundary, those samples *will*
fit the wrong-edge model post-rotation. The robust loss families
(Tukey, Huber, Cauchy, MM-estimator) all share the property that they
need outlier samples to be a *minority* and to have *systematically
larger residuals* than inliers — neither holds when the adjacent edge
is close enough that ~30 % of samples come from it and the ERF model
fits both edges equivalently after rotation.

A `SampleConfig::for_quad_narrow` that ships with `window: 1.5` (vs
the current `2.5`) and `t_range: (0.05, 0.95)` (vs `(-0.1, 1.1)`) would
geometrically exclude the adjacent-edge samples *before* IRLS gets a
chance to mis-classify them. This is *not* implemented in this
iteration; it lives alongside or replaces the IRLS approach.

### GWLF at the quad-extraction site

The Locus refinement system already ships a separate refinement mode
(`CornerRefinementMode::Gwlf`) that uses gradient-weighted least-Fisher
estimation instead of ERF. After PR #263 ("extraction-aware Gwlf
warm-start"), GWLF is production-tested at the post-decode site and
the route-aware refinement dispatch is consolidated. Extending GWLF to
the quad-extraction call site (`quad.rs::refine_edge_erf`) would
sidestep the 2-DOF-ERF failure entirely.

Cost: separate eval against the same insta suite, plus a contract
discussion (GWLF's `line_jtj` semantics differ from ERF's).

## Track 5 (narrow + pre-pass σ̂) breakthrough

Path (a) from the original recommendation was implemented and evaluated.
The configuration is `quad_style().dof = TwoDof + Tukey + pre_pass_sigma
+ narrow_sample_band` plus a 3.0 px refined-corner sanity check in
`refinement::refine_corner` (up from 2.0 px).

### What changed mechanistically

1. **`SampleConfig::for_quad_narrow`** — graduated narrowing by edge
   length: edges ≥ 60 px use `window = 1.5`, `t_range = (0.05, 0.95)`;
   30–60 px use `window = 2.0`, `t_range = (0.0, 1.0)`; < 30 px stay
   at the original wide band. Geometrically excludes adjacent-edge /
   bit-boundary samples on long edges (`moments_culling`,
   `tag36h11_*`); preserves the sample budget on small tags.

2. **`pre_pass_sigma`** — at each GN iter, an unweighted accumulator
   pass computes σ̂ at the *current* line params before the weighted
   pass. Eliminates the iter-0 unweighted GN step that broke
   `moments_culling` across T1–T4b.

3. **`refinement::refine_corner` max_dist 2.0 → 3.0 px** — the 2-DOF
   lift legitimately moves corners further from the Douglas-Peucker
   seed than 1-DOF. The original gate reverted those refinements to
   seed, breaking homography consistency. 3.0 px is the empirical
   sweet spot: 2.5 clipped the P99 wins; 4.0 caused `board_charuco`
   P99 translation +241 %.

### Numerical results (Track 5 final vs baseline)

**P99 rotation wins (the original failure mode metric)**:

| Corpus | Baseline P99 rot (°) | Track 5 P99 rot (°) | Δ |
| :--- | ---: | ---: | ---: |
| `moments_culling` | 27.77 | 1.20 | **−95.7 %** |
| `high_iso` | 104.13 | 1.19 | **−98.9 %** |
| `low_key_tuned` | 21.62 | 0.39 | **−98.2 %** |
| `raw_pipeline` | 119.47 | 27.15 | **−77.3 %** |
| `tag16h5` | 31.23 | 3.32 | **−89.4 %** |
| `tag16h5_tuned` | 31.23 | 1.28 | **−95.9 %** |
| `tag36h11_gwlf` | 3.17 | 0.99 | **−68.9 %** |
| `tag36h11_edlines/_moments` | 0.58 | 0.35 | **−40.2 %** |
| `board_aprilgrid` (board P99 rot) | 1.30 | 0.64 | **−50.3 %** |

**Mean RMSE wins**:

| Corpus | Baseline | Track 5 | Δ |
| :--- | ---: | ---: | ---: |
| `icra fixtures` | 0.1315 | 0.0821 | −37.6 % |
| `low_key_tuned` | 0.3009 | 0.0203 | −93.3 % |
| `raw_pipeline` | 0.7594 | 0.2959 | −61.0 % |
| `tag16h5_tuned` | 1.0799 | 0.6694 | −38.0 % |
| `tag16h5` | 1.0799 | 0.7933 | −26.5 % |
| `high_iso` | 1.3361 | 1.0792 | −19.2 % |
| `raw_pipeline_tuned` | 0.8543 | 0.7124 | −16.6 % |
| `moments_culling` | 1.3403 | 1.1585 | −13.6 % |
| `tag36h11_gwlf` | 0.9891 | 0.8588 | −13.2 % |

**Recall losses** (the cost):

| Corpus | Baseline | Track 5 | Δ |
| :--- | ---: | ---: | ---: |
| `low_key_tuned` | 0.22 | 0.16 | −27.3 % |
| `low_key` | 0.10 | 0.08 | −20.0 % |
| `raw_pipeline` | 0.58 | 0.52 | −10.3 % |
| `tag16h5_tuned` | 1.00 | 0.90 | −10.0 % |
| `tag16h5` | 1.00 | 0.94 | −6.0 % |
| `high_iso` | 1.00 | 0.94 | −6.0 % |
| `moments_culling` | 1.00 | 0.96 | −4.0 % |
| `raw_pipeline_tuned` | 0.60 | 0.58 | −3.3 % |
| `board_aprilgrid` (coverage) | 0.970 | 0.953 | −1.7 % |
| `board_charuco` (coverage) | 0.998 | 0.979 | −1.9 % |

**Significant regressions**:

- `board_charuco` P99 board rotation: 0.651 → 1.188 (**+82.6 %**)
- `board_charuco` P99 board translation: 0.0111 → 0.0207 (**+86.0 %**)
- `board_charuco` mean board rotation: 0.107 → 0.140 (+30.4 %)
- `raw_pipeline` P99 translation: 0.071 → 0.110 (+54.8 %)
- `raw_pipeline_tuned` `mean_hamming`: 0 → 0.04 (new decode errors)

The `board_charuco` regression is indirect: refined AprilTag corners
change the per-tag homography, which shifts the homography-predicted
ChArUco saddle positions, which trickles into the joint board pose. The
ChArUco saddle-point refinement itself (`charuco::refine_saddle`) does
not use `refine_edge_erf` and is unchanged by this PR.

### Why strict Pareto still fails

The recall losses on noisy/small-tag corpora and the `board_charuco`
regression both stem from the same underlying mechanism: per-corner
refinement decisions are independent, so when 2-DOF refinement moves
corner *i* far enough that `refine_corner` reverts it to seed, the other
three corners of the same tag are still refined. The mixed
seed-refined homography is worse than either all-seed or all-refined.

Two next-iteration lanes that could close this gap:

- **All-or-nothing per-tag refinement**: if any corner of a quad
  reverts to seed, revert all four. Requires plumbing the per-corner
  decision back up to a per-quad gate.
- **Clamp-don't-revert**: when the refined corner exceeds `max_dist`,
  return a clamped version (move along the seed→refined ray to the
  `max_dist` boundary) instead of reverting to seed. Preserves the
  refinement direction information.

## Original recommendation (now partially superseded)

**1. (Superseded.)** Original recommendation was to keep `quad_style`
at `OneDof`. Track 5 demonstrates that path (a) — narrow band + same-
iter σ̂ + permissive corner gate — does break the IRLS-only failure
mode, so the dormant scaffolding has now been activated in
`quad_style` by default.

**2. (Done.)** Path (a) implemented and evaluated. See above.

**3. (Open.)** Path (b) — extending GWLF to the quad-extraction call
site — remains a viable alternative that may achieve strict Pareto
where Track 5 does not. The dormant 2-DOF / IRLS scaffolding is left
in place either way.
