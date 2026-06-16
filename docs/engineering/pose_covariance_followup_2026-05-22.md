# Pose-Covariance Calibration Follow-Up — 2026-05-22

Negative-result postmortem for the apriltag3-style empirical-covariance work
that started from the 2026-05-03 audit
(`pose_covariance_calibration_audit_2026-05-03.md`). No production code
changes ship from this investigation; this memo records what was tried,
what didn't move, and the one lever that actually can.

## 1. Starting point

The 2026-05-03 audit measured the 6×6 SE(3) covariance returned by
`refine_pose_lm_weighted` on the hub `tag36h11_1080p` corpus:

| Statistic     | Empirical | χ²(6) ideal | Ratio  |
| :------------ | --------: | ----------: | -----: |
| mean d²       | 714.7     | 6.0         | 119×   |
| KL(emp‖χ²(6)) | 13.93     | 0           | —      |
| Per-axis ratio (tx, ty, tz, rx, ry, rz) | 0.026, 0.021, 0.27, 0.094, 0.15, 0.17 | 1.0 | — |

Verdict: `Σ` is ~119× too tight on the joint metric, with 99 % of the
miscalibration concentrated in the two stiffest principal axes.

The apriltag3 brief from the same week proposed an empirical-covariance
inflation: scale `Σ_ℓ` by `max(1, MSE_emp / σ²_th)` before propagation. This
memo documents the three concrete attempts at adapting that brief to Locus.

## 2. What was tried, and why none of it shipped

### Phase 1 — GWLF empirical inflation (`refine_quad_gwlf_with_cov`)

Inflate the per-side `cov_l` by `max(1, (λ_min/W) / theoretical_variance_px2)`
inside the GWLF eigendecomposition. Faithful to the apriltag3 brief; the
empirical perpendicular MSE falls out of the existing weighted moment
solve for free.

**Why it didn't ship:** none of the four shipped profiles
(`standard`, `grid`, `high_accuracy`, `max_recall_adaptive`) routes
candidates through `CornerRefinementMode::Gwlf`. `apply_detector_gwlf`
short-circuits via `any_route_uses_gwlf()` and the inflation code is
unreachable. Verified by flipping the flag on `high_accuracy`'s GWLF-route
test fixtures — every protected snapshot stayed byte-identical because
the structure-tensor covariance is sourced from `pose_weighted.rs`, not
`gwlf.rs`. The audit was run against the ERF/structure-tensor path; the
GWLF lever can't move it.

### Phase 2 — pre-LM scalar multiplier on `Σ_c`

Multiply the structure-tensor corner covariance `Σ_c = σ_n²·S⁻¹ + α·I`
by a uniform per-profile scalar `m` before the weighted-LM solve. The
probe showed that on real corners the Tikhonov term `α·I` dominates the
trace by ~5 orders of magnitude (`trace(σ_n²·S⁻¹) ≈ 4e-6` vs
`trace(α·I) ≈ 0.18`); scaling `σ_n²` alone barely moves anything, so
the lever has to cover the *whole* `Σ_c`.

Calibration sweep at `m = 250` on `high_accuracy`:
`mean d² 714.7 → 6.45`, `KL 13.9 → 5.08`. **But it regressed render-tag
p99 rotation from 0.86° → 1.90° (+120 %).** The LM's inflated weights
let IPPE branch-ambiguity outliers through the pose-consistency gate.
Per the project's "never trade render-tag tail" rule
(`feedback_dataset_priority`), this lever is empirically harmful.

### Phase 3 — post-LM per-axis 6×6 congruence

Apply `Σ_new[i,j] = √(m_i·m_j)·Σ[i,j]` to the 6×6 LM output, where
`m = [tx, ty, tz, rx, ry, rz]`. PSD- and correlation-coefficient-
preserving. The pose value is computed before the inflation runs, so by
construction it cannot regress any tracked metric (corner RMSE, p99
rotation/translation, recall, latency).

Audit-derived `m = [38.75, 47.43, 3.65, 10.65, 6.51, 5.92]` (the
empirical per-axis ratios from the baseline run) gives
`mean d² 714.7 → 83.4`, `KL 13.9 → 6.3`. **But KL stays well above the
"well-calibrated" gate of 0.5**, and no protected suite moves because the
pose value is unchanged.

Sweeping per-axis values that calibrate each diagonal ratio to exactly
1.0 (i.e. the audit's stated target) *worsens* the joint metric:
`mean d² 9188`, `KL 27.5`. The diagonal lever is in irreconcilable
conflict with the joint Mahalanobis when off-diagonals are wrong.

## 3. The structural finding

**No global multiplicative scheme can bring KL below ~3 on this dataset.**

A diagonal congruence `D·Σ·D` preserves the correlation matrix and only
rescales per-axis variances. The audit's evidence is that the
correlation matrix of the LM-output `Σ` is wrong — empirical
`δᵢ`-components are correlated differently from what `(JᵀWJ)⁻¹`
predicts. Heavy-tailed scene-specific outliers in d² show this clearly:
some scenes have d² in the thousands while the median sits below 1,
which cannot be calibrated by any monotonic rescaling.

Mechanically, the only lever left is **per-scene / per-corner
adaptivity** — letting the covariance reshape itself based on local
image evidence rather than a single global table.

## 4. Recommended next step

> **Status (2026-05-31): Empirically falsified.** Phase 4 was built and
> measured end-to-end on branch `feat/activate-per-corner-empirical-noise`
> (PR #290). The §4 gate below was **KL < 0.5**; Phase 4 leaves KL at the
> baseline **13.93**, unchanged to ±0.02 even at the theoretical upper bound
> (`ε` = per-corner GT residual²). It also fires on nothing in production —
> `corner_empirical_noise` is all-zero on every shipped profile because the
> hub corpus has no low-PPB tags to route through ERF. Not shipped; plumbing
> stays recoverable on the branch. Full analysis:
> `pose_covariance_phase4_postmortem_2026-05-31.md`. **All four phases are now
> closed — there is no known surviving covariance-calibration lever.**

**Phase 4 — per-corner ERF residual MSE plumbing.**

This is the original apriltag3 idea applied honestly to Locus's
production code path:

1. Modify `refine_accumulate_optimized` in `crates/locus-core/src/edge_refinement.rs`
   to also accumulate `Σ residual²` and `sample_count` alongside the
   existing `sum_jtj` / `sum_jt_res`. ~10 lines in the SIMD inner loop;
   one extra `_mm256_fmadd_pd`.
2. Store the converged MSE on `ErfEdgeFitter` (new field `last_residual_mse`).
3. In `crates/locus-core/src/refinement.rs::refine_corner`, combine the two
   adjacent edges' MSE into a per-corner empirical noise estimate.
4. Add a new SoA column on `DetectionBatch` (`corner_empirical_noise:
   [[f32; 4]; MAX_CANDIDATES]`).
5. Plumb that into `compute_framework_uncertainty`. Inside
   `finalize_corner_covariance`, replace the constant `sigma_n_sq` with
   `max(sigma_n_sq, empirical_per_corner_noise)`.

Estimated diff: ~200 LOC, touches AVX2 / AVX-512 / NEON multiversion
code (highest-stakes section per `docs/engineering/constraints.md`).
Risk: high — SIMD edits are unforgiving. Reward: the only lever the
2026-05-21 sweep didn't rule out.

A Phase 4 PR should land with:

- An SIMD unit test pinning the new `Σ residual²` accumulator
  against a scalar reference.
- A re-audit confirming KL drops below 0.5 (the gating threshold) on
  the hub `tag36h11_1080p` corpus.
- Zero regression on the four protected suites — same constraint as
  Phase 3, harder to honor because per-corner reweighting *can* change
  the LM solution.

## 5. Anti-pattern entry (do-not-re-attempt)

For the auto-memory index and future agents:

- **Pre-LM scalar `Σ_c` multiplier**: empirically harmful (`m=250`
  regresses render-tag p99 rotation 0.86° → 1.90°). Do not re-attempt
  unless the LM's IPPE branch-ambiguity behaviour is also re-evaluated.
- **Post-LM diagonal `Σ` congruence (any flavour — uniform scalar,
  per-axis vector, trans/rot pair)**: KL bottoms at ~3 because the
  off-diagonal correlation structure of `(JᵀWJ)⁻¹` is wrong, not just
  its scale. Do not re-attempt without changing the correlation
  structure (which a diagonal lever cannot).
- **GWLF empirical inflation in `refine_quad_gwlf_with_cov`**: mechanism
  is sound but no shipped profile routes through GWLF refinement, so
  any "performance improvement" claim must first ship a profile that
  actually exercises that path.
- **Phase 4 — per-corner ERF residual MSE inflation** (added 2026-05-31):
  falsified. KL stays at the baseline 13.93 even at the theoretical upper
  bound — a per-corner *diagonal* `σ_n²` rescale cannot fix the *off-diagonal*
  correlation structure of `(JᵀWJ)⁻¹` (same wall as the post-LM diagonal
  lever above). Also dormant in production: the empirical column is all-zero
  on every shipped profile because the hub corpus has no low-PPB tags to
  route through ERF. Do not re-attempt without **both** a corpus where ERF
  fires **and** a non-diagonal correlation-structure fix. See
  `pose_covariance_phase4_postmortem_2026-05-31.md`.

## 6. Diff that was prepared and rolled back

The investigation produced ~580 lines of mechanism (`DetectorConfig`
fields, FFI wiring, Pydantic mirror, JSON profile entries, tests, schema,
audit-script hooks) for two knobs that ended up at byte-identical
defaults. None of it is on `main`. The exploration branch was
`feat/gwlf-empirical-cov-inflation`; recover via `git show` if a future
Phase 4 wants to copy the FFI/config plumbing template.
