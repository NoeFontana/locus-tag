# EdLines SOTA experiments — empirical results (2026-05-03)

Empirical follow-up to `edlines_sota_design_2026-05-03.md`. Tested two
small-surface variants of the proposed improvement #1 on
`scene_0008_cam_0000` and the full 50-scene
`locus_v1_tag36h11_1920x1080` corpus. Both confirm the design memo's
diagnosis: **the dominant failure mechanism is Phase 5's chord-locked
GN cost (F4), not Phase 3 sub-pixel error (F1) or the exclusion zone
alone (F5)**.

## §1 Verified hardware

AMD EPYC-Milan KVM, 8 logical CPUs, AVX2/FMA/F16C. `--release` build
with `bench-internals`. CPython 3.14.3, rustc 1.92.0.

## §2 Experiment 1 — iterate Phase 3 / Phase 4 with refined-line trajectory

Implemented per design memo §5.1 (~85 LoC in `edlines.rs`):

- Phase 3 v1 walks the binary IRLS line (existing).
- Phase 4 v1 fits sub-pixel line (existing).
- **NEW Phase 3 v2** walks the v1 sub-pixel line in gray space (`dec=1.0`).
- **NEW Phase 4 v2** re-fits to v2 sub-pixel points; falls back to v1 line
  when too few points.
- Phase 5 unchanged, consumes v2 lines and v2 sub-pixel points.

### §2.1 Result

| Metric | Baseline | Iterated | Verdict |
|---|---|---|---|
| scene_0008 corner 1 ‖r‖ | 3.833 px | **4.086 px** | **Worse** |
| scene_0005 d² | 6 344 | **NaN (Cholesky failure)** | **Broke** |
| Corpus mean ‖r‖ | 0.191 px | regressed | Worse |
| Corpus KL | 13.93 | 24.34 (with NaN scenes) | Worse |

scene_0005's baseline was the second-worst scene (d² = 6 344). With the
iteration the LM solver hit `convergence = 3` (Cholesky failure) in 3
iterations, producing NaN pose covariance and 1.2-meter translation
error. Corner residuals on scene_0005 jumped from ~0.5 px (baseline) to
~1.0–1.85 px per corner — corpus-wide regression.

### §2.2 Why it failed

The hypothesis was that re-running Phase 3 on the sub-pixel line
trajectory would centre the parabolic fit window on the true edge,
eliminating clamp-induced bias.

**Empirically false.** The mechanism that breaks it:

- For typical scenes (residuals ~0.14 px), Phase 3 v1 already produces
  sub-pixel points well-centred on the gray edge. Phase 4 v1's line is
  accurate to ~0.06 px RMS (verified for scene_0008's left edge in the
  root-cause memo).
- Phase 3 v2 walks along this v1 line. Bbox projection in gray space
  produces a t-range that matches v1's. Probes are placed at edge-centred
  positions.
- **But:** subtle changes in v2 sub-pixel point distribution (different
  point ordering, slightly different exclusion at bbox edges, slightly
  different gradient-threshold behaviour at edge-centred probes) shift
  Phase 4 v2's IRLS fit by ~0.05 px relative to v1.
- This shift compounds through Phase 5's chord-locked GN: the cost
  gradient pulls corners along directions that depend on the chord, which
  depends on the corners. Small initial perturbation amplifies.

The amplification is only mild on most scenes (corpus mean ‖r‖ regresses
~0.05 px) but catastrophic on scene_0005, where baseline corner residuals
of 0.5 px sit close to a Phase-5 stability boundary. Pushing them by
~0.1 px tips the LM solver into Cholesky failure.

This is a textbook case of **Phase 5's chord-locked GN being
ill-conditioned**: small perturbations in inputs produce non-monotonic
output changes. The Tikhonov λ = 1e-6 is far too small to stabilise
this for non-trivial inputs.

## §3 Experiment 2 — shrink Phase 5 exclusion zone (5% → 1%)

A one-line change to `refine_corners_gauss_newton`:
```rust
// Was: if !(0.05_f64..=0.95_f64).contains(&alpha) { continue; }
   Now: if !(0.01_f64..=0.99_f64).contains(&alpha) { continue; }
```

Hypothesis (memo §4 F5): when Phase 4's corner is wrong, the 5%
exclusion zone — computed against the *current* (wrong) chord —
excludes points that lie near the *true* corner. Shrinking to 1%
includes those points so they can constrain the GN gradient back
toward the right configuration.

### §3.1 Result

| Metric | Baseline | 1% exclusion | Δ |
|---|---|---|---|
| scene_0008 corner 1 ‖r‖ | 3.833 px | 3.657 px | **−0.18 px (−4.6 %)** |
| scene_0008 d² | 12 405 | 11 999 | −3.3 % |
| Corpus mean ‖r‖ | 0.191 px | 0.187 px | −2.1 % |
| Corpus p99 ‖r‖ | 0.697 px | 0.654 px | −6.2 % |
| Corpus mean d² | 715 | 769 | **+7.6 % (worse)** |
| Corpus p99 d² | 9 435 | 9 675 | +2.5 % |

**Two-sided result.** Corner-residual metrics improve 2–6 % across
the board (F5 is real and produces measurable bias). Pose-d² metrics
*regress* by 3–8 % — the closer-to-corner points in the GN cost
contribute more bias to corner localisation than to pose accuracy in
some interaction we did not predict.

### §3.2 Why it doesn't fix scene_0008

scene_0008's corner-1 error stays at ~3.7 px (vs the 0.5 px target).
The exclusion-zone fix recovers ~0.2 px (4.6 %); the remaining 3.5 px
is from the chord-direction lock (F4). **F5 is real but minor; F4
is the dominant mechanism.**

## §4 Conclusion

The two cheap interventions tested both confirm the design memo's
ranking: F4 (Phase 5 chord lock) is the dominant root cause of
scene_0008. F5 (exclusion zone) contributes ~5 %; F1 (Phase 3 clamp)
contributes negligibly given current per-scene corner accuracy.

**No small-surface fix exists for scene_0008.** The fix requires either:

- Improvement #2 — full re-parameterisation of Phase 5 with line-DoF
  state vector (~3 days; medium risk to 49 typical scenes via snapshot
  rebless).
- Improvement #3 — replace Phase 3's parabolic fit with ERF model
  (~5 days; high risk per Phase C.5 negative-result memo).

Both remain on the table architecturally, but the empirical case for
investing on synthetic data alone is weak:

1. scene_0008's failure mode (near-axis-aligned tag at ~90° rotation)
   may or may not appear at meaningful rates on real cameras.
2. The 49 typical scenes in this corpus have corner residuals at the
   noise floor (~0.14 px isotropic). A Phase 5 re-parameterisation that
   helps scene_0008 must not regress them — a real risk because the
   line-DoF parameterisation changes the Phase 5 convergence basin for
   *all* scenes.
3. **The runtime gate (Track A's `corner_geometry_outlier`) already
   detects scene_0008** and offers a ~4 h fix path (downweight failing
   corner / inflate Σ_pose / reject detection).

## §5 Recommendation

**Defer #2 and #3 until either:**

1. **Track C real-camera audit** shows scene_0008-class failures occur
   at non-trivial rates on real data, OR
2. The runtime gate proves insufficient (e.g., a multi-tag scene where
   one tag fires the gate but its measurement is needed for board-level
   pose).

In the interim, ship the runtime gate (~4 h) as the practical fix.

## §6 What this PR closes

- The empirical hypothesis-testing chain on scene_0008 (Path B-2 →
  Path α → root-cause → SOTA design → these experiments).
- The case for revisiting EdLines Phase 5 architecture *on synthetic
  data alone*.

What it does NOT close:

- The validity of #2 / #3 as architectural improvements *if real-camera
  evidence emerges*.
- The runtime-gate work for `corner_geometry_outlier` (separate small
  track).
