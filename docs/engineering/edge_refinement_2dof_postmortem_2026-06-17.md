# 2-DOF Tukey IRLS quad-corner refinement — Postmortem — 2026-06-17

Negative-result postmortem for **PR #264** (`feat/erf-2dof-tukey-narrow-band`),
which lifted quad-extraction corner refinement from 1-DOF (ρ-only) to **2-DOF
(θ, ρ) Tukey IRLS**. The kernel is sound and the pose-tail wins are large and
real, but the change cannot meet the project's strict-Pareto bar, and there is
no profile/route boundary that isolates its wins from its losses. **Production
stays on the proven 1-DOF refinement.** The hardened implementation is preserved
on the branch (not deleted) for a future re-attempt or the GWLF alternative.

This follows the docs-only negative-result pattern of
`edlines_sota_followup_postmortem_2026-05-25.md` and
`pose_covariance_phase4_postmortem_2026-05-31.md`: the learning is recorded
here, the code stays recoverable in git, and `main` stays clean.

## TL;DR

- 2-DOF refinement fixes badly-misrotated Douglas-Peucker corner seeds and cuts
  the **p99 rotation error by 68–98 %** on several corpora (and mean RMSE by
  13–54 %, ICRA RMSE −54 %). That part works.
- It also **trades protected metrics**: decode-recall on noisy/small-tag
  corpora, p99-**translation** on render-tag, and — structurally — the ChArUco
  board pose, because refined AprilTag corners shift the per-tag homography that
  predicts saddle search locations.
- Universal hardening was developed and **kept on the branch** (the salvage),
  framing the rotation as a 1-DOF-vs-2-DOF nested-model selection:
  1. **Deadband** — accept the 2-DOF rotation only past ~0.4°, else keep the
     proven 1-DOF ρ-only solve. Fixes a real correctness regression (a
     perfectly-aligned control corner degraded 0.05 px → 0.16 px) and keeps the
     clean render-tag hub byte-identical.
  2. **Robust, fair, degenerate-safe acceptance** — score the 1-DOF reference
     and the 2-DOF candidate by the same bounded-influence cost over the same
     full sample set, and keep the rotation only if it beats 1-DOF by a margin.
     (Replaces an earlier gate that could accept an empty-band "perfect 0.0",
     score a robust fit with an unweighted residual, or be gamed by shedding
     samples — all found in code review and fixed here.)
- Even with the hardening, **the strict-Pareto bar (no recall drop > 0.5 %, no
  p99 regression > 2 % on any corpus) is missed on ~12 corpora**, and the board
  regressions are structural at every gate tuning. **Not shipped.**

## 1. What 2-DOF does, and why it was attempted

The quad-extraction stage seeds each edge from a Douglas-Peucker chord midpoint,
which can be 1–3° off the true edge orientation on blurry/degraded tags. The
shipped 1-DOF ERF refinement solves only the perpendicular offset ρ with the
seed normal frozen, so it cannot correct that rotation — the residual
misorientation propagates (amplified) into the pose, producing rotation-error
tails as large as 104° p99 on the worst corpora.

2-DOF adds the normal rotation θ to the Gauss-Newton solve. The branch's kernel
is the result of a careful T0–T4b ablation: a naive 2-DOF iter-0 unweighted step
rotates onto adjacent edges (a deterministic 104° tail of its own), fixed with a
**same-iter σ̂ pre-pass** + **graduated narrow-band sampling**, an observability
guard, a per-iter rotation clamp, and SIMD-parity accumulators (AVX2/AVX-512/
NEON). This kernel is good engineering and is retained on the branch.

## 2. Problems with PR #264 as it stood

1. **Correctness regression (CI red).** `test_refine_corner_subpixel_accuracy`
   failed: the integer-aligned control corner refined to 0.16 px (threshold
   0.15), worse than 1-DOF's 0.05 px. On a clean, well-aligned edge the θ DOF
   should be a no-op, but the joint solve added a small ρ bias.
2. **It regressed the clean hub** and shipped despite its own doc stating strict
   Pareto was not met ("ships anyway").
3. **Stale numbers** — measured 25+ commits behind main (pose-numerics
   #281/#282/#288, nalgebra/pyo3 bumps).

## 3. The hardening (the salvage, kept on the branch)

Rebased onto current main, then hardened into a clean nested-model selection
(the design after a full code-review pass — see the commits on the branch):

- **Deadband** (`THETA_DEADBAND_RAD ≈ 0.4°`): diagnostics showed the
  clean-corner failure was *not* rotation (θ ≈ 0.08°) but a ρ bias from the
  joint solve. The 2-DOF rotation is accepted only when the converged normal
  rotated past the deadband; otherwise the proven 1-DOF ρ-only result stands.
  Result: control corner back to 0.05 px, and **the entire clean render-tag hub
  (640/720/1080/2160 + primary 1080p) is byte-identical to main**, where the
  original #264 regressed it.
- **Robust, bounded-influence acceptance** (`robust_edge_cost`): the 1-DOF
  reference and the 2-DOF candidate are scored by the *same* aggregate over the
  *same* full sample set — the mean of `min(r², (0.5·|B−A|)²)` — and the
  rotation is kept only if it beats 1-DOF by a margin (`TWO_DOF_ACCEPT_COST_FRAC
  = 0.97`). Scoring over all collected samples (≥10) is degenerate-safe (no
  empty-band "perfect 0.0" can be accepted) and fair (a rotation cannot win by
  shedding high-residual samples out of an in-range band); clipping bounds the
  leverage of off-edge / adjacent-structure outliers so a correct robust
  rotation is not penalised by the contamination it moved away from. By
  construction the path is never worse than 1-DOF on robust edge fit.
- **Supporting fixes** from the review pass: the unweighted and Tukey 2-DOF SIMD
  accumulators were unified into one kernel (the unweighted path passes
  `inv_c_sigma = 0` → weights all 1.0, byte-identical), `sigma_at` gained its
  missing `aarch64+neon` target, the GN convergence test compares the rotation
  step as an edge-tip pixel displacement (not radians-vs-pixels), and the 3.0 px
  `refine_corner` displacement gate is scoped to the ERF path only (the
  gradient-peak path keeps its 2.0 px gate). The same-iter σ̂ pre-pass (the
  load-bearing fix for the 104° iter-0 misrotation) and the per-iteration
  conditioning guard are retained.

This gate is deliberately *more conservative* than a looser residual gate:
because edge-fit cost is an imperfect proxy for pose error, it forgoes a few
marginal pose-only wins (e.g. the gwlf render-tag variant is now byte-identical)
in exchange for correctness — it never accepts a degenerate or unfair
improvement.

## 4. Strict-Pareto measurement (hardened config, on current main)

All four protected suites re-run on current `main` with the hardened 2-DOF
active, vs. the 1-DOF baseline snapshots. Verified hardware: AMD EPYC-Milan,
8 vCPU (2 threads/core), L3 32 MiB, x86_64; `--release --features
bench-internals`; `cargo nextest` (latency redacted in snapshots). Pose mode:
Accurate.

**Wins (representative):**

| Corpus | metric | 1-DOF → 2-DOF | Δ |
| :-- | :-- | :-- | --: |
| moments_culling | p99 rotation | 27.8° → 1.61° | −94.2 % |
| high_iso | p99 rotation | 104.1° → 1.59° | −98.5 % |
| tag16h5 | p99 rotation | 31.2° → 3.23° | −89.6 % |
| tag16h5 | mean RMSE | 1.080 → 0.876 | −18.9 % |
| low_key_tuned | p99 rotation | 21.6° → 0.52° | −97.6 % |
| raw_pipeline | mean RMSE | 0.759 → 0.40 | −47.1 % |
| icra fixtures | mean RMSE | 0.1315 → 0.0645 | −51.0 % |
| icra pure (×2) | mean recall / RMSE | +1.7 % / −10…18 % | improve |

**Persistent strict-Pareto violations (~12 corpora; not tunable away):**

| Corpus | metric | 1-DOF → 2-DOF | Δ |
| :-- | :-- | :-- | --: |
| charuco_refiner | p99 board rot / trans | +76 % / +74 % | regress |
| board_charuco | p99 board rot / trans | +24 % / +48 % | regress |
| board_charuco_forstner | p99 board trans | +140 % | regress |
| board_charuco (all) | mean tag coverage | 0.998 → 0.980 | −1.9 % |
| raw_pipeline | mean recall | 0.58 → 0.52 | −10.3 % |
| high_iso / tag16h5 | mean recall | −6.0 % each | regress |
| moments_culling | mean recall | 1.00 → 0.96 | −4.0 % |
| raw_pipeline / low_key / high_iso | p99 translation | +24 … +55 % | regress |
| raw_pipeline_tuned | mean hamming | 0 → 0.04 | new decode errors |
| edlines / edlines_moments | mean precision | 1.00 → 0.99 | −1.0 % (1 FP) |

(The big-tail-win corpora pay a recall cost on the same corpus; e.g. tag16h5
gains −89.6 % p99 rotation but loses −6 % recall — the entangled trade-off
described in §5. `low_key_tuned` is a near-pure win on accuracy but loses
−27 % recall.)

`negative_detection` stays clean (no new false positives, zero allocation
growth).

## 5. Why strict Pareto is unreachable here

- **The board regression is structural.** The ChArUco refiner predicts saddle
  search locations through the per-tag homography built from the AprilTag
  corners. *Any* 2-DOF corner movement shifts that homography, moving saddle
  predictions — so `board_charuco` / `charuco_refiner` p99 regress at every gate
  tuning, independent of recall/residual thresholds. (Cross-ref
  `project_refine_saddle_noop.md`.)
- **The render-tag trade-off is entangled.** Residual improvement does not
  cleanly track pose-correctness: tightening the gate to protect recall also
  rejects beneficial rotations (tag16h5 / low_key_tuned lose their tail wins),
  while loosening it re-introduces recall loss. The same mechanism (rotating the
  edge normal) that fixes tails also drifts corners onto adjacent tag structure.
- **No profile/route boundary isolates wins from losses.** 2-DOF fires wherever
  the quad-extraction stage routes through `Erf` (standard & grid via Static
  `refinement_mode`; high_accuracy/max_recall via `AdaptivePpb.low_refinement`).
  The wins and losses live on the **same profiles and same corpora**: gating to
  `high_accuracy` makes 2-DOF a no-op (the deadband already neutralizes it on the
  clean hub), and `standard` carries both the tail wins and the recall losses
  together. Gating by profile therefore either changes nothing or loses the wins
  with the losses.

This independently confirms the original author's conclusion that 2-DOF cannot
clear strict Pareto.

## 6. Decision & re-attempt conditions

- **Production stays on proven 1-DOF.** `main` is unchanged (byte-identical
  protected snapshots).
- **Hardened code preserved** on branch `feat/erf-2dof-tukey-narrow-band`
  (the "harden 2-DOF ERF…" + "address code-review findings…" commits) —
  recoverable via `git show`. PR #264 is closed in favour of this postmortem.
- **Carry the hardening forward.** Any future quad-refinement rotation work must
  include, from the start, the deadband and the robust same-set bounded-influence
  acceptance gate (§3) — they fix the clean-corner regression, keep well-aligned
  edges byte-identical, and avoid the degenerate / unfair acceptance modes a
  naive residual gate falls into.

Re-attempt is only worth it if **both** hold:

1. **The board path is decoupled** — the ChArUco saddle prediction must not
   depend on the same corners that 2-DOF moves (e.g. predict saddles from a
   1-DOF / unrefined homography while using refined corners only for single-tag
   pose), or the board regression returns regardless of edge-fit quality.
2. **A discriminator separates beneficial from harmful rotations** better than
   residual improvement does — residual is a poor proxy for pose-correctness
   near tag structure.

**Recommended alternative:** extend the production-tested **GWLF** refiner to the
quad-extraction site (Lane 3). It is a different algorithm (not ERF-IRLS), so it
is immune to the iter-0 / residual-proxy failure modes; it warrants its own
evaluation against the same protected suites.

## 7. Reproduction

On `feat/erf-2dof-tukey-narrow-band` (hardened 2-DOF active):

```bash
INSTA_UPDATE=new TRACY_NO_INVARIANT_CHECK=1 \
LOCUS_ICRA_DATASET_DIR=tests/data/icra2020 \
LOCUS_HUB_DATASET_DIR=tests/data/hub_cache \
cargo nextest run --release --features bench-internals -p locus-core \
  --ignore-default-filter --no-fail-fast \
  -E 'binary(regression_render_tag) | binary(regression_render_tag_robustness) | binary(regression_board_hub) | binary(regression_icra2020)'
```

Diff each generated `*.snap.new` against its committed `*.snap` baseline to read
the per-corpus deltas in §4.

## References

- PR #264 (`feat/erf-2dof-tukey-narrow-band`) — hardened 2-DOF, not merged.
- `docs/explanation/edge_refinement_2dof_failure_analysis.md` (on the branch) —
  the full T0–T4b ablation and per-corpus detail.
- `edlines_sota_followup_postmortem_2026-05-25.md`,
  `pose_covariance_phase4_postmortem_2026-05-31.md` — the docs-only
  negative-result pattern this follows.
- `project_refine_saddle_noop.md` — the corner→homography→saddle coupling behind
  the board regression.
