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
- Two genuinely-universal hardening fixes were developed and **kept on the
  branch** (the salvage):
  1. **Deadband fallback** — when the converged normal rotated < ~0.4° from the
     seed, defer to the proven 1-DOF ρ-only solve. Fixes a real correctness
     regression (a perfectly-aligned control corner degraded 0.05 px → 0.16 px)
     and makes 2-DOF a no-op on well-aligned edges, so the clean render-tag hub
     stays byte-identical.
  2. **No-worse / residual-acceptance gate** — keep the 2-DOF rotation only when
     the 1-DOF reference fit is poor relative to edge contrast *and* 2-DOF cuts
     the robust residual by a margin, i.e. only on genuinely misrotated seeds.
- Even with the hardening, **the strict-Pareto bar (no recall drop > 0.5 %, no
  p99 regression > 2 % on any corpus) is missed on ~10 corpora**, and the board
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

Rebased onto current main, then:

- **Deadband fallback** (`THETA_DEADBAND_RAD ≈ 0.4°`): diagnostics showed the
  clean-corner failure was *not* rotation (θ ≈ 0.08°) but a ρ bias from the
  joint solve. When the converged rotation is below the deadband, the seed
  normal was already right, so the path restores the seed and runs the proven
  1-DOF ρ solve — byte-identical to `OneDof`. Result: control corner back to
  0.05 px; **the entire clean render-tag hub (640/720/1080/2160 + primary
  1080p) is byte-identical to main**, where the original #264 regressed it.
- **No-worse / residual-acceptance gate** (`TWO_DOF_POOR_FIT_FRAC = 0.15`,
  `TWO_DOF_ACCEPT_RESID_FRAC = 0.97`): the 2-DOF rotation is accepted only when
  the 1-DOF reference fit is poor relative to contrast (evidence the seed is
  genuinely misrotated) *and* 2-DOF reduces the robust residual by a margin.
  Tags that decode fine under 1-DOF (good fit) keep 1-DOF, recovering much of
  the recall loss while preserving the tail wins.

## 4. Strict-Pareto measurement (hardened config, on current main)

All four protected suites re-run on current `main` with the hardened 2-DOF
active, vs. the 1-DOF baseline snapshots. Verified hardware: AMD EPYC-Milan,
8 vCPU (2 threads/core), L3 32 MiB, x86_64; `--release --features
bench-internals`; `cargo nextest` (latency redacted in snapshots). Pose mode:
Accurate.

**Wins (representative):**

| Corpus | metric | 1-DOF → 2-DOF | Δ |
| :-- | :-- | :-- | --: |
| tag36h11 gwlf | p99 rotation | 3.17° → 0.99° | −68.9 % |
| tag36h11 gwlf | mean RMSE | 0.989 → 0.859 | −13.2 % |
| moments_culling | p99 rotation | 27.8° → 1.56° | −94.4 % |
| high_iso | p99 rotation | 104.1° → 2.09° | −98.0 % |
| raw_pipeline | mean RMSE | 0.759 → 0.40 | −47.3 % |
| icra fixtures | mean RMSE | 0.1315 → 0.060 | −54.5 % |
| icra pure (×2) | mean recall | 0.738 → 0.745 | +1.0 % |
| board_aprilgrid | p99 board rotation | 1.30° → 0.76° | −41.3 % |

**Persistent strict-Pareto violations (~10 corpora; not tunable away):**

| Corpus | metric | 1-DOF → 2-DOF | Δ |
| :-- | :-- | :-- | --: |
| board_charuco | p99 board rot / trans | +35 % / +61 % | regress |
| charuco_refiner | p99 board rot / trans | +82 % / +86 % | regress |
| raw_pipeline | mean recall | 0.58 → 0.52 | −10.3 % |
| low_key_tuned | p99 rotation | 21.6° → 29.4° | +35.8 % |
| high_iso / moments_culling / raw_pipeline | p99 translation | +6 % … +24 % | regress |
| tag16h5 | mean recall | 1.00 → 0.99 | −1.0 % |
| edlines / edlines_moments | mean precision | 1.00 → 0.99 | −1.0 % (1 FP) |

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
  (commit message: "harden 2-DOF ERF against clean-corner regression + bad
  rotations") — recoverable via `git show`. PR #264 is closed in favour of this
  postmortem.
- **Carry the hardening forward.** Any future quad-refinement rotation work must
  include the deadband + no-worse acceptance guard from the start — they fix the
  clean-corner regression and keep well-aligned edges byte-identical.

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
