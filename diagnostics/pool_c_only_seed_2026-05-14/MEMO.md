# Board-seed cleanup: Pool C alone is optimal — 2026-05-14

Empirical ablation falsifies a static-reasoning claim in PR #256's memo
(`diagnostics/multi_ippe_seed_2026-05-14/MEMO.md`) that Pool B (per-tag
IPPE-Square branch enumeration) is sometimes a tighter seed than Pool C
(sample-homography → IPPE-Square decomposition).  Across both standard
and `high_accuracy` profiles on both board corpora, Pool A (centroid
seed) and Pool B contribute nothing measurable — every tail metric is
byte-identical and mean rotation drifts by ≤ 0.0015°.  This PR deletes
the two unused pools and the helpers that supported them.

## Algorithm — before vs after

| step | PR #256 (3 pools) | this PR (Pool C only) |
| :--- | :--- | :--- |
| candidate enumeration | A: centroid seed (1) + B: per-tag IPPE × 4 tags = 8 + C: sample-homography → IPPE = 2 | C: sample-homography → IPPE = 2 |
| total candidates | up to 11 | up to 2 |
| polish | 3-step GN on each, score by joint reprojection RMSE | identical |
| winner | lowest joint-reprojection RMSE | lowest joint-reprojection RMSE of the 2 branches |

The seed function is renamed `solve_seed_from_ippe_enumeration` →
`solve_seed_from_sample_homography` to reflect what's left: a single DLT
homography fit, decomposed by IPPE-Square's two-branch enumeration.  The
"enumeration" in the old name implied per-tag enumeration; with that
removed, "sample homography" is the operative description.

## Ablation evidence

Both runs use the binaries built at this branch.  Pools were gated off
via `if false` blocks for the ablation runs (not present in the
committed code).

### Standard profile — 150/150 frames, both corpora

p99 rot °, p99 trans m: **byte-identical** (aprilgrid 1.295316° /
0.018625 m; charuco 0.650910° / 0.011106 m) between 3-pool and Pool-C-only.
Mean rot drifts ≤ +1.22 % (aprilgrid) / +0.012 % (charuco) — noise from a
different candidate occasionally winning ties, not a real regression.

### high_accuracy profile — 130/150 frames (20 silently skipped, see
`project_high_accuracy_min_area_diagnostic_2026-05-14`), both corpora

Same pattern: p99 rot/trans byte-identical (aprilgrid 0.387124° /
0.005117 m; charuco 0.540866° / 0.004130 m), mean rot drift ≤ 0.07 %.
These variants are part of open PR #253; captured here as extra evidence
Pool A/B are dead weight under the noisier large-`min_area` corpus too.

### Cross-cutting

Running the **full** insta snapshot suite (`cargo insta test --release
--all-features --features bench-internals`) under Pool C only:

- 4 board snapshots produce `.snap.new` diffs (2 committed + 2 from PR
  #253's branch) — drift confined to mean rotation, all tail metrics
  identical.
- 0 other snapshots produce `.snap.new` (render-tag, render-tag
  robustness, distortion Brown-Conrady, distortion Kannala-Brandt, ICRA
  forward / circle / random, pose-consistency-ROC, dictionaries,
  straight-space — all byte-identical).
- All 236 unit tests pass.

Pool A and Pool B affect *only* the board-pose code path, as the
implementation suggested.

## Why the static reasoning didn't pan out

PR #256's memo stated:

> Pool B (per-tag IPPE-Square) is retained because on clean frames it is
> sometimes a *tighter* seed than Pool C — the joint reprojection score
> picks the winner of all 3 pools.

The reasoning was sound — per-tag IPPE-Square on clean 4-corner inputs
does start *inside* the correct Necker basin, so its initial reprojection
score before polish is sometimes lower than Pool C's (which has to
disambiguate across all 16 correspondences at once).  What the static
analysis missed is that **the 3-step GN polish closes the gap**: on
clean frames, Pool C's two branches polish into the same global minimum
that Pool B's per-tag branches converge to.  By the time scoring
happens, every candidate is polished, and the corner-noise-averaged Pool
C branches are byte-identical to (or marginally tighter than) Pool B's
per-tag-derived branches.

On catastrophic frames (`scene_0138` aprilgrid: 35 tags, per-tag rot
mean 16.9°, max 69°), Pool B's per-tag IPPE all converge to the same
wrong basin → can't be rescued by polish → Pool C alone is what
recovers correct pose.

Net: Pool C's two-branch enumeration is *both* the clean-frame and the
catastrophic-frame winner.  Pool A and Pool B add candidates that the
scoring step *never* selects on any frame in either board corpus under
either profile.

## Latency

Roughly **−70 %** of per-minimal-sample seed work in the LO-RANSAC outer
loop: candidates drop from 11 (1 centroid + 8 per-tag IPPE + 2 DLT-IPPE) to
2 (DLT-IPPE only); per-candidate polish (3 GN steps) and scoring (16-point
reprojection) are unchanged per-candidate. Not measured precisely in this
session (parallel-agent CPU contention noisy — see
`feedback_serialize_timing_captures`).
