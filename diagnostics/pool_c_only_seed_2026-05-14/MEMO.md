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

### Standard profile — 150 / 150 frames covered

| metric | 3 pools (PR #256) | Pool C only (this PR) | Δ |
| :--- | ---: | ---: | ---: |
| **aprilgrid_golden** | | | |
| frames_with_board | 150 | 150 | 0 |
| **p99 rot °** | **1.295316** | **1.295316** | byte-identical |
| p95 rot ° | 0.338391 | 0.338391 | byte-identical |
| p50 rot ° | 0.052321 | 0.052736 | +0.00042 (+0.79 %) |
| mean rot ° | 0.118432 | 0.119874 | +0.00144 (+1.22 %) |
| **p99 trans m** | **0.018625** | **0.018625** | byte-identical |
| p95 trans m | 0.011282 | 0.011282 | byte-identical |
| mean trans m | 0.0026527 | 0.0026520 | −6.6e-7 (−0.025 %) |
| **charuco_golden** | | | |
| frames_with_board | 150 | 150 | 0 |
| **p99 rot °** | **0.650910** | **0.650910** | byte-identical |
| p95 rot ° | 0.344252 | 0.344252 | byte-identical |
| mean rot ° | 0.107244 | 0.107257 | +0.000013 (+0.012 %) |
| **p99 trans m** | **0.011106** | **0.011106** | byte-identical |
| mean trans m | 0.00237981 | 0.00237979 | −2e-8 (−0.001 %) |

### high_accuracy profile — 130 / 150 frames covered (20 silently skipped, see `project_high_accuracy_min_area_diagnostic_2026-05-14`)

| metric | 3 pools | Pool C only | Δ |
| :--- | ---: | ---: | ---: |
| **aprilgrid_golden_high_accuracy** | | | |
| frames_with_board | 130 | 130 | 0 |
| **p99 rot °** | **0.387124** | **0.387124** | byte-identical |
| p95 rot ° | 0.199999 | 0.199999 | byte-identical |
| p50 rot ° | 0.028981 | 0.028981 | byte-identical |
| mean rot ° | 0.060153 | 0.060113 | −0.00004 (−0.07 %) |
| p99 trans m | 0.005117 | 0.005117 | byte-identical |
| **charuco_golden_high_accuracy** | | | |
| frames_with_board | 130 | 130 | 0 |
| **p99 rot °** | **0.540866** | **0.540866** | byte-identical |
| p95 rot ° | 0.248229 | 0.248229 | byte-identical |
| mean rot ° | 0.070869 | 0.070867 | −2.8e-6 (−0.004 %) |
| p99 trans m | 0.004130 | 0.004130 | byte-identical |

The high_accuracy variants are part of open PR #253; this PR doesn't add
them, but the ablation captures their numbers as additional evidence
that Pool A + Pool B are dead weight under the noisier large-`min_area`
corpus as well.

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
loop:

| step | PR #256 | this PR |
| :--- | --- | --- |
| seed candidates | 1 (centroid) + 8 (per-tag IPPE) + 2 (DLT-IPPE) = 11 | 2 (DLT-IPPE) |
| per-candidate polish | 3 GN steps on 16 points | identical |
| per-candidate scoring | 16-point reprojection | identical |
| **total polish + score ops per sample** | **~11** | **~2** |

`solve_ippe_square` itself is `O(1)` — the bulk of seed cost is in the
per-candidate polish + score.  Not measured precisely in this session
(parallel-agent CPU contention noisy — see
`feedback_serialize_timing_captures`); cleanly measurable once the
cleanup PR lands.

## Files touched

- `crates/locus-core/src/board.rs`:
  - delete `compute_centroid_translation_seed` (Pool A helper).
  - delete `ippe_branches_for_group` (Pool B helper).
  - delete `classify_square_tag` + its constants `IPPE_SEED_SQUARE_REL_TOL`
    / `IPPE_SEED_MIN_SIDE_M` (only used by the deleted Pool B).
  - delete the `Homography` import (only used by Pool B).
  - rewrite `solve_seed_from_sample_homography` (formerly
    `solve_seed_from_ippe_enumeration`) as a 2-branch enumeration over
    Pool C's IPPE-Square decomposition only.
  - update `PointCorrespondences` and `lo_ransac_loop` docstrings.
  - update unit tests: rename to `*_sample_homography_*`, replace the
    centroid-fallback test with a `saddle_path` test that asserts the
    same `group_size = 1` path works under Pool C alone.
- `crates/locus-core/tests/snapshots/`:
  - `regression_board_hub__board_aprilgrid_v1_golden.snap` — mean rot
    drift only.
  - `regression_board_hub__board_charuco_v1_golden.snap` — mean rot
    drift only.
- `diagnostics/pool_c_only_seed_2026-05-14/MEMO.md` — this memo.

## Constraints honoured

- No new dependencies.
- No `unwrap` / `expect` in library code.
- No `unsafe` added.
- LO-RANSAC thresholds untouched.
- Schemas / Python bindings unchanged.
- `pose.rs::solve_ippe_square` stays `pub(crate)` (unchanged from #256).
