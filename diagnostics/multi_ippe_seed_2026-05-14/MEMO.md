# Multi-IPPE-Square branch enumeration RANSAC seed — 2026-05-14

Replaces PR #255's DLT+Zhang seed (`solve_pnp_planar_seed`) with theoretically
sound multi-IPPE-Square branch enumeration that exposes the planar Necker
ambiguity to the joint-reprojection consensus score, instead of silently
collapsing it at the SVD step.

## Algorithm summary

Inside `RobustPoseSolver::lo_ransac_loop`, each 4-tag minimal sample now drives
a 3-pool candidate enumeration in `solve_seed_from_ippe_enumeration`:

1. **Pool A — centroid + identity rescue** (1 candidate). `R = I`, `t`
   anchored so the centroid of the 4 sampled object points projects to the
   centroid of the 4 sampled image points at `z = 1 m`.
2. **Pool B — per-tag IPPE-Square branch enumeration** (≤ 8 candidates).
   For each sampled tag whose 4 object points form an axis-aligned square,
   call the existing `pose.rs::solve_ippe_square` on its 4 corners → 2
   candidate `camera-from-tag` poses → 2 `camera-from-board` poses (composed
   with the known tag-in-board rigid transform `T_cam_from_board = T_cam_from_tag · [I | −c_b]`).
3. **Pool C — 16-point DLT homography + IPPE-Square decomposition**
   (2 candidates). Fit a totals-least-squares planar homography
   `[x_n, y_n, 1]^T ∝ H · [X − c, Y − c, 1]^T` over all 16 sample
   correspondences via the smallest-eigenvalue eigenvector of the 9 × 9
   `A^T A`, then run `solve_ippe_square` on `H` to recover *both* Necker
   branches. Un-shift the translation by the sample centroid: `t_board =
   t_centred − R · c`.

Every candidate is polished with a 3-step unweighted Gauss-Newton refinement
over the 16 sample correspondences and scored by joint reprojection RMSE.
Winner becomes the seed for `evaluate_inliers` → `lo_inner` →
`refine_aw_lm` (unchanged).

The Necker ambiguity is disambiguated *jointly*: the correct branch of any
tag, when transported to board frame, agrees with the correct branches of
the other 3 tags on a single global pose, producing low joint reprojection.
Pool C is essential for robustness on small / noisy tags (≈ 20-30 px edges)
where per-tag IPPE collapses entirely — the DLT homography averages corner
noise across 16 points and IPPE-Square then recovers both well-conditioned
branches. Pool B's per-tag enumeration handles the common case where one
tag's solution is the cleanest. Pool A is the rescue seed when both fail.

### Signature

```rust
fn solve_seed_from_ippe_enumeration(
    sample: &[usize; 4],
    corr: &PointCorrespondences<'_>,
    intrinsics: &CameraIntrinsics,
) -> Option<Pose>
```

(No `topology` parameter: the per-tag tag-center / tag-size is recovered
from the 4 object points of each correspondence group via
`classify_square_tag`, so the function stays topology-agnostic and works
unchanged for both `BoardEstimator` and `CharucoRefiner`.)

### Frame convention

For `group_size == 4`, each group's 4 object points are assumed to be the
`[TL, TR, BR, BL]` corners of an axis-aligned square in the board frame —
the standard AprilGrid / ChAruco-marker convention. The tag and board share
orientation (no per-tag rotation), so the camera-from-tag → camera-from-board
composition is purely translational:

```
R_cam_from_board = R_cam_from_tag
t_cam_from_board = t_cam_from_tag − R_cam_from_tag · tag_center_in_board
```

This matches the convention used by the deleted `board_seed_from_pose6d`
helper (pre-PR #255 at commit `4798241`) and by `solve_ippe_square`'s
internal centred-tag convention (corners at
`(−L/2, −L/2) .. (L/2, L/2)` for `[TL, TR, BR, BL]`).

For `group_size == 1` (ChAruco saddle path) the per-group square is not
available; only Pool A and Pool C are exercised.

## Before/after on the regression-board-hub snapshots

| Metric | pre-#255 (main) | PR #255 (DLT+Zhang) | This PR (multi-IPPE) |
| :--- | ---: | ---: | ---: |
| **aprilgrid_golden — 150 frames** | | | |
| frames_with_board | 150 | 150 | **150** |
| mean rot ° | 1.570 | 0.120 | **0.118** |
| p50 rot ° | 0.498 | 0.051 | **0.052** |
| p95 rot ° | 6.259 | 0.338 | **0.338** |
| p99 rot ° | 19.370 | 1.295 | **1.295** |
| mean trans m | 0.0111 | 0.00259 | **0.00265** |
| p99 trans m | 0.214 | 0.01862 | **0.01863** |
| **charuco_golden — 150 frames** | | | |
| frames_with_board | 150 | 150 | **150** |
| mean rot ° | 2.299 | 0.108 | **0.107** |
| p50 rot ° | 0.303 | 0.068 | **0.068** |
| p95 rot ° | 7.922 | 0.344 | **0.344** |
| p99 rot ° | 56.172 | 0.651 | **0.651** |
| mean trans m | 0.0107 | 0.00224 | **0.00238** |
| p99 trans m | 0.162 | 0.01115 | **0.01111** |

Result: numerically indistinguishable from PR #255 on both p99 rotation and
mean translation across both boards. **The expected p99 ≈ 1.30° aprilgrid /
0.65° charuco bound from the brief is preserved exactly.**

## Branch-selection diagnostics

(Static reasoning from the algorithm — no per-frame instrumentation was
added.)

- On **clean (high tag-edge px) frames**, both Pool B IPPE branches per
  tag converge under polish to the same global minimum; Pool C's two
  branches separate cleanly into "correct" and "Necker-flipped"; the
  winner is the lowest-reprojection candidate from either pool — typically
  a Pool B candidate because its starting point is already inside the
  correct basin.
- On **catastrophic (≈ 20-30 px tag-edge) frames** identified in
  `diagnostics/board_p99_investigation_2026-05-14/MEMO.md` (e.g.
  `scene_0138` aprilgrid: 35 tags, per-tag rot mean 16.9°, max 69°),
  per-tag IPPE collapses on every tag — both Pool B branches end up
  wrong-sided. Pool C's 16-point homography is the **necessary** rescue
  — it averages per-corner noise across 16 points and produces a
  well-conditioned homography even when no individual tag is reliable.
- Pool A (`R = I` + centroid translation) very rarely wins after polish
  because the 3 GN steps can't bridge an arbitrary 30°+ rotation gap.
  It exists as a defensive fallback for extreme degenerate frames where
  both IPPE pools return `None` (e.g. all sampled tags share the same
  pixel position — should never happen in practice).

A diagnostic mode that emits per-frame pool-winner stats would be a
worthwhile follow-up but was not in scope for this redo.

## Why this works where literal-brief-spec multi-IPPE doesn't

The brief literally specifies "for each tag … `solve_ippe_square` →
2 candidates per tag → 8 total". When implemented strictly:

| Metric | Brief-literal (Pool B only, 3-step GN polish) |
| :--- | ---: |
| frames_with_board (aprilgrid) | **108 / 150** |
| frames_with_board (charuco) | **107 / 150** |
| p99 rot ° (frames that succeeded) | 1.028 (aprilgrid) / 0.464 (charuco) |

The 42-43 lost frames are exactly the "catastrophic per-tag-IPPE collapse"
frames documented in
`diagnostics/board_p99_investigation_2026-05-14/MEMO.md`. On these frames
per-tag IPPE-Square on 4 noisy corners produces a 50°+ biased pose with
both Necker branches landing in the same wrong basin; the 3-step polish
can't escape, and `outer_count < min_inliers = 4` kills the seed.

PR #255's DLT-on-16-points avoided this by averaging corner noise across 16
points. The fix is to keep the DLT 16-point homography fit but **decompose
with `solve_ippe_square` (which returns both branches) instead of with
Zhang's `R = SO(3)-project(K⁻¹ H)` (which silently collapses to one
branch)**. That gives us Pool C, which adds:

- DLT's noise-averaging robustness (matches PR #255 empirically).
- IPPE-Square's two-branch enumeration (theoretical soundness goal of the
  brief).
- Joint-reprojection disambiguation (the brief's Necker-ambiguity argument).

Pool B (per-tag IPPE-Square) is retained because on clean frames it is
sometimes a *tighter* seed than Pool C — the joint reprojection score
picks the winner of all 3 pools.

## Latency delta vs PR #255

Not measured precisely in this session (per
`.agent/feedback/feedback_serialize_timing_captures.md` — measurements
during a parallel-agent session are CPU-contention-noisy). Static cost
analysis per minimal sample:

| Step | PR #255 cost | This PR cost |
| :--- | --- | --- |
| Seed | 1× DLT (9×9 symmetric eig) | 8× IPPE-Square + 1× DLT-eig + 1× IPPE-Square decomp |
| Polish | 3× GN step on 16 points | 11× polish (each = 3 GN steps on 16 points) |
| Score | implicit | 11× sample reprojection (16 points each) |

So per minimal sample, this PR does ~11× the work of PR #255. The RANSAC
loop runs `k_min = 15` to `k_max = 50` iterations per frame; per-sample
cost dominates over the LO inner-loop. Empirical latency delta therefore
expected in the **+5% to +20%** range (well within Locus's typical
profile gates), but flagged as a follow-up: re-measure under serial
nextest in a clean session before merging.

## What I tried (failed variants)

- **Pool B only (literal brief spec) + 3-step GN polish**: 108 / 150
  aprilgrid frames, 107 / 150 charuco. Pool B alone is insufficient — see
  the table above.
- **Pool B only + no polish** (raw IPPE branch scoring): much worse —
  scoring on un-polished noisy candidates picks the wrong branch more
  often. Numbers not captured before adding polish.
- **Pool A + Pool B (centroid + per-tag IPPE)**: ~108 / 150. Centroid
  fallback rarely wins after polish because the polish can't bridge a
  30°+ rotation gap in 3 steps. Confirmed that Pool A is decorative on
  these datasets.

The Pool C addition (16-point DLT homography → IPPE-Square decomposition)
is what closes the gap to PR #255 baseline. **This combines DLT's noise
averaging with IPPE-Square's two-branch decomposition** — addressing the
brief's theoretical concern (Zhang silently collapses the ambiguity)
while preserving the empirical robustness.

## Files touched

- `crates/locus-core/src/board.rs` — new `solve_seed_from_ippe_enumeration`,
  `ippe_branches_for_group`, `ippe_branches_from_sample_homography`,
  `compute_centroid_translation_seed`, `gn_step_on_sample`,
  `sample_reprojection_score`, `classify_square_tag` helpers; removed
  `solve_pnp_planar_seed` (PR #255) and `board_seed_from_pose6d` (pre-#255);
  removed `seed_poses` field from `PointCorrespondences`; removed
  `scratch_seeds` from `BoardEstimator`.
- `crates/locus-core/src/pose.rs` — `solve_ippe_square` lifted from `fn` to
  `pub(crate) fn` so `board.rs` can call it.
- `crates/locus-core/src/charuco.rs` — removed `scratch_seeds` /
  `seed_poses` plumbing (no behavioural change; matches PR #255).
- `crates/locus-core/tests/snapshots/regression_board_hub__board_aprilgrid_v1_golden.snap`
  and `…__board_charuco_v1_golden.snap` — refreshed to the new (numerically
  equivalent to PR #255) values.

## Constraints honoured

- No new dependencies.
- No `unwrap` / `expect` in library code.
- No naked `unsafe` blocks (none added).
- No changes outside `crates/locus-core/src/board.rs`, `pose.rs`,
  `charuco.rs`, the two board snapshots, and this diagnostics dir.
- LO-RANSAC thresholds (`tau_outer_sq`, `tau_inner_sq`, etc.) untouched.
- Schemas / Python bindings unchanged (board.rs is a pure Rust internal).
