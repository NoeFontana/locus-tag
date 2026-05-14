# Multi-tag non-minimal-PnP RANSAC seed — 2026-05-14

## TL;DR

Replaces the per-tag IPPE seed inside `RobustPoseSolver::lo_ransac_loop`
with a **non-minimal planar PnP** computed from 16 corner correspondences
per minimal sample (4 sampled tags × 4 corners). On the `standard` profile,
board-pose p99 rotation drops **15× on AprilGrid (19.4° → 1.30°)** and
**86× on ChAruco (56.2° → 0.65°)**, mean drops **13× / 21×**, with no
recall regression and an acceptable +0.20 ms / +60 % per-frame latency cost.

Every non-board snapshot is byte-identical: render-tag (×17), distortion
(×2), robustness (×7), ICRA (×8), dictionaries (×4), pose-consistency-ROC
(×1). Only the four board-hub snapshots changed.

## Algorithm — planar non-minimal PnP via DLT homography

The board is planar (`Z = 0` for AprilGrid corners, ChAruco saddles, and
ChAruco-as-AprilGrid markers), which lets us solve PnP via a Zhang-style
homography decomposition:

1. Normalise object points to mean zero / mean ‖·‖ = √2 for numerical
   conditioning (standard DLT trick).
2. Build the symmetric 9 × 9 matrix `A^T A` by accumulating per-point DLT
   row outer products. Two rows per correspondence, never materialised.
3. Solve for `H` via the smallest-eigenvalue eigenvector of `A^T A`
   (nalgebra `symmetric_eigen`, fixed 9 × 9, no heap allocation).
4. Un-normalise, decompose `H = [r1 | r2 | t] · γ` with
   `γ = ½(‖h1‖ + ‖h2‖)`, sign-correct via centroid depth.
5. Gram-Schmidt + SVD polish to produce a proper rotation
   (`R = U V^T` with `det R = +1`).

This seed is then polished with 3 unweighted GN steps over the same 4-tag
sample before the LO-RANSAC `evaluate_inliers` consensus gate. The
existing `lo_inner` GN loop and `refine_aw_lm` final stage are unchanged.

At 16+ correspondences the planar-PnP solution is unambiguous (no
two-fold IPPE flip), and the per-corner noise averages over a much larger
constraint set than per-tag IPPE.

## Before/after — standard profile (origin/main, commit 4798241)

| dataset | metric | before | after | delta |
| :--- | :--- | ---: | ---: | ---: |
| aprilgrid_golden_v1 | p99 rot° | 19.37 | **1.30** | **−93 %** |
| aprilgrid_golden_v1 | mean rot° | 1.57 | **0.12** | **−92 %** |
| aprilgrid_golden_v1 | p95 rot° | 6.26 | **0.34** | **−95 %** |
| aprilgrid_golden_v1 | p99 trans m | 0.214 | **0.019** | **−91 %** |
| aprilgrid_golden_v1 | mean trans m | 0.0111 | **0.0026** | **−77 %** |
| aprilgrid_golden_v1 | frames_with_board | 150/150 | 150/150 | unchanged |
| charuco_golden_v1 | p99 rot° | 56.17 | **0.65** | **−99 %** |
| charuco_golden_v1 | mean rot° | 2.30 | **0.11** | **−95 %** |
| charuco_golden_v1 | p95 rot° | 7.92 | **0.34** | **−96 %** |
| charuco_golden_v1 | p99 trans m | 0.162 | **0.011** | **−93 %** |
| charuco_golden_v1 | mean trans m | 0.0107 | **0.0022** | **−79 %** |
| charuco_golden_v1 | frames_with_board | 150/150 | 150/150 | unchanged |

These are exact insta-snapshot values from the regression suite.

## Latency

Captured by instrumenting `crates/locus-core/tests/regression_board_hub.rs`
to print per-frame board-pose `board_ms` over both datasets in both
profiles (n = 600 frame-pose measurements, single-threaded `--release`).

| metric | baseline (per-tag IPPE seed) | after (non-minimal PnP seed) | delta |
| :--- | ---: | ---: | ---: |
| mean board ms | 0.326 | **0.523** | **+0.20 ms (+60 %)** |
| p50 board ms | 0.314 | **0.423** | **+0.11 ms (+35 %)** |
| p95 board ms | 0.524 | **0.943** | **+0.42 ms (+80 %)** |
| max board ms | 0.587 | **1.009** | **+0.42 ms (+72 %)** |

The added cost is the per-RANSAC-iteration planar-PnP solve (≈ 4–5 k FP
ops via `symmetric_eigen` + SVD polish + 3 GN steps over 16 points).
Detector-side latency is unchanged. The full per-frame board pose still
fits comfortably under 1 ms.

Hardware: AMD EPYC-Milan (x86_64, 8 vCPU, Linux 6.8). Build profile:
`cargo test --release --all-features --features bench-internals`.
Latency captures were serialised (no parallel agents running).

## Algorithmic invariants preserved

- `RobustPoseSolver::estimate` public signature is byte-identical.
- `BoardEstimator::estimate(batch, intrinsics)` Python signature unchanged.
- LO-RANSAC `k_min`/`k_max`/`tau_*_sq` thresholds untouched. They are
  slightly conservative for the new seed quality — leaving headroom is
  fine; tuning is out of scope.
- Phase-isolated execution privileges (`detection-batch-contract.md`)
  unchanged: `BoardEstimator::flatten_batch` still only reads
  `corners` / `corner_covariances` and only writes the scratch arrays
  it owns.
- Zero heap allocation in the hot path: the 4-tag sample buffers are
  fixed-size `[Point2f; MAX_SEED_SAMPLES]` / `[[f64; 3]; MAX_SEED_SAMPLES]`
  stack arrays inside `lo_ransac_loop`. The 9 × 9 `A^T A` matrix is a
  fixed nalgebra `Matrix<f64, U9, U9>`.

## Snapshot drift (insta)

Only 4 snapshots changed:

- `regression_board_hub__board_aprilgrid_v1_golden.snap` (standard)
- `regression_board_hub__board_charuco_v1_golden.snap` (standard)
- `regression_board_hub__board_aprilgrid_v1_golden_high_accuracy.snap`
  (high_accuracy) — **only exists on PR #253 / `feat/high-accuracy-min-area-400`;
  not on `origin/main` (4798241). The diff is captured for the eventual
  rebase but is not part of this branch.**
- `regression_board_hub__board_charuco_v1_golden_high_accuracy.snap`
  (high_accuracy) — same caveat.

Every other tracked snapshot (render-tag, distortion, robustness, ICRA,
dictionaries, pose-consistency-ROC, charuco) is byte-identical, confirmed
via `cargo insta test --release --all-features --features bench-internals
--unreferenced=ignore` — "no snapshots to review".

## Open questions / next steps

1. **Tighten `tau_outer_sq` / `tau_inner_sq`?** The current 100 px² /
   1 px² thresholds were calibrated against the noisier per-tag IPPE
   seed. With the new seed the LO inner-loop GN consistently converges
   into the sub-pixel basin — `tau_inner_sq` could likely move from
   1 px² to ≈ 0.25 px² without losing recall. Left for a follow-up
   tuning PR.
2. **Drop the 3-step GN polish?** The first 1–2 GN steps account for most
   of the seed sharpening; step 3 is rarely productive. Profiling
   suggests skipping it would recover ≈ 0.05 ms / frame at modest
   accuracy cost. Not measured for this branch.
3. **Charuco saddle-only path.** `CharucoRefiner::estimate` uses
   `group_size = 1` (1 bit = 1 saddle). The new seed code path correctly
   handles `group_size = 1` (4-sample → 4-correspondence → 4-point
   planar PnP), but the `4-point` case is exactly the minimal sample
   for planar PnP — the seed is uniquely determined but noise-sensitive.
   The LO-RANSAC consensus gate handles the noise; the saddle path's
   snapshots are unchanged. Leaving as-is.
4. **Non-planar boards.** The current solver assumes `Z = 0` for all
   object points. If a non-planar board topology is ever added, the
   solver returns `None` (gracefully) and the minimal sample is
   skipped. A full 3D PnP (EPnP or SQPnP) would be needed; out of scope.

## Branch / commit

Branch: `feat/multi-tag-p4p-ransac-seed` (off `origin/main` at
`4798241`). Single commit per the brief. To open the PR: parent agent
runs `gh pr create`.
