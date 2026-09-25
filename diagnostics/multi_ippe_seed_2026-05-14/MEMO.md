# Multi-IPPE-Square branch enumeration RANSAC seed — 2026-05-14

Replaced PR #255's DLT+Zhang seed (`solve_pnp_planar_seed`) with multi-IPPE-Square
branch enumeration that exposes the planar Necker ambiguity to the joint-reprojection
consensus score, instead of silently collapsing it at the SVD step.

**Superseded 2026-05-14**: the 3-pool design below was ablated same-day and Pool
C alone (DLT homography + IPPE-Square decomposition) turned out
byte-identical on every tail metric to the full ensemble — Pools A/B were
decorative. Production now runs Pool C only; see
`diagnostics/pool_c_only_seed_2026-05-14/MEMO.md`. This memo is kept for the
Pool C rationale below, which is still load-bearing (cited from
`board.rs::ippe_branches_from_sample_homography`).

## Why per-tag IPPE isn't enough

Per-tag IPPE-Square on 4 noisy corners (≈ 20-30 px tag edges) can produce a
pose biased by 50°+, with both Necker branches landing in the same wrong
basin. On catastrophic frames (e.g. `scene_0138` aprilgrid: 35 tags,
per-tag rot mean 16.9°, max 69°; the underlying per-frame investigation
dump was not committed to this repo) this happens on *every* tag — a
brief-literal implementation (Pool B only: per-tag IPPE, 8 candidates,
3-step GN polish) drops 42-43 / 150 frames outright (`frames_with_board`:
108/150 aprilgrid, 107/150 charuco) because `outer_count < min_inliers`
kills the seed.

**Fix**: fit a single 16-point DLT homography over all sample
correspondences (totals-least-squares, smallest-eigenvalue eigenvector of
the 9×9 `A^T A`) — this averages corner noise across all 4 sampled tags —
then decompose it with `solve_ippe_square` (both Necker branches) instead
of Zhang's `R = SO(3)-project(K⁻¹H)` (which silently collapses to one
branch). This is Pool C: DLT's noise-averaging robustness +
IPPE-Square's two-branch enumeration + joint-reprojection disambiguation.

## Before/after on regression-board-hub snapshots

| Metric | pre-#255 (main) | PR #255 (DLT+Zhang) | This PR (multi-IPPE) |
| :--- | ---: | ---: | ---: |
| **aprilgrid_golden — 150 frames** | | | |
| frames_with_board | 150 | 150 | **150** |
| mean rot ° | 1.570 | 0.120 | **0.118** |
| p99 rot ° | 19.370 | 1.295 | **1.295** |
| mean trans m | 0.0111 | 0.00259 | **0.00265** |
| p99 trans m | 0.214 | 0.01862 | **0.01863** |
| **charuco_golden — 150 frames** | | | |
| frames_with_board | 150 | 150 | **150** |
| mean rot ° | 2.299 | 0.108 | **0.107** |
| p99 rot ° | 56.172 | 0.651 | **0.651** |
| mean trans m | 0.0107 | 0.00224 | **0.00238** |
| p99 trans m | 0.162 | 0.01115 | **0.01111** |

Numerically indistinguishable from PR #255 on both p99 rotation and mean
translation across both boards — the brief's p99 ≈ 1.30° aprilgrid / 0.65°
charuco bound is preserved exactly.

## Rejected variants

- **Pool B only (literal brief spec) + 3-step GN polish**: 108/150
  aprilgrid, 107/150 charuco — insufficient on its own (see above).
- **Pool B only + no polish**: worse — scoring un-polished noisy
  candidates picks the wrong branch more often.
- **Pool A (centroid+identity) + Pool B**: ~108/150 — centroid fallback
  can't bridge a 30°+ rotation gap in 3 GN steps; decorative.

Pool C closes the gap to PR #255 baseline on its own; see the Follow-up
ablation for why Pools A/B were then dropped entirely.
