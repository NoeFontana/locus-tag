# ERF Refinement SIMD Rewrite — Abandoned

**Date:** 2026-04-19
**Branch (dropped):** `perf/edge-refinement-simd`
**Plan:** historical session notes, not checked in (Phase 1.5)

## Outcome

Abandoned. The branch regressed all four primary bench axes vs the
`feat/unified-edge-refinement` baseline (quad +55–117%, decoder +15–28%,
`collect_samples` +349%) and a partial revert could not close the gap
without undoing the thesis commit.

## Root cause

The SoA refactor (commit `d109c8d`, `AoS (f64,f64,f64) → Samples { xs, ys, is }`)
was the regression source — confirmed by bisect. The later commits
(branchless range-mask, vectorized `exp(-s²)`, u8 gather, NEON bodies)
were correctness-neutral on top of SoA but couldn't recover the loss.

Why SoA lost on this workload:

1. **`collect_samples` is push-bound.** Every `push(x, y, i)` now touches
   three cache lines instead of one, so store bandwidth tripled.
2. **Typical edges yield ~300 samples per lane** — with `BumpVec` at
   `with_capacity_in(128, arena)`, this triggered 2–3 reallocations per
   lane per edge. Scan-geometry-sized `MaybeUninit` slabs helped but did
   not fully close the gap.
3. **The refine-side load win (`_mm256_loadu_pd` vs `_mm256_set_pd`) is
   real but small** — the GN loop runs 3–15 iterations per edge, but the
   collect loop runs once over every scanned pixel. Collect dominates.

## What was tried and discarded

- Restored movemask early-exit in `refine_accumulate` (AVX2 + NEON).
- Restored scalar libm `exp` in `exp_neg_sq_v4` (reverted vectorized exp).
- Reverted u8 vector gather to per-lane scalar read.
- `SamplesBuilder` rewritten to use `MaybeUninit<f64>` slabs (no zero-init).
- Scan-geometry-sized capacity allocation.

None of the above restored parity — SoA `push` itself is the floor.

## What was tried on AoS (post-revert) and also discarded

- **`estimate_ab_per_iter` vectorized via `sample_bilinear_v8`**:
  regressed decoder by ~3%. f32 round-trip overhead exceeded the AVX2
  gather gain at 60–90 sample chunks. Reverted.

## Lessons

1. **Bench first, refactor second.** The SoA thesis was plausible on
   paper (`_mm256_loadu_pd` vs gather) but never measured standalone
   before the downstream optimizations were built on top. A single
   mechanical SoA commit with no behavior change should have been
   benched in isolation before building the rest of the stack.
2. **Push-heavy loops fight SoA.** SoA wins for streams that are written
   once and read many times. `collect_samples` writes once and reads
   twice (gradient check + refine). The read wins didn't amortize.
3. **Small-chunk SIMD dispatch has an overhead floor.** `sample_bilinear_v8`
   wins on long streams (grid sampling, thousands of points) but loses
   on the 60–90-sample chunks typical of per-edge refinement. The f64↔f32
   round-trip and the mask-load ceremony dominate at this scale.
4. **"NEON parity" is not free.** Adding a 200-LOC NEON body requires
   aarch64 bench validation. Without it, it's speculation that also
   adds maintenance burden on x86_64 (more conditional branches, more
   code to keep consistent with the AVX2 path).

## What was salvageable

Nothing landed. The isolated wins either regressed (SoA, `estimate_ab`)
or were infrastructure with no production caller (`exp_approx_v4/v2`,
`exp_neg_sq_v4/v2`). Future perf work on ERF refinement should either:

- Target the collect-side directly on the AoS layout (e.g., SIMD u8
  gather into `(f64, f64, f64)` tuples via intrinsics + scalar store).
- Or replace the whole algorithm (e.g., integer-pixel scan + sub-pixel
  parabolic fit) rather than vectorize the existing formulation.

## Baseline (pre-attempt, for reference)

| Bench | Median |
|---|---|
| `decoder_style_sigma_0_6` | 5.81 ms |
| `decoder_style_sigma_0_8` | 6.14 ms |
| `quad_style_sigma_0_6` | 1.13 ms |
| `quad_style_sigma_0_8` | 1.19 ms |

Measured on `feat/unified-edge-refinement @ 2b42a83`, release profile,
single thread, x86_64 AVX2+FMA+BMI2.
