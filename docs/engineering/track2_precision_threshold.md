# Track 2 — Pose-Consistency Gate Calibration

> Date: 2026-04-26
> Profiles affected: `high_accuracy`, `render_tag_hub`
> Source: `crates/locus-core/src/pose.rs` (`pose_consistency_check`, `select_ippe_branch`)
> Tests: `crates/locus-core/tests/regression_pose_consistency_roc.rs`

## What the gate does

The pose stage has a final, statistically principled reprojection-consistency
check. Given the LM-refined pose, it forms the per-corner Mahalanobis
statistic

```
d²_i = r_iᵀ Σ_i⁻¹ r_i
```

where `r_i` is the residual between the observed (distorted) corner `i` and the
distorted reprojection of the corresponding object-space corner. `Σ_i` is the
per-corner covariance (see *Sources of Σ* below). The aggregate statistic is
`d² = Σ_i d²_i`.

Two gates fire:

| Gate         | DOF | Threshold                    | Catches                                              |
| :----------- | :-- | :--------------------------- | :--------------------------------------------------- |
| Aggregate    | 2   | `χ²(2; fpr)`  = `−2 ln(fpr)` | Globally bad fits (wrong IPPE branch, decoder dirt). |
| Per-corner   | 1   | `χ²(1; fpr)`                 | One contaminated corner the aggregate averages over. |

DOF = 8 observations − 6 fitted pose parameters = 2 for the aggregate; each
individual corner contributes χ²(1) to that sum.

`fpr == 0.0` short-circuits to *accept* — disabled profiles run the legacy
path with byte-identical behaviour. Only `high_accuracy` and `render_tag_hub`
opt in at `1.0e-3`.

## Sources of Σ (per refinement mode)

| Refinement mode | Σ source                                                                              |
| :-------------- | :------------------------------------------------------------------------------------ |
| GWLF            | Per-corner external covariances written into `corner_covariances` SoA column.         |
| Accurate (no GWLF) | `pose_weighted::compute_framework_uncertainty` (Structure-Tensor + sigma_n_sq). |
| Fast            | Isotropic `Σ⁻¹ = (1 / sigma_n_sq) · I` (`sigma_n_sq` defaults to 4 px²).              |

The same `Σ` matrices are passed to (a) the IPPE branch selector, (b) the LM
solver, (c) the consistency check — so the per-frame statistic is internally
consistent.

## IPPE branch handling

When `fpr > 0`, the branch selector runs Mahalanobis-min in *observed*
(distorted) space. It then applies a swap rule:

```
swap to alternate branch  ⇔  alt.d² < 0.5 · primary.d²
                            AND  alt.d² < χ²(2; fpr)
```

Both conditions are required. The 2× margin is intentionally tight so random
jitter near the χ²(2) tail cannot flip the branch. When `fpr == 0`, the legacy
ideal-corner Euclidean comparison runs unchanged.

The `ippe_branch_d2_ratio = alt.d² / primary.d²` telemetry is exposed via the
bench-internals SoA column. `≫ 1` means the primary branch was a clear winner;
`≪ 1` means the primary branch was wrong (the rotation-tail bug pattern).

## Calibration: ROC harness

`tests/regression_pose_consistency_roc.rs` exercises the gate end-to-end with
controlled isotropic noise:

1. Project a known pose through pinhole intrinsics.
2. Add Gaussian(0, σ²) noise (σ = 1 px) per corner.
3. **LM-refit** the pose on the noisy observations, so residuals genuinely
   span 8 obs − 6 fitted = 2 DOF (matching the χ²(2) production model).
4. Run the gate at every sweep `fpr` and count rejections.

A *rejected consistent pose* is a **false positive of the gate itself** — the
realized FPR. The harness asserts that at `fpr = 1e-3` the realized FPR lies
in `[1e-4, 1e-2]` (within one decade of modeled). The test fails the build
if the χ²(2) calibration drifts.

The current snapshot:

| Modeled `fpr` | Realized FPR (n = 20,000) |
| :-----------: | :-----------------------: |
| 1e-1          | 0.169                     |
| 1e-2          | 0.0278                    |
| 1e-3          | 0.0049                    |
| 1e-4          | 0.0008                    |
| 1e-5          | 0.00005                   |

Realized rates run consistently 2–5× *higher* than modeled because (a) the
per-corner gate adds an independent test on top of the aggregate
(Bonferroni-like inflation), and (b) LM converges to a local minimum that
doesn't perfectly null all 6 DOF. This is acceptable for a precision-improving
gate — it errs on the side of rejecting marginal poses.

### What to do if the assertion fires

If a future change pushes realized FPR outside `[1e-4, 1e-2]`:

1. Run `cargo insta test --release --features bench-internals -p locus-core --test regression_pose_consistency_roc --review` and inspect the snapshot diff.
2. If the realized FPR is now too high, the gate is rejecting valid poses — usually because Σ was tightened without lowering the threshold. Either back out the Σ change or move the affected profile to an *empirical_d2_threshold* (a fixed numeric threshold derived from the new realized distribution).
3. If realized FPR is now too low, the gate is too loose — usually because Σ was loosened. Same remedy in the opposite direction.
4. Document the empirical fallback in this file and the affected profile JSON.

## Per-profile defaults

| Profile           | `pose_consistency_fpr` | Rationale                                                                                  |
| :---------------- | :--------------------: | :----------------------------------------------------------------------------------------- |
| `standard`        | 0.0 (disabled)         | General-purpose default. No precision tail to fix at this configuration.                   |
| `grid`            | 0.0 (disabled)         | AprilGrid sub-tags need every recall point we can get; gate would reject distorted corners. |
| `general`         | 0.0 (disabled)         | Same reasoning as `standard`.                                                              |
| `high_accuracy`   | 1.0e-3                 | Metrology preset — precision matters more than recall.                                     |
| `render_tag_hub`  | 1.0e-3                 | Render-tag 1080p had a 1.96 % precision miss + 1.897° rotation p99 tail driven by IPPE branch errors. |

## Latency

The gate adds:

- 8 distorted projections per tag (same `project_with_distortion` used by LM).
- 4 Mahalanobis evaluations (4 × 2×2 multiply-add).
- 2 χ² critical evaluations per branch decision (closed form).

Branch handling adds 8 projections per tag (the alternate-branch evaluation),
totalling ~100 ns on a Zen 4 core for the four-corner pinhole case. Distorted
projection is ~3× more expensive but still < 1 µs per tag at the gate level.
The micro-bench (`benches/pose_bench.rs`, arms `bench_pose_gate_*`) reports
the four-cell matrix `{pinhole, brown_conrady} × {gate off, gate on}` so the
distortion overhead is separately visible.

## Out of scope (deferred)

- Inlier-only pose re-fit when one corner trips the per-corner gate.
- LM-convergence diagnostics as a third independent gate.
- Per-family `pose_consistency_fpr` defaults.
- Multi-source ROC sweep (ICRA + Hub negative-recall delta) — recall deltas at the production threshold are already covered by the existing `regression_icra2020`, `regression_render_tag`, and `regression_distortion_hub` snapshots.
