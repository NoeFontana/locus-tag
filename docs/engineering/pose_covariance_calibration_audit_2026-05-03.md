# Pose-Covariance Calibration Audit — 2026-05-03

This memo documents an empirical calibration audit of the 6×6 covariance
returned by `refine_pose_lm_weighted` (the production "Accurate" pose solver
in `crates/locus-core/src/pose_weighted.rs`). Customers using Locus poses
for downstream Kalman / factor-graph fusion need this covariance to be
honest. It currently is not.

The harness (`tools/bench/pose_cov_audit.py`) runs the production
detector on the hub 1080p tag36h11 corpus, extracts the 6×6 covariance via
the bench-internals `refine_pose_lm_weighted_with_telemetry` entry-point,
and compares the empirical squared 6-DOF Mahalanobis distance against the
theoretical χ²(6) reference distribution.

## 1. Verified hardware metadata (per `constraints.md` §6)

The audit was executed in this same session.

```text
$ lscpu | grep -E 'Model name|Architecture|CPU\(s\):|Thread|Vendor|cache'
Architecture:                            x86_64
CPU(s):                                  8
Vendor ID:                               AuthenticAMD
Model name:                              AMD EPYC-Milan Processor
Thread(s) per core:                      2
L1d cache:                               128 KiB (4 instances)
L1i cache:                               128 KiB (4 instances)
L2 cache:                                2 MiB (4 instances)
L3 cache:                                32 MiB (1 instance)
```

| Item                  | Value                                                 |
| :-------------------- | :---------------------------------------------------- |
| CPU                   | AMD EPYC-Milan, x86_64, 8 logical CPUs, 2 SMT threads |
| AVX support           | avx, avx2, fma, f16c (AVX-512 absent)                 |
| Build profile         | `--release` (`cargo build --release` via maturin)     |
| Cargo features        | `bench-internals`                                     |
| Rust toolchain        | `rustc 1.92.0 (ded5c06cf 2025-12-08)`                 |
| `RAYON_NUM_THREADS`   | `8`                                                   |
| Python                | `CPython 3.14.3` (uv venv)                            |
| Locus profile         | `high_accuracy` (50 scenes, single tag36h11 per scene)|
| Hub config            | `locus_v1_tag36h11_1920x1080`                         |
| Pose mode             | `PoseEstimationMode.Accurate`                         |
| Detector              | `detect()` (single-frame, full pipeline)              |
| `sigma_n_sq`          | 4.0 px²                                               |
| `tikhonov_alpha_max`  | 0.25                                                  |
| `structure_tensor_radius` | 2 px                                              |

Raw outputs: `diagnostics/pose_cov_audit_2026-05-03/`
(`report.json`, `samples.json`, `qq_plot.png`, `histogram.png`).

## 2. Mahalanobis distribution — empirical vs. χ²(6)

The 6-DOF residual is

    δ = [Δt; Δθ]
    Δt = R_gtᵀ (t_det − t_gt)        ∈ ℝ³
    Δθ = log_SO3(R_gtᵀ R_det)         ∈ ℝ³

so `δ ∈ ℝ⁶` lives in the GT tangent space, parameter-aligned with the
solver's 6×6 Jacobian (translation block 0..2, axis-angle 3..5). The
audited statistic is

    d² = δᵀ Σ⁻¹ δ

with `Σ` the 36-float row-major covariance returned by
`refine_pose_lm_weighted_with_telemetry`. Under correct calibration,
`d² ∼ χ²(6)` — mean 6, variance 12.

Across all 50 hub scenes (50/50 detected, no LM singularities):

| Statistic                     | Empirical          | χ²(6) ideal | Ratio  |
| :---------------------------- | -----------------: | ----------: | -----: |
| Mean d²                       | **714.72**         | 6.00        | 119×   |
| Variance d²                   | **4 081 616.6**    | 12.00       | 340 000× |
| p10 d²                        | 6.41               | 2.20        | 2.9×   |
| p50 d²                        | 62.76              | 5.35        | 11.7×  |
| p90 d²                        | 1635.95            | 10.64       | 154×   |
| p99 d²                        | 9435.32            | 16.81       | 561×   |
| KL(empirical ‖ χ²(6))         | **13.93** (nats)   | 0           | —      |

The KL divergence is **>> 0.5**, putting the covariance firmly in the
"miscalibrated" bucket per the gating rubric (≤ 0.1 well-calibrated,
0.1–0.5 drift, > 0.5 miscalibrated). The Q-Q plot (`qq_plot.png`) shows
the empirical curve climbs ~30× faster than `y = x` past the 70th
percentile — i.e. there is a heavy right tail driven by a handful of
scenes where Σ is severely overconfident along one or two principal
axes of the Hessian.

### Eigen-decomposed view (key insight)

Diagonalising each per-scene Σ and binning the d² contribution by
eigendirection (sorted ascending, so eigvec[0] is the *stiffest* — the
direction Σ claims is most certain):

| Eigendirection (ascending λ) | Mean d² | Median d² |
| :--------------------------- | ------: | --------: |
| eig[0] (stiffest)            | 302.06  | 11.63     |
| eig[1]                       | 411.78  | 29.92     |
| eig[2]                       | 0.19    | 0.07      |
| eig[3]                       | 0.62    | 0.11      |
| eig[4]                       | 0.03    | 0.006     |
| eig[5] (softest)             | 0.02    | 0.005     |

Each eigendirection contributes 1.0 d²-units under correct calibration.
**99% of the d² mass concentrates in the two stiffest directions** — Σ's
two smallest eigenvalues understate the true error along those axes by a
factor of `√300 ≈ 17×`. Conversely, the four softest directions over-state
uncertainty (median d² < 0.1 → Σ is roughly 3–10× too loose). The
covariance is thus not uniformly "too small" — it is *anisotropically*
miscalibrated.

## 3. Per-axis breakdown (1-DOF diagonal slice)

Each axis treated independently as `δᵢ² / Σᵢᵢ` (ideal mean 1.0):

| Axis | Empirical mean d² | Calibration ratio | Verdict (diagonal-only)            |
| :--- | ----------------: | ----------------: | :--------------------------------- |
| tx   | 0.026             | 0.026             | over-estimating uncertainty (Σᵢᵢ too large) |
| ty   | 0.021             | 0.021             | over-estimating uncertainty (Σᵢᵢ too large) |
| tz   | 0.274             | 0.274             | over-estimating uncertainty (Σᵢᵢ too large) |
| rx   | 0.094             | 0.094             | over-estimating uncertainty (Σᵢᵢ too large) |
| ry   | 0.154             | 0.154             | over-estimating uncertainty (Σᵢᵢ too large) |
| rz   | 0.169             | 0.169             | over-estimating uncertainty (Σᵢᵢ too large) |

Read carefully — the **diagonal** Σᵢᵢ values are loose (the residuals
δᵢ are smaller than 1σ along every cardinal axis), but the **full**
inverse Σ⁻¹ is overconfident along the principal stiffness axes
(see eigen-decomposition above). This is a textbook signature of a
covariance whose **off-diagonal correlations are wrong**: the Hessian
has ~7-orders-of-magnitude eigenvalue spread (e.g. λ_min ≈ 1e-8,
λ_max ≈ 3e-4 for a typical scene), and the principal axes don't align
with translation/rotation cardinals — they live in mixed translation-
rotation directions where the LM has fitted very tightly to corner pixel
locations but the *world* error has a different correlation structure.

## 4. Most likely formula correction

The miscalibration mass (KL = 13.9, 99% concentrated in the two
stiffest principal directions) points squarely at `sigma_n_sq` and the
Tikhonov regulariser. Three candidate corrections, ranked:

1. **`sigma_n_sq = 4.0 px²` is too low by ~30×.** It lives in
   `crates/locus-core/profiles/high_accuracy.json:45` and feeds
   `compute_corner_covariance` at
   `crates/locus-core/src/pose_weighted.rs:154`:
   ```rust
   s_inv.scale(sigma_n_sq) + Matrix2::identity().scale(alpha)
   ```
   The audit's mean d² is 715 ≈ `sigma_n_sq` undercount factor squared
   times 6; raising `sigma_n_sq` to ~ `30² × 4 / 715 ≈ 5–8 px²` *along
   the two stiff axes* would not help by itself because the axes also
   need correlation surgery — but raising it would compress the per-axis
   ratios closer to 1. **Recommend running the harness twice with
   `sigma_n_sq ∈ {16, 25}` and tracking the mean d² and KL.** This is
   probably the cheapest fix.

2. **`tikhonov_alpha_max = 0.25` is far too small** to dominate
   `s_inv.scale(sigma_n_sq)` when the structure tensor is well-conditioned
   (sharp tag-corner). The Tikhonov term in
   `pose_weighted.rs:152-154`:
   ```rust
   let alpha = alpha_max * (1.0 - r).powi(2);
   …
   s_inv.scale(sigma_n_sq) + Matrix2::identity().scale(alpha)
   ```
   evaluates to `~0.0025` for an isotropic corner (`r ≈ 0.9`), so it
   contributes essentially nothing to Σ_corner. Yet *unmodelled* error
   sources — discretisation, motion blur, render-PSF — easily dominate
   ~1 px standard deviation. **Recommend bumping `tikhonov_alpha_max`
   to ~1.0 px²** so the floor catches systematic corner-location bias
   that the structure tensor cannot see.

3. **The covariance return path skips the Tikhonov ε on the 6×6
   Hessian.** At `pose_weighted.rs:456`:
   ```rust
   let covariance = current_jtj.try_inverse().unwrap_or_else(Matrix6::identity);
   ```
   This inverts the *un-damped* `JᵀWJ`. The LM solver damps it during the
   step (line 388: `jtj_damped[(k, k)] += lambda * current_jtj[(k, k)].max(1e-6)`)
   but the returned covariance does not. For under-determined geometry
   (small tag, near-perpendicular view), one or two Hessian eigenvalues
   collapse, the inverse blows up, and the principal-axis variance is
   wildly overestimated *for those scenes only* — the very scenes with
   d² > 1000. **Recommend retaining the final accepted `lambda` and
   returning `(JᵀWJ + λ · diag(JᵀWJ))⁻¹`** as a Tikhonov-stabilised
   covariance, matching the regularisation the solver itself used.

The cheapest first experiment is (2)+(3): they are config-only changes
that should compress the right tail without touching the kernel math.
(1) requires re-blessing snapshots and is best deferred to a follow-up.

## 5. Follow-up TODO

KL = 13.9 ≫ 0.1 — file follow-up track to apply correction (3) (Tikhonov
ε on the returned Hessian inverse, `pose_weighted.rs:456`) plus correction
(2) (`tikhonov_alpha_max` bump from 0.25 to ~1.0 in
`crates/locus-core/profiles/high_accuracy.json:44`). Re-run this audit
after the change and target KL ≤ 0.1 with mean d² ∈ [4, 9] and per-axis
calibration ratios ∈ [0.5, 2.0].

## 6. Reproducing the audit

```bash
RAYON_NUM_THREADS=8 \
PYTHONPATH=. \
uv run --group bench tools/bench/pose_cov_audit.py \
    --hub-config locus_v1_tag36h11_1920x1080 \
    --rayon-threads 8
```

Outputs land in `diagnostics/pose_cov_audit_<ISO-date>/`:

- `report.json` — aggregate stats + per-axis breakdown.
- `samples.json` — per-scene δ, Σ, d² (debug-friendly).
- `qq_plot.png` — empirical d² quantiles vs. χ²(6).
- `histogram.png` — empirical PMF overlaid on χ²(6).

Lint/type gates:

```bash
uv run --group lint --group bench    ruff check     tools/bench/pose_cov_audit.py
uv run --group types --group bench --group etl basedpyright tools/bench/pose_cov_audit.py
```

Both pass clean as of this commit.
