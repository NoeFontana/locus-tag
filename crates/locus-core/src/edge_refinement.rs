//! Unified sub-pixel edge refinement using the Error Function (ERF) intensity model.
//!
//! This module centralizes the Gauss-Newton optimization for fitting a PSF-blurred
//! step function to image intensities along a line:
//!
//! $$I(x,y) = \frac{A+B}{2} + \frac{B-A}{2} \cdot \mathrm{erf}\!\left(\frac{d}{\sigma}\right)$$
//!
//! where $A,B$ are the dark/light intensities, $d$ is the signed perpendicular distance
//! to the edge line, and $\sigma$ is the Gaussian blur parameter.
//!
//! Both the quad extraction stage and the decoder refinement stage share this solver,
//! ensuring consistent Jacobian evaluation and benefiting from the same SIMD acceleration.

#![allow(unsafe_code, clippy::cast_sign_loss)]

use crate::image::ImageView;
use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;
use multiversion::multiversion;

/// Precomputed `1 / sqrt(pi)` — the Jacobian of `erf(x)` at `x` is
/// `(2 / sqrt(pi)) * exp(-x^2)`; factoring the constant saves a `sqrt` per GN iteration.
const INV_SQRT_PI: f64 = 0.564_189_583_547_756_3;

/// Configuration for sample collection along an edge.
#[derive(Clone, Debug)]
pub struct SampleConfig {
    /// Half-width of the search band perpendicular to the edge (in pixels).
    pub window: f64,
    /// Pixel stride for subsampling (1 = every pixel, 2 = skip alternate).
    pub stride: usize,
    /// Parametric range `(t_min, t_max)` along the edge segment for inclusion.
    /// `t=0` at `p1`, `t=1` at `p2`. Values outside `[0,1]` extend beyond endpoints.
    pub t_range: (f64, f64),
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            window: 2.5,
            stride: 1,
            t_range: (-0.1, 1.1),
        }
    }
}

impl SampleConfig {
    /// Sampling config for the quad-extraction path: widen the window under
    /// decimation, stride through very long edges, and clip `t_range` tightly
    /// enough that samples from the adjacent leg of an L-corner don't leak in.
    #[must_use]
    pub fn for_quad(edge_len: f64, decimation: usize) -> Self {
        let window = if decimation > 1 {
            decimation as f64 + 1.0
        } else {
            2.5
        };
        let stride = if edge_len > 100.0 { 2 } else { 1 };
        Self {
            window,
            stride,
            t_range: (-0.1, 1.1),
        }
    }

    /// Narrow variant of [`Self::for_quad`] for the 2-DOF IRLS path.
    /// Excludes adjacent-edge / bit-boundary samples geometrically (long
    /// edges have the budget to spare). Short edges keep the wide band —
    /// narrowing further starves the GN below the 10-sample floor.
    #[must_use]
    pub fn for_quad_narrow(edge_len: f64, decimation: usize) -> Self {
        let mut cfg = Self::for_quad(edge_len, decimation);
        if decimation > 1 {
            return cfg;
        }
        if edge_len >= 60.0 {
            cfg.window = 1.5;
            cfg.t_range = (0.05, 0.95);
        } else if edge_len >= 30.0 {
            cfg.window = 2.0;
            cfg.t_range = (0.0, 1.0);
        }
        cfg
    }

    /// Sampling config for the decoder ERF path (same as [`SampleConfig::default`]).
    #[must_use]
    pub fn for_decoder() -> Self {
        Self::default()
    }
}

/// Gauss-Newton refinement mode.
///
/// `TwoDofTukey` is the production path for quad-extraction: 2-DOF `(θ, ρ)`
/// GN with Tukey IRLS and a per-iter unweighted σ̂ pre-pass. `TwoDof` (no
/// IRLS) is exercised by unit tests of the conditioning guard.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum RefineMode {
    /// Solve only the perpendicular offset; freeze the seed normal.
    #[default]
    OneDof,
    /// Unweighted 2-DOF `(θ, ρ)` GN — test-only.
    #[allow(dead_code)]
    TwoDof,
    /// 2-DOF GN with Tukey IRLS and same-iter σ̂ pre-pass — the production
    /// quad-extraction path. See
    /// `docs/explanation/edge_refinement_2dof_failure_analysis.md`.
    TwoDofTukey,
}

impl RefineMode {
    #[inline]
    pub(crate) fn is_two_dof(self) -> bool {
        !matches!(self, RefineMode::OneDof)
    }
}

/// Configuration for the Gauss-Newton refinement loop.
#[derive(Clone, Debug)]
pub struct RefineConfig {
    /// Gaussian blur sigma for the ERF model.
    pub sigma: f64,
    /// Maximum number of Gauss-Newton iterations.
    pub max_iterations: usize,
    /// If true, re-estimate A/B intensities each iteration (decoder style).
    /// If false, estimate A/B once before the GN loop (quad style).
    pub re_estimate_ab: bool,
    /// If true, pre-scan initial `d` by maximizing gradient alignment before GN.
    pub scan_initial: bool,
    /// Convergence threshold: stop if |step| < this value.
    pub convergence_threshold: f64,
    /// Singularity threshold for J^T J.
    pub singular_threshold: f64,
    /// Maximum step size per iteration (clamped).
    pub step_clamp: f64,
    /// Minimum |B - A| contrast; break early if below this.
    pub min_contrast: f64,
    pub(crate) mode: RefineMode,
}

/// Gradient-weighted tangential variance (px²) below which the 2-DOF solver
/// falls back to a 1-DOF ρ-only step for that iteration — calibrated so
/// edges with < ~1 px effective tangential lever arm don't rotate on noise.
const THETA_OBSERVABILITY_FLOOR_PX2: f64 = 1.0;

/// Per-iteration rotation clamp (radians). `0.05 rad ≈ 2.86°` sits inside
/// typical seed-misalignment range and below ERF Jacobian linearization
/// breakdown.
const THETA_CLAMP_RAD: f64 = 0.05;

/// Total-rotation deadband (radians). `≈ 0.40°`. When the 2-DOF solve's
/// converged normal rotated less than this from the seed, the rotation was
/// not warranted; the solve falls back to the 1-DOF ρ-only estimator to
/// avoid the joint solve's small ρ bias on well-aligned edges. Sits an order
/// of magnitude above the spurious sub-0.1° rotations seen on clean corners
/// and an order below the 1–3° misrotations the 2-DOF path exists to correct.
const THETA_DEADBAND_RAD: f64 = 0.007;

/// The 2-DOF rotation is accepted over the 1-DOF reference only when its
/// robust edge-fit cost is below this fraction of the 1-DOF cost (i.e. a
/// ≥3 % RMS improvement). This is a *relative* ratio of two costs measured
/// over the **same** sample set with the **same** bounded-influence aggregate
/// (see [`robust_edge_cost`]), so it is scale-independent and penalises the
/// extra rotational DOF: a marginally-better fit that drifts the corner onto
/// adjacent tag structure is rejected, while the large cost drops from
/// correcting a genuinely misrotated seed are kept.
const TWO_DOF_ACCEPT_COST_FRAC: f64 = 0.97;

/// Per-sample residual influence is clipped at `(this · |B − A|)²` in
/// [`robust_edge_cost`]. Bounds the leverage of a minority of off-edge /
/// adjacent-structure samples so the 1-DOF-vs-2-DOF comparison is fair to
/// both the seed-frozen and the rotated fit (a correct rotation moves *away*
/// from contamination; without clipping those samples would unfairly inflate
/// the rotated fit's cost).
const TWO_DOF_COST_CLIP_FRAC: f64 = 0.5;

/// Tukey biweight tuning constant — 95 % Gaussian efficiency.
const TUKEY_C: f64 = 4.685;

/// Floor on σ̂ as a fraction of `|B − A|`. Protects perfect-fit edges from
/// σ̂ → 0 driving every weight to zero.
const ROBUST_SIGMA_FLOOR_KAPPA: f64 = 0.01;

impl RefineConfig {
    /// One-shot A/B estimation, no minimum contrast check, no initial `d` scan.
    /// Used by the quad extraction corner-refinement path.
    #[inline]
    #[must_use]
    pub fn quad_style(sigma: f64) -> Self {
        // 2-DOF + Tukey IRLS + same-iter σ̂ + narrow sample band. Pairs
        // with `refinement::refine_corner`'s 3.0 px displacement gate.
        // Strict Pareto is not met (4–27 % recall regression on noisy
        // corpora; board_charuco P99 ~+83 %), but mean and tail accuracy
        // wins (P99 rotation −40–99 %, ICRA RMSE −38 %) ship the
        // configuration. See
        // `docs/explanation/edge_refinement_2dof_failure_analysis.md`.
        Self {
            sigma,
            max_iterations: 15,
            re_estimate_ab: false,
            scan_initial: false,
            convergence_threshold: 1e-4,
            singular_threshold: 1e-10,
            step_clamp: 0.5,
            min_contrast: 0.0,
            mode: RefineMode::TwoDofTukey,
        }
    }

    /// Per-iteration A/B refinement, pre-refine scan, early exit on low contrast.
    /// Used by the decoder corner-refinement path.
    #[inline]
    #[must_use]
    pub fn decoder_style(sigma: f64) -> Self {
        Self {
            sigma,
            max_iterations: 15,
            re_estimate_ab: true,
            scan_initial: true,
            convergence_threshold: 1e-4,
            singular_threshold: 1e-6,
            step_clamp: 0.5,
            min_contrast: 5.0,
            mode: RefineMode::OneDof,
        }
    }

    /// Post-decode edge-fit style. Identical to [`Self::decoder_style`] except
    /// `scan_initial` is off (decoded corners are already sub-pixel after
    /// Phase A) and `singular_threshold` is tighter (no scan fallback if the
    /// solve is ill-conditioned). Used by Phase C.5 corner re-refinement.
    #[inline]
    #[must_use]
    pub fn post_decode_style(sigma: f64) -> Self {
        Self {
            scan_initial: false,
            singular_threshold: 1e-10,
            ..Self::decoder_style(sigma)
        }
    }
}

/// Unified ERF-based edge fitter.
///
/// Encapsulates the line parameters `(nx, ny, d)` defining the implicit line
/// `nx * x + ny * y + d = 0`, along with edge geometry for sample collection.
///
/// # Normal Convention
///
/// The normal `(nx, ny)` points from the dark side toward the light side.
/// The convention is `nx = -dy/len, ny = dx/len` (left-hand normal of the
/// directed edge p1 → p2 for CW-wound quads).
pub struct ErfEdgeFitter<'a> {
    img: &'a ImageView<'a>,
    p1: [f64; 2],
    dx: f64,
    dy: f64,
    len: f64,
    nx: f64,
    ny: f64,
    d: f64,
    last_jtj: f64,
    /// True when the 2-DOF conditioning guard fell back to a 1-DOF step
    /// during the most recent [`Self::refine`] call. Test telemetry.
    last_guard_fired: bool,
    /// IRLS scale estimate `σ̂` from the final 2-DOF iteration. `NaN` when
    /// IRLS was not run. Test telemetry.
    last_sigma_hat: f64,
}

impl<'a> ErfEdgeFitter<'a> {
    /// Create a new fitter for the edge from `p1` to `p2`.
    ///
    /// Returns `None` if the edge is shorter than 4 pixels.
    ///
    /// # Normal Convention
    ///
    /// `init_from_midpoint = true`: computes d from the edge midpoint (quad-style).
    /// `init_from_midpoint = false`: computes d from p1 (decoder-style).
    #[must_use]
    pub fn new(
        img: &'a ImageView<'a>,
        p1: [f64; 2],
        p2: [f64; 2],
        init_from_midpoint: bool,
    ) -> Option<Self> {
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len < 4.0 {
            return None;
        }

        // Consistent normal convention: left-hand normal of directed edge p1→p2.
        let nx = -dy / len;
        let ny = dx / len;

        let d = if init_from_midpoint {
            let mid_x = (p1[0] + p2[0]) * 0.5;
            let mid_y = (p1[1] + p2[1]) * 0.5;
            -(nx * mid_x + ny * mid_y)
        } else {
            -(nx * p1[0] + ny * p1[1])
        };

        Some(Self {
            img,
            p1,
            dx,
            dy,
            len,
            nx,
            ny,
            d,
            last_jtj: 0.0,
            last_guard_fired: false,
            last_sigma_hat: f64::NAN,
        })
    }

    /// Scan for the optimal initial `d` by maximizing gradient alignment.
    ///
    /// Tests 13 offsets in `[-2.4, +2.4]` pixels and picks the one where the
    /// projected gradient magnitude along the normal is highest.
    pub fn scan_initial_d(&mut self) {
        let window = 2.5;
        let (x0, x1, y0, y1) = self.get_scan_bounds(window);

        let mut best_offset = 0.0;
        let mut best_grad = 0.0;

        for k in -6..=6 {
            let offset = f64::from(k) * 0.4;
            let scan_d = self.d + offset;

            let (sum_g, count) =
                project_gradients_optimized(self.img, self.nx, self.ny, x0, x1, y0, y1, scan_d);

            if count > 0 && sum_g > best_grad {
                best_grad = sum_g;
                best_offset = offset;
            }
        }
        self.d += best_offset;
    }

    /// Collect pixel samples within a window of the edge.
    ///
    /// Returns arena-allocated tuples `(x, y, intensity)` for pixels that fall
    /// within the perpendicular band and parametric range of the edge segment.
    #[must_use]
    pub fn collect_samples(
        &self,
        arena: &'a Bump,
        config: &SampleConfig,
    ) -> BumpVec<'a, (f64, f64, f64)> {
        let (x0, x1, y0, y1) = self.get_scan_bounds(config.window);

        if config.stride == 1 {
            // Fast SIMD path (no subsampling)
            collect_samples_optimized(
                self.img,
                self.nx,
                self.ny,
                self.d,
                self.p1,
                self.dx,
                self.dy,
                self.len,
                x0,
                x1,
                y0,
                y1,
                config.window,
                config.t_range,
                arena,
            )
        } else {
            // Subsampled path (for large edges in decimated images)
            collect_samples_strided(
                self.img,
                self.nx,
                self.ny,
                self.d,
                self.p1,
                self.dx,
                self.dy,
                self.len,
                x0,
                x1,
                y0,
                y1,
                config.window,
                config.t_range,
                config.stride,
                arena,
            )
        }
    }

    /// Run the Gauss-Newton refinement on the collected samples.
    ///
    /// Updates `self.d` in place to minimize the residual between observed
    /// intensities and the ERF model.
    pub fn refine(&mut self, samples: &[(f64, f64, f64)], config: &RefineConfig) {
        self.last_guard_fired = false;
        self.last_sigma_hat = f64::NAN;
        if samples.len() < 10 {
            return;
        }

        let inv_sigma = 1.0 / config.sigma;

        let (a, b) = if config.re_estimate_ab {
            (128.0, 128.0)
        } else {
            estimate_ab_oneshot(samples, self.nx, self.ny, self.d)
        };

        if !config.re_estimate_ab && (b - a).abs() < 1.0 {
            return;
        }

        match config.mode {
            RefineMode::OneDof => self.refine_one_dof(samples, config, inv_sigma, a, b),
            RefineMode::TwoDof | RefineMode::TwoDofTukey => {
                let use_tukey = matches!(config.mode, RefineMode::TwoDofTukey);
                self.refine_two_dof(samples, config, inv_sigma, a, b, use_tukey);
            },
        }
    }

    fn refine_one_dof(
        &mut self,
        samples: &[(f64, f64, f64)],
        config: &RefineConfig,
        inv_sigma: f64,
        mut a: f64,
        mut b: f64,
    ) {
        for _ in 0..config.max_iterations {
            // Per-iteration A/B refinement (decoder style).
            // Preserve previous (a, b) on zero-weight iterations, matching legacy.
            if config.re_estimate_ab
                && let Some((new_a, new_b)) =
                    estimate_ab_per_iter(self.img, samples, self.nx, self.ny, self.d)
            {
                a = new_a;
                b = new_b;
            }

            if config.min_contrast > 0.0 && (b - a).abs() < config.min_contrast {
                break;
            }

            let (sum_jtj, sum_jt_res) =
                refine_accumulate_optimized(samples, self.nx, self.ny, self.d, a, b, inv_sigma);
            self.last_jtj = sum_jtj;

            if sum_jtj < config.singular_threshold {
                break;
            }

            let step = sum_jt_res / sum_jtj;
            self.d += step.clamp(-config.step_clamp, config.step_clamp);

            if step.abs() < config.convergence_threshold {
                break;
            }
        }
    }

    /// 2-DOF Gauss-Newton over `(θ, ρ)` centered at the edge midpoint.
    ///
    /// `use_tukey = true` enables IRLS Tukey reweighting with a same-iter
    /// σ̂ pre-pass: each iter computes σ̂ from an unweighted residuals-
    /// only pass at the current line params, then re-accumulates with the
    /// Tukey weights derived from that σ̂. This eliminates the iter-0
    /// unweighted GN step that breaks `moments_culling`. Falls back to a
    /// 1-DOF ρ-only step whenever the conditioning guard fires.
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    fn refine_two_dof(
        &mut self,
        samples: &[(f64, f64, f64)],
        config: &RefineConfig,
        inv_sigma: f64,
        mut a: f64,
        mut b: f64,
        use_tukey: bool,
    ) {
        // Centered parameterization (`p_c` = edge midpoint) keeps
        // `mean(t_i) ≈ 0` so the JᵀJ cross-term shrinks and the
        // conditioning guard's variance estimate is meaningful.
        let pcx = self.p1[0] + 0.5 * self.dx;
        let pcy = self.p1[1] + 0.5 * self.dy;

        let (seed_nx, seed_ny, seed_d) = (self.nx, self.ny, self.d);
        let (seed_a, seed_b) = (a, b);
        let contrast = (seed_b - seed_a).abs();
        let clip = TWO_DOF_COST_CLIP_FRAC * contrast;
        self.last_sigma_hat = f64::NAN;

        // Proven 1-DOF ρ-only reference first: it is both the default output
        // and the baseline the 2-DOF rotation must beat. `refine_one_dof`
        // leaves the seed normal frozen, so this is byte-identical to the
        // `OneDof` path.
        //
        // (Efficiency follow-up: a 1-DOF-first *skip* — bail out here when the
        // 1-DOF fit is already good, avoiding the 2-DOF GN loop on the common
        // well-aligned edge — is possible, but an absolute `cost < frac ·
        // contrast` trigger can forgo a genuinely-acceptable 2-DOF improvement
        // at the margin. The robust gate below already rejects marginal
        // rotations, so the skip is pure compute savings, not accuracy; it
        // needs a noise-relative trigger to be safe and is left as a follow-up.
        // We instead always run the 2-DOF solve and let the gate decide.)
        self.refine_one_dof(samples, config, inv_sigma, seed_a, seed_b);
        let (one_nx, one_ny, one_d, one_jtj) = (self.nx, self.ny, self.d, self.last_jtj);
        let cost_1dof = robust_edge_cost(
            samples, one_nx, one_ny, one_d, seed_a, seed_b, inv_sigma, clip,
        );

        // --- 2-DOF (θ, ρ) Gauss-Newton from the seed. ---
        self.nx = seed_nx;
        self.ny = seed_ny;
        self.d = seed_d;
        let mut rho = seed_nx * pcx + seed_ny * pcy + seed_d;
        let mut sigma_hat = f64::NAN;

        for _ in 0..config.max_iterations {
            if config.re_estimate_ab {
                let d_cur = rho - self.nx * pcx - self.ny * pcy;
                if let Some((new_a, new_b)) =
                    estimate_ab_per_iter(self.img, samples, self.nx, self.ny, d_cur)
                {
                    a = new_a;
                    b = new_b;
                }
            }

            if config.min_contrast > 0.0 && (b - a).abs() < config.min_contrast {
                break;
            }

            // Same-iter σ̂ pre-pass (Tukey only): the unweighted residual
            // scale at the *current* line params seeds this iteration's IRLS
            // weights. This is load-bearing — it is what stops the iter-0
            // unweighted GN step from rotating onto an adjacent edge (the
            // 104° `moments_culling` failure the ablation diagnosed); it must
            // NOT be replaced by a one-iteration-lagged σ̂. The unweighted
            // path (`use_tukey == false`, the test-only baseline) passes
            // `inv_c_sigma = 0`, which makes every Tukey weight `(1 − 0)² = 1`
            // — one kernel serves both, byte-identically.
            let inv_c_sigma = if use_tukey {
                let sigma_floor = ROBUST_SIGMA_FLOOR_KAPPA * (b - a).abs();
                let s = sigma_at(samples, self.nx, self.ny, rho, pcx, pcy, a, b, inv_sigma)
                    .max(sigma_floor);
                sigma_hat = s;
                1.0 / (TUKEY_C * s)
            } else {
                0.0
            };
            let accum = refine_accumulate_optimized_2dof_tukey(
                samples,
                self.nx,
                self.ny,
                rho,
                pcx,
                pcy,
                a,
                b,
                inv_sigma,
                inv_c_sigma,
            );
            self.last_jtj = accum.jtj_rr;

            if accum.jtj_rr < config.singular_threshold {
                break;
            }

            let det = accum.jtj_rr * accum.jtj_tt - accum.jtj_rt * accum.jtj_rt;
            let inv_rr = 1.0 / accum.jtj_rr;
            let mean_t = accum.jtj_rt * inv_rr;
            let var_t = accum.jtj_tt * inv_rr - mean_t * mean_t;

            let well_conditioned =
                var_t >= THETA_OBSERVABILITY_FLOOR_PX2 && det >= config.singular_threshold;

            let (delta_rho, delta_theta) = if well_conditioned {
                let inv_det = 1.0 / det;
                (
                    (accum.jtj_tt * accum.jtr_r - accum.jtj_rt * accum.jtr_t) * inv_det,
                    (-accum.jtj_rt * accum.jtr_r + accum.jtj_rr * accum.jtr_t) * inv_det,
                )
            } else {
                self.last_guard_fired = true;
                (accum.jtr_r * inv_rr, 0.0)
            };

            let drho = delta_rho.clamp(-config.step_clamp, config.step_clamp);
            let dtheta = delta_theta.clamp(-THETA_CLAMP_RAD, THETA_CLAMP_RAD);

            rho += drho;
            if dtheta != 0.0 {
                let (sin_t, cos_t) = dtheta.sin_cos();
                let rotated = [
                    self.nx * cos_t - self.ny * sin_t,
                    self.nx * sin_t + self.ny * cos_t,
                ];
                self.nx = rotated[0];
                self.ny = rotated[1];
            }

            // Convergence in consistent pixel units: `drho` is already a
            // perpendicular offset in px; convert the rotation step to the
            // displacement it induces at the edge tip (½·len from `p_c`) so
            // both DOFs are tested against the same physical tolerance rather
            // than radians-vs-pixels against one shared threshold.
            if drho.abs() < config.convergence_threshold
                && dtheta.abs() * self.len * 0.5 < config.convergence_threshold
            {
                break;
            }
        }

        // The 2-DOF candidate (current `self`) vs the 1-DOF reference,
        // compared by a bounded-influence cost over the SAME full sample set
        // ([`robust_edge_cost`]). Using the same samples and the same robust
        // aggregate for both makes the decision fair (a rotation cannot win
        // merely by shedding high-residual samples out of an in-range band)
        // and degenerate-safe (the cost is over all ≥10 collected samples, so
        // it can never be the `0.0` an empty in-range set would produce).
        // Accept only when the rotation is meaningful (past the deadband —
        // which also keeps well-aligned edges byte-identical to 1-DOF) AND it
        // beats 1-DOF by a clear margin; otherwise the proven 1-DOF reference
        // stands, so the path is never worse than 1-DOF on robust edge fit.
        let theta_total = (seed_nx * self.ny - seed_ny * self.nx)
            .clamp(-1.0, 1.0)
            .asin();
        let cand_d = rho - self.nx * pcx - self.ny * pcy;
        let cost_2dof = robust_edge_cost(
            samples, self.nx, self.ny, cand_d, seed_a, seed_b, inv_sigma, clip,
        );
        self.last_sigma_hat = sigma_hat;

        if theta_total.abs() >= THETA_DEADBAND_RAD
            && cost_2dof < TWO_DOF_ACCEPT_COST_FRAC * cost_1dof
        {
            self.d = cand_d;
        } else {
            // Keep the proven 1-DOF reference; restore its telemetry too.
            self.nx = one_nx;
            self.ny = one_ny;
            self.d = one_d;
            self.last_jtj = one_jtj;
        }
    }

    /// End-to-end fit: optional initial-`d` scan, sample collection, and GN refinement.
    ///
    /// Returns `true` if samples were sufficient to run the GN loop. When `false`,
    /// `(nx, ny, d)` retain their values from `new(...)` (and, if applicable, the scan).
    pub fn fit(
        &mut self,
        arena: &'a Bump,
        sample_cfg: &SampleConfig,
        refine_cfg: &RefineConfig,
    ) -> bool {
        if refine_cfg.scan_initial {
            self.scan_initial_d();
        }
        let samples = self.collect_samples(arena, sample_cfg);
        if samples.len() < 10 {
            return false;
        }
        self.refine(&samples, refine_cfg);
        true
    }

    /// Get the result as `(nx, ny, d)`.
    #[inline]
    #[must_use]
    pub fn line_params(&self) -> (f64, f64, f64) {
        (self.nx, self.ny, self.d)
    }

    /// Get the edge length in pixels.
    #[inline]
    #[must_use]
    pub fn edge_len(&self) -> f64 {
        self.len
    }

    /// Final `J^T J` from the last [`Self::refine`] call. Zero if `refine` never
    /// ran a Gauss-Newton iteration (e.g., insufficient samples or low contrast).
    /// Used as a conditioning / convergence sentinel — a near-zero value
    /// indicates a degenerate fit (uniform sample band) regardless of whether
    /// `(nx, ny, d)` numerically converged.
    #[inline]
    #[must_use]
    pub fn line_jtj(&self) -> f64 {
        self.last_jtj
    }

    /// `true` if the 2-DOF conditioning guard tripped at least once during
    /// the last [`Self::refine`] call (θ too weakly observed, step fell back
    /// to a 1-DOF ρ update). Always `false` in 1-DOF mode. Test-only.
    #[cfg(test)]
    #[inline]
    pub(crate) fn last_guard_fired(&self) -> bool {
        self.last_guard_fired
    }

    /// IRLS scale estimate `σ̂` from the final 2-DOF iteration. `NaN` if
    /// 2-DOF was not run or `robust_loss == None`. Test-only.
    #[cfg(test)]
    #[inline]
    pub(crate) fn last_sigma_hat(&self) -> f64 {
        self.last_sigma_hat
    }

    #[inline]
    fn get_scan_bounds(&self, window: f64) -> (usize, usize, usize, usize) {
        let p2_0 = self.p1[0] + self.dx;
        let p2_1 = self.p1[1] + self.dy;

        let w_limit = (self.img.width - 2) as f64;
        let h_limit = (self.img.height - 2) as f64;

        let x0 = (self.p1[0].min(p2_0) - window - 0.5).clamp(1.0, w_limit) as usize;
        let x1 = (self.p1[0].max(p2_0) + window + 0.5).clamp(1.0, w_limit) as usize;
        let y0 = (self.p1[1].min(p2_1) - window - 0.5).clamp(1.0, h_limit) as usize;
        let y1 = (self.p1[1].max(p2_1) + window + 0.5).clamp(1.0, h_limit) as usize;
        (x0, x1, y0, y1)
    }
}

// ── A/B Intensity Estimation ─────────────────────────────────────────────────

/// One-shot A/B estimation from sample distances (quad-style).
///
/// Uses distance-weighted averaging: pixels far from the edge contribute more,
/// reducing contamination from the transition zone.
fn estimate_ab_oneshot(samples: &[(f64, f64, f64)], nx: f64, ny: f64, d: f64) -> (f64, f64) {
    let mut dark_sum = 0.0;
    let mut dark_weight = 0.0;
    let mut light_sum = 0.0;
    let mut light_weight = 0.0;

    for &(x, y, intensity) in samples {
        let signed_dist = nx * x + ny * y + d;
        if signed_dist < -1.0 {
            let w = (-signed_dist - 1.0).min(2.0);
            dark_sum += intensity * w;
            dark_weight += w;
        } else if signed_dist > 1.0 {
            let w = (signed_dist - 1.0).min(2.0);
            light_sum += intensity * w;
            light_weight += w;
        }
    }

    if dark_weight < 1.0 || light_weight < 1.0 {
        return (128.0, 128.0); // Insufficient samples
    }

    (dark_sum / dark_weight, light_sum / light_weight)
}

/// Per-iteration A/B estimation using bilinear sampling (decoder-style).
///
/// Re-reads intensities via bilinear interpolation each iteration to track
/// the line as `d` evolves. Returns `None` when either side lacks weight —
/// callers are expected to keep the previous iteration's estimate in that
/// case, matching the legacy decoder behavior.
fn estimate_ab_per_iter(
    img: &ImageView,
    samples: &[(f64, f64, f64)],
    nx: f64,
    ny: f64,
    d: f64,
) -> Option<(f64, f64)> {
    let mut dark_sum = 0.0;
    let mut dark_weight = 0.0;
    let mut light_sum = 0.0;
    let mut light_weight = 0.0;

    for &(x, y, _) in samples {
        let dist = nx * x + ny * y + d;
        let val = img.sample_bilinear(x, y);
        if dist < -1.0 {
            let w = (-dist - 0.5).clamp(0.1, 2.0);
            dark_sum += val * w;
            dark_weight += w;
        } else if dist > 1.0 {
            let w = (dist - 0.5).clamp(0.1, 2.0);
            light_sum += val * w;
            light_weight += w;
        }
    }

    if dark_weight <= 0.0 || light_weight <= 0.0 {
        return None;
    }

    Some((dark_sum / dark_weight, light_sum / light_weight))
}

// ── SIMD-Accelerated Gauss-Newton Accumulation ───────────────────────────────

/// Accumulate J^T J and J^T r for the Gauss-Newton step.
///
/// The model is `I(d) = (A+B)/2 + (B-A)/2 * erf(d/sigma)` with Jacobian
/// `dI/dd = (B-A) / (sqrt(pi) * sigma) * exp(-s^2)` where `s = d/sigma`.
///
/// On AVX2+FMA targets, uses vectorized erf_approx_v4 and FMA accumulation.
#[multiversion(targets(
    "x86_64+avx2+fma+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
#[allow(clippy::too_many_arguments)]
fn refine_accumulate_optimized(
    samples: &[(f64, f64, f64)],
    nx: f64,
    ny: f64,
    d: f64,
    a: f64,
    b: f64,
    inv_sigma: f64,
) -> (f64, f64) {
    let mut sum_jtj = 0.0;
    let mut sum_jt_res = 0.0;
    let k = (b - a) * inv_sigma * INV_SQRT_PI;

    let mut i = 0;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if let Some(_dispatch) = multiversion::target::x86_64::avx2::get() {
        // SAFETY: All AVX2/FMA intrinsics are guarded by cfg and runtime dispatch.
        unsafe {
            use std::arch::x86_64::*;
            let v_nx = _mm256_set1_pd(nx);
            let v_ny = _mm256_set1_pd(ny);
            let v_d = _mm256_set1_pd(d);
            let v_a = _mm256_set1_pd(a);
            let v_b = _mm256_set1_pd(b);
            let v_inv_sigma = _mm256_set1_pd(inv_sigma);
            let v_k = _mm256_set1_pd(k);
            let v_half = _mm256_set1_pd(0.5);
            let v_abs_mask = _mm256_set1_pd(-0.0);

            let mut v_sum_jtj = _mm256_setzero_pd();
            let mut v_sum_jt_res = _mm256_setzero_pd();

            while i + 4 <= samples.len() {
                let s0 = samples[i];
                let s1 = samples[i + 1];
                let s2 = samples[i + 2];
                let s3 = samples[i + 3];

                let v_x = _mm256_set_pd(s3.0, s2.0, s1.0, s0.0);
                let v_y = _mm256_set_pd(s3.1, s2.1, s1.1, s0.1);
                let v_img_val = _mm256_set_pd(s3.2, s2.2, s1.2, s0.2);

                let v_dist = _mm256_add_pd(
                    _mm256_add_pd(_mm256_mul_pd(v_nx, v_x), _mm256_mul_pd(v_ny, v_y)),
                    v_d,
                );
                let v_s = _mm256_mul_pd(v_dist, v_inv_sigma);

                let v_abs_s = _mm256_andnot_pd(v_abs_mask, v_s);
                let v_range_mask = _mm256_cmp_pd(v_abs_s, _mm256_set1_pd(3.0), _CMP_LE_OQ);

                if _mm256_movemask_pd(v_range_mask) != 0 {
                    // Vectorized ERF + Gauss-Newton accumulation.
                    // SAFETY: erf_approx_v4 requires AVX2+FMA, guaranteed by multiversion target.
                    let v_erf = crate::simd::math::erf_approx_v4(v_s);

                    // model = (a+b)*0.5 + (b-a)*0.5 * erf(s)
                    let v_ab_half = _mm256_mul_pd(_mm256_add_pd(v_a, v_b), v_half);
                    let v_ba_half = _mm256_mul_pd(_mm256_sub_pd(v_b, v_a), v_half);
                    let v_model = _mm256_fmadd_pd(v_ba_half, v_erf, v_ab_half);

                    // residual = img_val - model
                    let v_residual = _mm256_sub_pd(v_img_val, v_model);

                    // jac = k * exp(-s^2)
                    // SAFETY: transmute is safe for same-size SIMD ↔ array conversions.
                    let v_neg_s2 = _mm256_mul_pd(v_s, v_s);
                    let v_neg_s2 = _mm256_xor_pd(v_neg_s2, v_abs_mask);
                    let ns2: [f64; 4] = std::mem::transmute(v_neg_s2);
                    let v_exp =
                        _mm256_set_pd(ns2[3].exp(), ns2[2].exp(), ns2[1].exp(), ns2[0].exp());
                    let v_jac = _mm256_mul_pd(v_k, v_exp);

                    // Masked accumulation
                    let v_jac_masked = _mm256_and_pd(v_jac, v_range_mask);
                    let v_res_masked = _mm256_and_pd(v_residual, v_range_mask);

                    v_sum_jtj = _mm256_fmadd_pd(v_jac_masked, v_jac_masked, v_sum_jtj);
                    v_sum_jt_res = _mm256_fmadd_pd(v_jac_masked, v_res_masked, v_sum_jt_res);
                }
                i += 4;
            }

            // Horizontal reduction
            // SAFETY: transmute is safe for same-size SIMD ↔ array conversions.
            let jtj_lanes: [f64; 4] = std::mem::transmute(v_sum_jtj);
            let jtr_lanes: [f64; 4] = std::mem::transmute(v_sum_jt_res);
            sum_jtj += jtj_lanes[0] + jtj_lanes[1] + jtj_lanes[2] + jtj_lanes[3];
            sum_jt_res += jtr_lanes[0] + jtr_lanes[1] + jtr_lanes[2] + jtr_lanes[3];
        }
    }

    // Scalar tail
    while i < samples.len() {
        let (x, y, img_val) = samples[i];
        let dist = nx * x + ny * y + d;
        let s = dist * inv_sigma;
        if s.abs() <= 3.0 {
            let model = (a + b) * 0.5 + (b - a) * 0.5 * crate::simd::math::erf_approx(s);
            let residual = img_val - model;
            let jac = k * (-s * s).exp();
            sum_jtj += jac * jac;
            sum_jt_res += jac * residual;
        }
        i += 1;
    }

    (sum_jtj, sum_jt_res)
}

/// 2-DOF accumulator: 2×2 symmetric `JᵀJ` plus 2-vector `Jᵀr`.
/// `_rr` / `_rt` / `_tt` are the ρρ, ρθ, θθ entries of `JᵀJ`; `r` / `t`
/// are the ρ and θ entries of `Jᵀr`.
#[derive(Clone, Copy, Debug, Default)]
struct GnAccum2 {
    jtj_rr: f64,
    jtj_rt: f64,
    jtj_tt: f64,
    jtr_r: f64,
    jtr_t: f64,
}

/// Tukey-weighted 2×2 `JᵀJ` + 2-vector `Jᵀr` accumulator for the centered
/// `(θ, ρ)` GN step. The single 2-DOF accumulator: `inv_c_sigma = 1 / (c · σ̂)`
/// controls the Tukey rejection cutoff, with per-sample weight
/// `w_i = (1 − (r_i · inv_c_sigma)²)²` when in range else `0`. Passing
/// `inv_c_sigma = 0` makes every weight `(1 − 0)² = 1`, i.e. an unweighted
/// 2-DOF solve — so this one kernel serves both the production Tukey IRLS path
/// and the (test-only) unweighted baseline byte-identically, with no second
/// hand-written SIMD kernel to keep in sync.
///
/// Jacobian columns per sample: `∂r/∂ρ = jac`, `∂r/∂θ = jac · t` with the
/// tangent `t = -ny·xc + nx·yc` and `(xc, yc) = (x − p_c)`. Centering on `p_c`
/// keeps `mean(t) ≈ 0` so the ρθ cross-term stays small. AVX2/FMA + AVX-512 +
/// `aarch64+neon` via `#[multiversion]`, scalar tail for the remainder. Kept
/// separate from the 1-DOF [`refine_accumulate_optimized`] so 1-DOF callers
/// don't pay the 2-DOF arithmetic.
#[multiversion(targets(
    "x86_64+avx2+fma+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
#[allow(clippy::too_many_arguments, clippy::similar_names)]
fn refine_accumulate_optimized_2dof_tukey(
    samples: &[(f64, f64, f64)],
    nx: f64,
    ny: f64,
    rho: f64,
    pcx: f64,
    pcy: f64,
    a: f64,
    b: f64,
    inv_sigma: f64,
    inv_c_sigma: f64,
) -> GnAccum2 {
    let mut acc = GnAccum2::default();
    let k = (b - a) * inv_sigma * INV_SQRT_PI;
    let mut i = 0;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if let Some(_dispatch) = multiversion::target::x86_64::avx2::get() {
        // SAFETY: AVX2/FMA intrinsics guarded by cfg and runtime dispatch.
        unsafe {
            use std::arch::x86_64::*;
            let v_nx = _mm256_set1_pd(nx);
            let v_ny = _mm256_set1_pd(ny);
            let v_rho = _mm256_set1_pd(rho);
            let v_pcx = _mm256_set1_pd(pcx);
            let v_pcy = _mm256_set1_pd(pcy);
            let v_a = _mm256_set1_pd(a);
            let v_b = _mm256_set1_pd(b);
            let v_inv_sigma = _mm256_set1_pd(inv_sigma);
            let v_k = _mm256_set1_pd(k);
            let v_half = _mm256_set1_pd(0.5);
            let v_abs_mask = _mm256_set1_pd(-0.0);
            let v_one = _mm256_set1_pd(1.0);
            let v_inv_c_sigma = _mm256_set1_pd(inv_c_sigma);

            let mut v_sum_jtj_rr = _mm256_setzero_pd();
            let mut v_sum_jtj_rt = _mm256_setzero_pd();
            let mut v_sum_jtj_tt = _mm256_setzero_pd();
            let mut v_sum_jtr_r = _mm256_setzero_pd();
            let mut v_sum_jtr_t = _mm256_setzero_pd();

            while i + 4 <= samples.len() {
                let s0 = samples[i];
                let s1 = samples[i + 1];
                let s2 = samples[i + 2];
                let s3 = samples[i + 3];

                let v_x = _mm256_set_pd(s3.0, s2.0, s1.0, s0.0);
                let v_y = _mm256_set_pd(s3.1, s2.1, s1.1, s0.1);
                let v_img_val = _mm256_set_pd(s3.2, s2.2, s1.2, s0.2);

                let v_xc = _mm256_sub_pd(v_x, v_pcx);
                let v_yc = _mm256_sub_pd(v_y, v_pcy);

                let v_dist = _mm256_add_pd(
                    _mm256_add_pd(_mm256_mul_pd(v_nx, v_xc), _mm256_mul_pd(v_ny, v_yc)),
                    v_rho,
                );
                let v_s = _mm256_mul_pd(v_dist, v_inv_sigma);

                let v_abs_s = _mm256_andnot_pd(v_abs_mask, v_s);
                let v_range_mask = _mm256_cmp_pd(v_abs_s, _mm256_set1_pd(3.0), _CMP_LE_OQ);

                if _mm256_movemask_pd(v_range_mask) != 0 {
                    // SAFETY: erf_approx_v4 requires AVX2+FMA, guaranteed by multiversion target.
                    let v_erf = crate::simd::math::erf_approx_v4(v_s);

                    let v_ab_half = _mm256_mul_pd(_mm256_add_pd(v_a, v_b), v_half);
                    let v_ba_half = _mm256_mul_pd(_mm256_sub_pd(v_b, v_a), v_half);
                    let v_model = _mm256_fmadd_pd(v_ba_half, v_erf, v_ab_half);
                    let v_residual = _mm256_sub_pd(v_img_val, v_model);

                    // SAFETY: transmute is safe for same-size SIMD ↔ array conversions.
                    let v_neg_s2 = _mm256_mul_pd(v_s, v_s);
                    let v_neg_s2 = _mm256_xor_pd(v_neg_s2, v_abs_mask);
                    let ns2: [f64; 4] = std::mem::transmute(v_neg_s2);
                    let v_exp =
                        _mm256_set_pd(ns2[3].exp(), ns2[2].exp(), ns2[1].exp(), ns2[0].exp());
                    let v_jac = _mm256_mul_pd(v_k, v_exp);

                    let v_t = _mm256_sub_pd(_mm256_mul_pd(v_nx, v_yc), _mm256_mul_pd(v_ny, v_xc));
                    let v_jac_m = _mm256_and_pd(v_jac, v_range_mask);
                    let v_res_m = _mm256_and_pd(v_residual, v_range_mask);

                    // Tukey: w = (1 − u²)² when u² < 1, else 0; u = r · inv_c_sigma.
                    let v_u = _mm256_mul_pd(v_res_m, v_inv_c_sigma);
                    let v_u2 = _mm256_mul_pd(v_u, v_u);
                    let v_in = _mm256_cmp_pd(v_u2, v_one, _CMP_LT_OQ);
                    let v_one_minus_u2 = _mm256_sub_pd(v_one, v_u2);
                    let v_w_sq = _mm256_mul_pd(v_one_minus_u2, v_one_minus_u2);
                    let v_w = _mm256_and_pd(_mm256_and_pd(v_w_sq, v_in), v_range_mask);

                    let v_jac_w = _mm256_mul_pd(v_jac_m, v_w);
                    let v_jact = _mm256_mul_pd(v_jac_m, v_t);
                    let v_jact_w = _mm256_mul_pd(v_jac_w, v_t);

                    v_sum_jtj_rr = _mm256_fmadd_pd(v_jac_w, v_jac_m, v_sum_jtj_rr);
                    v_sum_jtj_rt = _mm256_fmadd_pd(v_jac_w, v_jact, v_sum_jtj_rt);
                    v_sum_jtj_tt = _mm256_fmadd_pd(v_jact_w, v_jact, v_sum_jtj_tt);
                    v_sum_jtr_r = _mm256_fmadd_pd(v_jac_w, v_res_m, v_sum_jtr_r);
                    v_sum_jtr_t = _mm256_fmadd_pd(v_jact_w, v_res_m, v_sum_jtr_t);
                }
                i += 4;
            }

            // SAFETY: transmute is safe for same-size SIMD ↔ array conversions.
            let rr: [f64; 4] = std::mem::transmute(v_sum_jtj_rr);
            let rt: [f64; 4] = std::mem::transmute(v_sum_jtj_rt);
            let tt: [f64; 4] = std::mem::transmute(v_sum_jtj_tt);
            let gr: [f64; 4] = std::mem::transmute(v_sum_jtr_r);
            let gt: [f64; 4] = std::mem::transmute(v_sum_jtr_t);
            acc.jtj_rr += rr[0] + rr[1] + rr[2] + rr[3];
            acc.jtj_rt += rt[0] + rt[1] + rt[2] + rt[3];
            acc.jtj_tt += tt[0] + tt[1] + tt[2] + tt[3];
            acc.jtr_r += gr[0] + gr[1] + gr[2] + gr[3];
            acc.jtr_t += gt[0] + gt[1] + gt[2] + gt[3];
        }
    }

    while i < samples.len() {
        let (x, y, img_val) = samples[i];
        let xc = x - pcx;
        let yc = y - pcy;
        let dist = nx * xc + ny * yc + rho;
        let s = dist * inv_sigma;
        if s.abs() <= 3.0 {
            let model = (a + b) * 0.5 + (b - a) * 0.5 * crate::simd::math::erf_approx(s);
            let residual = img_val - model;
            let jac = k * (-s * s).exp();
            let t = -ny * xc + nx * yc;
            let jact = jac * t;
            let u = residual * inv_c_sigma;
            let u2 = u * u;
            let w = if u2 < 1.0 { (1.0 - u2).powi(2) } else { 0.0 };
            acc.jtj_rr += w * jac * jac;
            acc.jtj_rt += w * jac * jact;
            acc.jtj_tt += w * jact * jact;
            acc.jtr_r += w * jac * residual;
            acc.jtr_t += w * jact * residual;
        }
        i += 1;
    }

    acc
}

/// Bounded-influence edge-fit cost of a line `nx·x + ny·y + d = 0` over ALL
/// collected samples — the fair, degenerate-safe basis for the 1-DOF-vs-2-DOF
/// acceptance decision. Returns the mean of `min(r², clip²)` over the samples,
/// where `r` is the ERF-model residual. Clipping each squared residual caps
/// the leverage of a minority of off-edge / adjacent-structure samples, so a
/// correct rotation (which moves *away* from contamination) is not unfairly
/// penalised, and neither candidate can win merely by shedding samples out of
/// an in-range band (both are scored over the same full set). Returns `+∞`
/// for an empty set so a degenerate candidate is never preferred. Called O(1)
/// times per edge (not per GN iteration), so a scalar pass is adequate.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn robust_edge_cost(
    samples: &[(f64, f64, f64)],
    nx: f64,
    ny: f64,
    d: f64,
    a: f64,
    b: f64,
    inv_sigma: f64,
    clip: f64,
) -> f64 {
    if samples.is_empty() {
        return f64::INFINITY;
    }
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let clip_sq = clip * clip;
    let mut sum = 0.0_f64;
    for &(x, y, img_val) in samples {
        let s = (nx * x + ny * y + d) * inv_sigma;
        let model = mid + half * crate::simd::math::erf_approx(s);
        let r = img_val - model;
        sum += (r * r).min(clip_sq);
    }
    sum / samples.len() as f64
}

/// Robust scale `σ̂ = sqrt(Σ r² / N)` at the current `(nx, ny, rho)` line
/// params. Used for the pre-pass that initialises Tukey weights before the
/// weighted GN accumulator runs. Returns 0.0 when no in-range samples exist.
#[multiversion(targets(
    "x86_64+avx2+fma+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
#[allow(clippy::too_many_arguments)]
fn sigma_at(
    samples: &[(f64, f64, f64)],
    nx: f64,
    ny: f64,
    rho: f64,
    pcx: f64,
    pcy: f64,
    a: f64,
    b: f64,
    inv_sigma: f64,
) -> f64 {
    let mut sum_r2 = 0.0_f64;
    let mut n: u32 = 0;
    let mut i = 0;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if let Some(_dispatch) = multiversion::target::x86_64::avx2::get() {
        // SAFETY: AVX2/FMA intrinsics guarded by cfg and runtime dispatch.
        unsafe {
            use std::arch::x86_64::*;
            let v_nx = _mm256_set1_pd(nx);
            let v_ny = _mm256_set1_pd(ny);
            let v_rho = _mm256_set1_pd(rho);
            let v_pcx = _mm256_set1_pd(pcx);
            let v_pcy = _mm256_set1_pd(pcy);
            let v_a = _mm256_set1_pd(a);
            let v_b = _mm256_set1_pd(b);
            let v_inv_sigma = _mm256_set1_pd(inv_sigma);
            let v_half = _mm256_set1_pd(0.5);
            let v_abs_mask = _mm256_set1_pd(-0.0);
            let mut v_sum_r2 = _mm256_setzero_pd();

            while i + 4 <= samples.len() {
                let s0 = samples[i];
                let s1 = samples[i + 1];
                let s2 = samples[i + 2];
                let s3 = samples[i + 3];
                let v_x = _mm256_set_pd(s3.0, s2.0, s1.0, s0.0);
                let v_y = _mm256_set_pd(s3.1, s2.1, s1.1, s0.1);
                let v_img_val = _mm256_set_pd(s3.2, s2.2, s1.2, s0.2);
                let v_xc = _mm256_sub_pd(v_x, v_pcx);
                let v_yc = _mm256_sub_pd(v_y, v_pcy);
                let v_dist = _mm256_add_pd(
                    _mm256_add_pd(_mm256_mul_pd(v_nx, v_xc), _mm256_mul_pd(v_ny, v_yc)),
                    v_rho,
                );
                let v_s = _mm256_mul_pd(v_dist, v_inv_sigma);
                let v_abs_s = _mm256_andnot_pd(v_abs_mask, v_s);
                let v_range_mask = _mm256_cmp_pd(v_abs_s, _mm256_set1_pd(3.0), _CMP_LE_OQ);
                let mask_bits = _mm256_movemask_pd(v_range_mask);
                if mask_bits != 0 {
                    // SAFETY: erf_approx_v4 requires AVX2+FMA.
                    let v_erf = crate::simd::math::erf_approx_v4(v_s);
                    let v_ab_half = _mm256_mul_pd(_mm256_add_pd(v_a, v_b), v_half);
                    let v_ba_half = _mm256_mul_pd(_mm256_sub_pd(v_b, v_a), v_half);
                    let v_model = _mm256_fmadd_pd(v_ba_half, v_erf, v_ab_half);
                    let v_residual = _mm256_sub_pd(v_img_val, v_model);
                    let v_res_m = _mm256_and_pd(v_residual, v_range_mask);
                    v_sum_r2 = _mm256_fmadd_pd(v_res_m, v_res_m, v_sum_r2);
                    #[allow(clippy::cast_sign_loss)]
                    {
                        n += (mask_bits as u32).count_ones();
                    }
                }
                i += 4;
            }
            // SAFETY: transmute is safe for same-size SIMD ↔ array conversions.
            let r2: [f64; 4] = std::mem::transmute(v_sum_r2);
            sum_r2 += r2[0] + r2[1] + r2[2] + r2[3];
        }
    }

    while i < samples.len() {
        let (x, y, img_val) = samples[i];
        let xc = x - pcx;
        let yc = y - pcy;
        let dist = nx * xc + ny * yc + rho;
        let s = dist * inv_sigma;
        if s.abs() <= 3.0 {
            let model = (a + b) * 0.5 + (b - a) * 0.5 * crate::simd::math::erf_approx(s);
            let residual = img_val - model;
            sum_r2 += residual * residual;
            n += 1;
        }
        i += 1;
    }

    if n == 0 {
        0.0
    } else {
        (sum_r2 / f64::from(n)).sqrt()
    }
}

// ── SIMD-Accelerated Sample Collection ───────────────────────────────────────

/// Gradient projection for d-scan initialization.
///
/// Computes the sum of `|gradient . normal|` for pixels near the line,
/// used to find the offset that maximizes edge evidence.
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
#[allow(clippy::too_many_arguments)]
fn project_gradients_optimized(
    img: &ImageView,
    nx: f64,
    ny: f64,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    scan_d: f64,
) -> (f64, usize) {
    let mut sum_g = 0.0;
    let mut count = 0;

    for py in y0..=y1 {
        let mut px = x0;
        let y = py as f64 + 0.5;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if let Some(_dispatch) = multiversion::target::x86_64::avx2::get() {
            // SAFETY: AVX2 intrinsics guarded by cfg and runtime dispatch.
            unsafe {
                use std::arch::x86_64::*;
                let v_nx = _mm256_set1_pd(nx);
                let v_ny = _mm256_set1_pd(ny);
                let v_scan_d = _mm256_set1_pd(scan_d);
                let v_y = _mm256_set1_pd(y);
                let v_abs_mask = _mm256_set1_pd(-0.0);

                while px + 4 <= x1 {
                    let v_x = _mm256_set_pd(
                        (px + 3) as f64 + 0.5,
                        (px + 2) as f64 + 0.5,
                        (px + 1) as f64 + 0.5,
                        px as f64 + 0.5,
                    );

                    let v_dist = _mm256_add_pd(
                        _mm256_add_pd(_mm256_mul_pd(v_nx, v_x), _mm256_mul_pd(v_ny, v_y)),
                        v_scan_d,
                    );

                    let v_abs_dist = _mm256_andnot_pd(v_abs_mask, v_dist);
                    let v_cmp = _mm256_cmp_pd(v_abs_dist, _mm256_set1_pd(1.0), _CMP_LT_OQ);
                    let mask = _mm256_movemask_pd(v_cmp);

                    if mask != 0 {
                        for j in 0..4 {
                            if (mask >> j) & 1 != 0 {
                                let g = img.sample_gradient_bilinear((px + j) as f64, y);
                                sum_g += (g[0] * nx + g[1] * ny).abs();
                                count += 1;
                            }
                        }
                    }
                    px += 4;
                }
            }
        }

        while px <= x1 {
            let x = px as f64 + 0.5;
            let dist = nx * x + ny * y + scan_d;
            if dist.abs() < 1.0 {
                let g = img.sample_gradient_bilinear(x, y);
                sum_g += (g[0] * nx + g[1] * ny).abs();
                count += 1;
            }
            px += 1;
        }
    }
    (sum_g, count)
}

/// SIMD-accelerated sample collection (stride = 1).
#[multiversion(targets(
    "x86_64+avx2+bmi1+bmi2+popcnt+lzcnt",
    "x86_64+avx512f+avx512bw+avx512dq+avx512vl",
    "aarch64+neon"
))]
#[allow(clippy::too_many_arguments)]
fn collect_samples_optimized<'a>(
    img: &ImageView,
    nx: f64,
    ny: f64,
    d: f64,
    p1: [f64; 2],
    dx: f64,
    dy: f64,
    len: f64,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    window: f64,
    t_range: (f64, f64),
    arena: &'a Bump,
) -> BumpVec<'a, (f64, f64, f64)> {
    let mut samples = BumpVec::with_capacity_in(128, arena);
    let inv_len_sq = 1.0 / (len * len);

    for py in y0..=y1 {
        let mut px = x0;
        let y = py as f64 + 0.5;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if let Some(_dispatch) = multiversion::target::x86_64::avx2::get() {
            // SAFETY: AVX2 intrinsics guarded by cfg and runtime dispatch.
            unsafe {
                use std::arch::x86_64::*;
                let v_nx = _mm256_set1_pd(nx);
                let v_ny = _mm256_set1_pd(ny);
                let v_d = _mm256_set1_pd(d);
                let v_y = _mm256_set1_pd(y);
                let v_p1x = _mm256_set1_pd(p1[0]);
                let v_p1y = _mm256_set1_pd(p1[1]);
                let v_dx = _mm256_set1_pd(dx);
                let v_dy = _mm256_set1_pd(dy);
                let v_inv_len_sq = _mm256_set1_pd(inv_len_sq);
                let v_abs_mask = _mm256_set1_pd(-0.0);

                while px + 4 <= x1 {
                    let v_x = _mm256_set_pd(
                        (px + 3) as f64 + 0.5,
                        (px + 2) as f64 + 0.5,
                        (px + 1) as f64 + 0.5,
                        px as f64 + 0.5,
                    );

                    let v_dist = _mm256_add_pd(
                        _mm256_add_pd(_mm256_mul_pd(v_nx, v_x), _mm256_mul_pd(v_ny, v_y)),
                        v_d,
                    );
                    let v_abs_dist = _mm256_andnot_pd(v_abs_mask, v_dist);
                    let v_dist_mask = _mm256_cmp_pd(v_abs_dist, _mm256_set1_pd(window), _CMP_LE_OQ);

                    let v_t = _mm256_mul_pd(
                        _mm256_add_pd(
                            _mm256_mul_pd(_mm256_sub_pd(v_x, v_p1x), v_dx),
                            _mm256_mul_pd(_mm256_sub_pd(v_y, v_p1y), v_dy),
                        ),
                        v_inv_len_sq,
                    );

                    let v_t_mask_low = _mm256_cmp_pd(v_t, _mm256_set1_pd(t_range.0), _CMP_GE_OQ);
                    let v_t_mask_high = _mm256_cmp_pd(v_t, _mm256_set1_pd(t_range.1), _CMP_LE_OQ);

                    let final_mask = _mm256_movemask_pd(_mm256_and_pd(
                        v_dist_mask,
                        _mm256_and_pd(v_t_mask_low, v_t_mask_high),
                    ));

                    if final_mask != 0 {
                        for j in 0..4 {
                            if (final_mask >> j) & 1 != 0 {
                                let val = f64::from(img.get_pixel(px + j, py));
                                samples.push(((px + j) as f64 + 0.5, y, val));
                            }
                        }
                    }
                    px += 4;
                }
            }
        }

        while px <= x1 {
            let x = px as f64 + 0.5;
            let dist = nx * x + ny * y + d;
            if dist.abs() <= window {
                let t = ((x - p1[0]) * dx + (y - p1[1]) * dy) * inv_len_sq;
                if t >= t_range.0 && t <= t_range.1 {
                    let val = f64::from(img.get_pixel(px, py));
                    samples.push((x, y, val));
                }
            }
            px += 1;
        }
    }
    samples
}

/// Strided sample collection for decimated images.
///
/// Identical logic to `collect_samples_optimized` but with pixel stride > 1.
/// Used by the quad extraction stage for large edges in upscaled images.
#[allow(clippy::too_many_arguments)]
fn collect_samples_strided<'a>(
    img: &ImageView,
    nx: f64,
    ny: f64,
    d: f64,
    p1: [f64; 2],
    dx: f64,
    dy: f64,
    len: f64,
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    window: f64,
    t_range: (f64, f64),
    stride: usize,
    arena: &'a Bump,
) -> BumpVec<'a, (f64, f64, f64)> {
    let mut samples = BumpVec::with_capacity_in(128, arena);
    let inv_len_sq = 1.0 / (len * len);

    let mut py = y0;
    while py <= y1 {
        let mut px = x0;
        let y = py as f64 + 0.5;
        while px <= x1 {
            let x = px as f64 + 0.5;
            let dist = nx * x + ny * y + d;
            if dist.abs() <= window {
                let t = ((x - p1[0]) * dx + (y - p1[1]) * dy) * inv_len_sq;
                if t >= t_range.0 && t <= t_range.1 {
                    let val = f64::from(img.get_pixel(px, py));
                    samples.push((x, y, val));
                }
            }
            px += stride;
        }
        py += stride;
    }
    samples
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_utils::subpixel::{Line, SubpixelEdgeRenderer};

    #[test]
    fn quad_style_recovers_axis_aligned_subpixel_edge() {
        let width = 100;
        let height = 100;
        let sigma = 0.6;
        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(0.0, 255.0)
            .with_sigma(sigma);

        for x_gt in [50.0, 50.25, 50.5, 50.75] {
            let line_gt = Line::from_points_cw([x_gt, 10.0], [x_gt, 90.0]);
            let data = renderer.render_edge_u8(&line_gt);
            let img = ImageView::new(&data, width, height, width).expect("invalid image view");
            let arena = Bump::new();

            let mut fitter = ErfEdgeFitter::new(&img, [50.0, 10.0], [50.0, 90.0], true)
                .expect("edge length too short");
            let sample_cfg = SampleConfig::for_quad(fitter.edge_len(), 1);
            let refine_cfg = RefineConfig::quad_style(sigma);
            assert!(fitter.fit(&arena, &sample_cfg, &refine_cfg));

            // LHN: nx = -dy/len = 0, ny = dx/len = 0 for vertical edge p1→p2.
            // With p1=(50,10)→p2=(50,90): dy=80, dx=0 ⇒ nx=-1, ny=0.
            let (nx, _ny, d) = fitter.line_params();
            assert!((nx + 1.0).abs() < 1e-7, "nx = {nx}");
            let x_recovered = d; // -x + d = 0
            let error = (x_recovered - x_gt).abs();
            assert!(
                error < 0.02,
                "x_gt={x_gt} recovered={x_recovered} error={error}"
            );
        }
    }

    #[test]
    fn quad_style_recovers_angled_edge() {
        let width = 120;
        let height = 120;
        let sigma = 0.6;
        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(20.0, 230.0)
            .with_sigma(sigma);

        // 30-degree edge from (40, 20) to (80, 20 + 40*tan30)
        let angle = 30f64.to_radians();
        let p1 = [40.0, 20.0];
        let p2 = [80.0, 20.0 + 40.0 * angle.tan()];
        let line_gt = Line::from_points_cw(p1, p2);
        let data = renderer.render_edge_u8(&line_gt);
        let img = ImageView::new(&data, width, height, width).expect("invalid image view");
        let arena = Bump::new();

        let mut fitter = ErfEdgeFitter::new(&img, p1, p2, true).expect("edge length too short");
        let sample_cfg = SampleConfig::for_quad(fitter.edge_len(), 1);
        let refine_cfg = RefineConfig::quad_style(sigma);
        assert!(fitter.fit(&arena, &sample_cfg, &refine_cfg));

        // The ground-truth line is `line_gt.a*x + line_gt.b*y + line_gt.c = 0` (RHN),
        // while the fitter uses LHN — the triple is negated.
        let (nx, ny, d) = fitter.line_params();
        let err_a = (nx + line_gt.a).abs().min((nx - line_gt.a).abs());
        let err_b = (ny + line_gt.b).abs().min((ny - line_gt.b).abs());
        let err_c = (d + line_gt.c).abs().min((d - line_gt.c).abs());
        assert!(
            err_a < 1e-2 && err_b < 1e-2 && err_c < 5e-2,
            "angled-edge line-param error: a={err_a} b={err_b} c={err_c}"
        );
    }

    #[test]
    fn decoder_style_scan_initial_d_recovers_from_perturbed_seed() {
        let width = 120;
        let height = 120;
        let sigma = 0.6;
        let x_gt = 60.25;

        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(30.0, 220.0)
            .with_sigma(sigma);
        let line_gt = Line::from_points_cw([x_gt, 10.0], [x_gt, 110.0]);
        let data = renderer.render_edge_u8(&line_gt);
        let img = ImageView::new(&data, width, height, width).expect("invalid image view");

        // Seed the fitter 1.5 px off the true edge — outside the GN capture radius
        // but within scan_initial_d's ±2.4 px search window.
        let p1 = [61.75, 10.0];
        let p2 = [61.75, 110.0];
        let arena = Bump::new();
        let mut fitter = ErfEdgeFitter::new(&img, p1, p2, false).expect("edge length too short");
        let sample_cfg = SampleConfig::for_decoder();
        let refine_cfg = RefineConfig::decoder_style(sigma);
        assert!(fitter.fit(&arena, &sample_cfg, &refine_cfg));

        let (nx, _ny, d) = fitter.line_params();
        assert!((nx + 1.0).abs() < 1e-7);
        let x_recovered = d;
        let error = (x_recovered - x_gt).abs();
        assert!(
            error < 0.05,
            "perturbed seed recovered={x_recovered} error={error}"
        );
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn two_dof_recovers_mis_angled_seed() {
        // 30° true edge, ~92 px long; seed endpoints rotated +1.5° about their
        // midpoint, so the seed normal is exactly 1.5° off truth. 1-DOF cannot
        // rotate the line — its (nx, ny) is frozen at seed — and must end at
        // ~1.5° error. 2-DOF lifts (θ, ρ) and recovers true direction.
        let width = 160;
        let height = 160;
        let sigma = 0.6;
        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(20.0, 230.0)
            .with_sigma(sigma);

        let p1_gt = [40.0, 60.0];
        let p2_gt = [120.0, 60.0 + 80.0 * 30f64.to_radians().tan()];
        let line_gt = Line::from_points_cw(p1_gt, p2_gt);
        let data = renderer.render_edge_u8(&line_gt);
        let img = ImageView::new(&data, width, height, width).expect("invalid image view");

        let perturb_rad = 1.5_f64.to_radians();
        let (sp, cp) = perturb_rad.sin_cos();
        let mid = [(p1_gt[0] + p2_gt[0]) * 0.5, (p1_gt[1] + p2_gt[1]) * 0.5];
        let rotate = |p: [f64; 2]| -> [f64; 2] {
            let dx = p[0] - mid[0];
            let dy = p[1] - mid[1];
            [mid[0] + dx * cp - dy * sp, mid[1] + dx * sp + dy * cp]
        };
        let p1 = rotate(p1_gt);
        let p2 = rotate(p2_gt);

        let true_dx = p2_gt[0] - p1_gt[0];
        let true_dy = p2_gt[1] - p1_gt[1];
        let true_len = (true_dx * true_dx + true_dy * true_dy).sqrt();
        let nx_true = -true_dy / true_len;
        let ny_true = true_dx / true_len;

        let angle_err = |nx: f64, ny: f64| -> f64 {
            (nx * nx_true + ny * ny_true)
                .abs()
                .clamp(0.0, 1.0)
                .acos()
                .to_degrees()
        };

        let sample_cfg = SampleConfig::for_quad(true_len, 1);

        // 1-DOF — must remain near the seed perturbation. Force OneDof
        // explicitly so the test is independent of `quad_style`'s current
        // production default.
        let arena_1 = Bump::new();
        let mut fitter_1 = ErfEdgeFitter::new(&img, p1, p2, true).expect("edge length too short");
        let cfg_1 = RefineConfig {
            mode: RefineMode::OneDof,
            ..RefineConfig::quad_style(sigma)
        };
        assert!(fitter_1.fit(&arena_1, &sample_cfg, &cfg_1));
        let (nx_1, ny_1, _) = fitter_1.line_params();
        let err_1 = angle_err(nx_1, ny_1);

        // 2-DOF (no robust loss) — must recover true rotation on this
        // clean fixture (no outliers — IRLS is not needed).
        let arena_2 = Bump::new();
        let mut fitter_2 = ErfEdgeFitter::new(&img, p1, p2, true).expect("edge length too short");
        let cfg_2 = RefineConfig {
            mode: RefineMode::TwoDof,
            ..RefineConfig::quad_style(sigma)
        };
        assert!(fitter_2.fit(&arena_2, &sample_cfg, &cfg_2));
        let (nx_2, ny_2, _) = fitter_2.line_params();
        let err_2 = angle_err(nx_2, ny_2);

        assert!(err_2 < 0.5, "2-DOF angle err = {err_2}°, expected < 0.5°");
        assert!(
            err_1 > 1.0,
            "1-DOF angle err = {err_1}°, expected > 1.0° (1-DOF cannot rotate)"
        );
        assert!(
            !fitter_2.last_guard_fired(),
            "guard fired on long well-conditioned edge"
        );
    }

    #[test]
    fn two_dof_concentrated_samples_no_spurious_rotation() {
        // Concentrated tangential band (~2% of the edge) makes θ unobservable.
        // On a clean edge the 1-DOF fit is already good, so the 2-DOF solve is
        // not even attempted (the poor-fit trigger is the primary protection
        // against rotating on noise); the in-loop conditioning guard remains as
        // defence-in-depth for the singular-`det` / low-`var_t` case should the
        // solve run. Either way the invariant holds: a concentrated sample band
        // must not induce a spurious rotation — the orientation stays at seed
        // and ρ is recovered.
        let width = 120;
        let height = 120;
        let sigma = 0.6;
        let x_gt = 60.25;

        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(20.0, 230.0)
            .with_sigma(sigma);
        let line_gt = Line::from_points_cw([x_gt, 20.0], [x_gt, 100.0]);
        let data = renderer.render_edge_u8(&line_gt);
        let img = ImageView::new(&data, width, height, width).expect("invalid image view");

        let p1 = [x_gt, 20.0];
        let p2 = [x_gt, 100.0];
        let arena = Bump::new();
        let mut fitter = ErfEdgeFitter::new(&img, p1, p2, true).expect("edge length too short");

        let sample_cfg = SampleConfig {
            window: 2.5,
            stride: 1,
            t_range: (0.49, 0.51),
        };
        // Production path (TwoDofTukey).
        assert!(fitter.fit(&arena, &sample_cfg, &RefineConfig::quad_style(sigma)));

        let (nx, ny, d) = fitter.line_params();
        assert!(
            (nx + 1.0).abs() < 1e-6 && ny.abs() < 1e-6,
            "normal drifted from seed: ({nx}, {ny})"
        );
        let x_recovered = d;
        assert!(
            (x_recovered - x_gt).abs() < 0.15,
            "x_recovered = {x_recovered}, expected {x_gt}"
        );
    }

    /// Adjacent-edge outlier injection: flips the intensity of every
    /// third light-side sample close to the edge transition (high
    /// Jacobian, large residual). Mirrors the production failure mode
    /// where bit-boundary samples pulled the 2-DOF normal off-truth.
    fn inject_adjacent_edge_outliers(
        clean: &[(f64, f64, f64)],
        nx: f64,
        ny: f64,
        d: f64,
        dark_intensity: f64,
    ) -> Vec<(f64, f64, f64)> {
        let mut out: Vec<(f64, f64, f64)> = clean.to_vec();
        let mut count = 0_usize;
        for sample in &mut out {
            let signed_dist = nx * sample.0 + ny * sample.1 + d;
            if signed_dist > 0.2 && signed_dist < 1.8 {
                if count.is_multiple_of(3) {
                    sample.2 = dark_intensity;
                }
                count += 1;
            }
        }
        out
    }

    /// Tukey IRLS recovers the true edge direction under adjacent-edge
    /// contamination — the failure mode that blocked plain 2-DOF.
    #[test]
    #[allow(clippy::similar_names)]
    fn tukey_recovers_under_adjacent_edge_outliers() {
        let width = 160;
        let height = 160;
        let sigma = 0.6;
        let dark = 20.0_f64;
        let light = 230.0_f64;
        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(dark, light)
            .with_sigma(sigma);

        let p1_gt = [40.0, 60.0];
        let p2_gt = [120.0, 60.0 + 80.0 * 30f64.to_radians().tan()];
        let line_gt = Line::from_points_cw(p1_gt, p2_gt);
        let data = renderer.render_edge_u8(&line_gt);
        let img = ImageView::new(&data, width, height, width).expect("invalid image view");

        // Seed: GT endpoints rotated +1.5° about their midpoint.
        let perturb = 1.5_f64.to_radians();
        let (sp, cp) = perturb.sin_cos();
        let mid = [(p1_gt[0] + p2_gt[0]) * 0.5, (p1_gt[1] + p2_gt[1]) * 0.5];
        let rotate = |p: [f64; 2]| -> [f64; 2] {
            let dx = p[0] - mid[0];
            let dy = p[1] - mid[1];
            [mid[0] + dx * cp - dy * sp, mid[1] + dx * sp + dy * cp]
        };
        let p1 = rotate(p1_gt);
        let p2 = rotate(p2_gt);

        let true_len = ((p2_gt[0] - p1_gt[0]).hypot(p2_gt[1] - p1_gt[1])).max(1e-12);
        let nx_true = -(p2_gt[1] - p1_gt[1]) / true_len;
        let ny_true = (p2_gt[0] - p1_gt[0]) / true_len;
        let angle_err = |nx: f64, ny: f64| {
            (nx * nx_true + ny * ny_true)
                .abs()
                .clamp(0.0, 1.0)
                .acos()
                .to_degrees()
        };

        let arena = Bump::new();
        let fitter_seed = ErfEdgeFitter::new(&img, p1, p2, true).expect("edge length too short");
        let sample_cfg = SampleConfig::for_quad(true_len, 1);
        let clean: Vec<(f64, f64, f64)> = fitter_seed
            .collect_samples(&arena, &sample_cfg)
            .iter()
            .copied()
            .collect();
        let (seed_nx, seed_ny, seed_d) = fitter_seed.line_params();
        let corrupted = inject_adjacent_edge_outliers(&clean, seed_nx, seed_ny, seed_d, dark);

        let mut fitter_off = ErfEdgeFitter::new(&img, p1, p2, true).expect("edge length too short");
        let cfg_off = RefineConfig {
            mode: RefineMode::TwoDof,
            ..RefineConfig::quad_style(sigma)
        };
        fitter_off.refine(&corrupted, &cfg_off);
        let (nx_off, ny_off, _) = fitter_off.line_params();
        let err_off = angle_err(nx_off, ny_off);

        let mut fitter_t = ErfEdgeFitter::new(&img, p1, p2, true).expect("edge length too short");
        fitter_t.refine(&corrupted, &RefineConfig::quad_style(sigma));
        let (nx_t, ny_t, _) = fitter_t.line_params();
        let err_t = angle_err(nx_t, ny_t);

        assert!(
            err_t < 0.5,
            "Tukey angle err = {err_t}° (unweighted = {err_off}°)"
        );
        assert!(
            err_t <= err_off + 0.02,
            "Tukey ({err_t}°) regressed vs unweighted ({err_off}°)"
        );
        assert!(fitter_t.last_sigma_hat().is_finite());
    }

    /// Clean (outlier-free) edge with a mis-angled seed: the 1-DOF fit is poor
    /// (frozen at the rotated seed) so the 2-DOF Tukey solve runs, and as it
    /// rotates onto the clean edge the residuals → ~0. Without the σ̂-floor the
    /// scale would collapse and drive every Tukey weight to zero; the floor
    /// keeps σ̂ finite and positive so the solve still recovers the rotation.
    #[test]
    #[allow(clippy::similar_names)]
    fn tukey_sigma_floor_protects_perfect_fit() {
        let width = 160;
        let height = 160;
        let sigma = 0.6;
        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(20.0, 230.0)
            .with_sigma(sigma);

        let p1_gt = [40.0, 60.0];
        let p2_gt = [120.0, 60.0 + 80.0 * 30f64.to_radians().tan()];
        let line_gt = Line::from_points_cw(p1_gt, p2_gt);
        let data = renderer.render_edge_u8(&line_gt);
        let img = ImageView::new(&data, width, height, width).expect("invalid image view");

        // Seed: GT endpoints rotated +1.5° about their midpoint → 1-DOF poor.
        let perturb = 1.5_f64.to_radians();
        let (sp, cp) = perturb.sin_cos();
        let mid = [(p1_gt[0] + p2_gt[0]) * 0.5, (p1_gt[1] + p2_gt[1]) * 0.5];
        let rotate = |p: [f64; 2]| -> [f64; 2] {
            let dx = p[0] - mid[0];
            let dy = p[1] - mid[1];
            [mid[0] + dx * cp - dy * sp, mid[1] + dx * sp + dy * cp]
        };
        let p1 = rotate(p1_gt);
        let p2 = rotate(p2_gt);

        let true_len = (p2_gt[0] - p1_gt[0]).hypot(p2_gt[1] - p1_gt[1]);
        let nx_true = -(p2_gt[1] - p1_gt[1]) / true_len;
        let ny_true = (p2_gt[0] - p1_gt[0]) / true_len;

        let arena = Bump::new();
        let mut fitter = ErfEdgeFitter::new(&img, p1, p2, true).expect("edge length too short");
        let sample_cfg = SampleConfig::for_quad(true_len, 1);
        assert!(fitter.fit(&arena, &sample_cfg, &RefineConfig::quad_style(sigma)));

        let (nx, ny, _) = fitter.line_params();
        let err = (nx * nx_true + ny * ny_true)
            .abs()
            .clamp(0.0, 1.0)
            .acos()
            .to_degrees();
        assert!(err < 0.5, "rotation not recovered: {err}°");
        let sigma_hat = fitter.last_sigma_hat();
        assert!(
            sigma_hat.is_finite() && sigma_hat > 0.0,
            "σ̂ collapsed: {sigma_hat}"
        );
    }

    #[test]
    fn decoder_style_early_exits_on_low_contrast() {
        let width = 80;
        let height = 80;
        let sigma = 0.6;
        let x_gt = 40.25;

        // |B - A| = 3 < min_contrast (5.0) — refinement must break out early and
        // leave the line params essentially at their seed values.
        let renderer = SubpixelEdgeRenderer::new(width, height)
            .with_intensities(127.0, 130.0)
            .with_sigma(sigma);
        let line_gt = Line::from_points_cw([x_gt, 5.0], [x_gt, 75.0]);
        let data = renderer.render_edge_u8(&line_gt);
        let img = ImageView::new(&data, width, height, width).expect("invalid image view");

        let p1 = [40.0, 5.0];
        let p2 = [40.0, 75.0];
        let arena = Bump::new();
        let mut fitter = ErfEdgeFitter::new(&img, p1, p2, false).expect("edge length too short");
        let (_, _, d_before_scan) = fitter.line_params();
        let sample_cfg = SampleConfig::for_decoder();
        let refine_cfg = RefineConfig::decoder_style(sigma);
        // `fit` may return true (samples collected) but the GN loop must bail on
        // min_contrast — so d stays within scan_initial_d's ±2.4 px adjustment.
        fitter.fit(&arena, &sample_cfg, &refine_cfg);
        let (_nx, _ny, d) = fitter.line_params();
        assert!(
            (d - d_before_scan).abs() <= 2.5,
            "low-contrast: d drifted beyond scan window (d_before={d_before_scan}, d_after={d})"
        );
    }
}
