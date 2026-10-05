//! Zero-cost camera distortion model abstractions.
//!
//! This module defines the [`CameraModel`] trait and its three concrete implementations:
//! - [`PinholeModel`]: Ideal pinhole (no distortion). `IS_RECTIFIED = true` causes the
//!   compiler to eliminate all distortion branches in the hot path.
//! - [`BrownConradyModel`]: Standard polynomial radial + tangential distortion (OpenCV convention).
//! - [`KannalaBrandtModel`]: Equidistant fisheye projection model.
//!
//! All models operate on **normalized image coordinates** `(xn, yn)` — i.e., after
//! dividing pixel coordinates by the focal lengths and subtracting the principal point:
//! `xn = (px - cx) / fx`, `yn = (py - cy) / fy`.

/// A compile-time abstraction over camera distortion models.
///
/// Monomorphizing on this trait allows the compiler to completely eliminate all
/// distortion code when `IS_RECTIFIED = true` (e.g., for [`PinholeModel`]),
/// leaving zero overhead for the common rectified-image case.
pub trait CameraModel: Copy + Send + Sync + 'static {
    /// True iff the camera produces a rectified (undistorted) image.
    ///
    /// When `true`, the compiler can statically prove that `distort` and `undistort`
    /// are identity functions and will eliminate all branches guarded by `!C::IS_RECTIFIED`.
    const IS_RECTIFIED: bool;

    /// Map normalized ideal (undistorted) coordinates `(xn, yn)` to normalized
    /// distorted coordinates `(xd, yd)`.
    fn distort(&self, xn: f64, yn: f64) -> [f64; 2];

    /// Map normalized distorted (observed) coordinates `(xd, yd)` back to normalized
    /// ideal (undistorted) coordinates `(xu, yu)`.
    fn undistort(&self, xd: f64, yd: f64) -> [f64; 2];

    /// Compute the 2×2 Jacobian of the distortion map at `(xn, yn)`.
    ///
    /// Returns `[[∂xd/∂xn, ∂xd/∂yn], [∂yd/∂xn, ∂yd/∂yn]]`.
    ///
    /// Used in the LM Jacobian to correctly account for distortion derivatives.
    fn distort_jacobian(&self, xn: f64, yn: f64) -> [[f64; 2]; 2];

    /// The model's purely **radial** forward map `g(r)` and its derivative `g'(r)`, with any
    /// tangential terms excluded.
    ///
    /// `g` takes an ideal radius to the distorted radius it is observed at. Every central
    /// model in this module shares this one-dimensional sub-problem, and it is the only part
    /// of the forward map that needs iterative inversion — which is why
    /// [`RadialInverseTable`] tabulates it once per frame instead of solving it per point.
    ///
    /// Tangential terms are deliberately excluded: they break radial symmetry, so they cannot
    /// be folded into a function of `r` alone. [`CameraModel::tangential`] exposes them
    /// separately, and the tabulated inverse removes them with a fixed-point iteration.
    fn radial_forward(&self, r: f64) -> (f64, f64);

    /// The model's non-radial (tangential) displacement at ideal coordinates `(xn, yn)`.
    ///
    /// Defined so that the forward map splits exactly as
    /// `distort(u) = u · g(|u|)/|u| + tangential(u)`. Purely radial models return zeros,
    /// which is the default.
    #[inline]
    fn tangential(&self, _xn: f64, _yn: f64) -> [f64; 2] {
        [0.0, 0.0]
    }

    /// Whether [`CameraModel::tangential`] is identically zero for *this* instance.
    ///
    /// Not a `const`: Brown-Conrady is purely radial exactly when `p1 == p2 == 0`, which is a
    /// property of the calibration, not of the type. When `true`, inverting the radial
    /// sub-problem is the whole answer and the tangential fixed point is skipped.
    #[inline]
    fn is_purely_radial(&self) -> bool {
        true
    }

    /// [`Self::undistort`], but `None` unless re-distorting lands back on the input within
    /// [`MAX_UNDISTORT_RESIDUAL`].
    ///
    /// This is the only safe way to unproject a point that will then be used as geometry.
    /// `undistort` cannot fail by signature, so outside the model's invertible domain it
    /// returns a point that is not a preimage of its argument; feeding that into a homography
    /// or a pose silently corrupts the result instead of dropping the measurement.
    #[inline]
    #[cfg(feature = "non_rectified")]
    fn undistort_checked(&self, xd: f64, yd: f64) -> Option<[f64; 2]> {
        let [xu, yu] = self.undistort(xd, yd);
        let [xc, yc] = self.distort(xu, yu);
        let dx = xc - xd;
        let dy = yc - yd;
        let residual_sq = dx * dx + dy * dy;
        // Finiteness is checked separately: a NaN fails every comparison, so the bound alone
        // would accept it.
        if residual_sq.is_finite() && residual_sq <= MAX_UNDISTORT_RESIDUAL * MAX_UNDISTORT_RESIDUAL
        {
            Some([xu, yu])
        } else {
            None
        }
    }
}

/// Ideal pinhole camera model (no distortion).
///
/// All methods are `#[inline]` no-ops. The compiler eliminates every
/// code path guarded by `!C::IS_RECTIFIED` at compile time, producing
/// zero-overhead monomorphized code for the standard rectified-image case.
#[derive(Clone, Copy, Debug, Default)]
pub struct PinholeModel;

impl CameraModel for PinholeModel {
    const IS_RECTIFIED: bool = true;

    #[inline]
    fn distort(&self, xn: f64, yn: f64) -> [f64; 2] {
        [xn, yn]
    }

    #[inline]
    fn undistort(&self, xd: f64, yd: f64) -> [f64; 2] {
        [xd, yd]
    }

    #[inline]
    fn distort_jacobian(&self, _xn: f64, _yn: f64) -> [[f64; 2]; 2] {
        [[1.0, 0.0], [0.0, 1.0]]
    }

    #[inline]
    fn radial_forward(&self, r: f64) -> (f64, f64) {
        (r, 1.0)
    }
}

#[cfg(feature = "non_rectified")]
/// Brown-Conrady (OpenCV) lens distortion model.
///
/// Distortion formula (operating on normalized coordinates):
/// ```text
/// r² = xn² + yn²
/// radial = 1 + k1·r² + k2·r⁴ + k3·r⁶
/// xd = xn·radial + 2·p1·xn·yn + p2·(r² + 2·xn²)
/// yd = yn·radial + p1·(r² + 2·yn²) + 2·p2·xn·yn
/// ```
///
/// Coefficient ordering matches OpenCV's `distCoeffs` convention: `[k1, k2, p1, p2, k3]`.
#[derive(Clone, Copy, Debug)]
pub struct BrownConradyModel {
    /// Radial distortion coefficient k1.
    pub k1: f64,
    /// Radial distortion coefficient k2.
    pub k2: f64,
    /// Tangential distortion coefficient p1.
    pub p1: f64,
    /// Tangential distortion coefficient p2.
    pub p2: f64,
    /// Radial distortion coefficient k3.
    pub k3: f64,
}

#[cfg(feature = "non_rectified")]
impl BrownConradyModel {
    /// Construct from a flat coefficient slice `[k1, k2, p1, p2, k3]`.
    ///
    /// # Errors
    /// Returns an error string if `coeffs.len() != 5`.
    pub fn from_coeffs(coeffs: &[f64]) -> Result<Self, &'static str> {
        if coeffs.len() != 5 {
            return Err("BrownConrady requires exactly 5 coefficients: [k1, k2, p1, p2, k3]");
        }
        Ok(Self {
            k1: coeffs[0],
            k2: coeffs[1],
            p1: coeffs[2],
            p2: coeffs[3],
            k3: coeffs[4],
        })
    }
}

/// How far `distort(undistort(u))` may land from `u`, in **normalized-plane** units, for the
/// inverse to be trusted.
///
/// Both models' [`CameraModel::undistort`] is infallible by signature: outside the model's
/// invertible domain it returns its best effort, which is not a preimage. Every consumer that
/// feeds an unprojected point into geometry must therefore re-distort and check against this
/// budget — and must reject non-finite residuals explicitly, since a comparison with NaN is
/// false and a bare `>` would *admit* them.
///
/// 2e-4 normalized is ~0.27 px at the shipped hub datasets' focal lengths, three orders above
/// a converged inverse (~1e-12 normalized) and three below the corner accuracy it protects. It
/// was originally chosen at 2x the `robustness/camera_geometry.rs` round-trip proptest envelope
/// (`< 1e-4`), so a diverging solve bails while well-conditioned points stay.
///
/// Public because it is observable contract, not a private tuning knob: it is the bound the
/// default [`CameraModel::undistort_checked`] applies, so anyone implementing the trait or
/// reading its `None` needs the number.
#[cfg(feature = "non_rectified")]
pub const MAX_UNDISTORT_RESIDUAL: f64 = 2e-4;

/// Knots stored by [`RadialInverseTable`] (so `KNOTS - 1` interpolation intervals).
///
/// 1024 knots is 16 KiB in the frame arena. Only the handful of cache lines a candidate's
/// contour touches are ever hot — one contour spans a narrow radius band — so the table's size
/// costs no cache in practice, and 1024 knots buys enough headroom that the interpolation
/// error (measured 2.0e-5 px on the Kannala-Brandt hub, 2.1e-7 px on Brown-Conrady) sits four
/// orders below the corner accuracy the pipeline reports, which is what keeps every regression
/// snapshot digit unchanged. Halving it would still meet a looser budget, at 3.2e-4 px.
#[cfg(feature = "non_rectified")]
const RADIAL_TABLE_KNOTS: usize = 1024;

/// Knot storage for a disabled [`RadialInverseTable`].
///
/// Never read: a disabled table has `s_max = 0`, and `scale` is only reachable once a caller
/// has established `s <= s_max`. It exists so the live table can hold a *fixed-size array*
/// reference instead of an `Option` or a slice, which is what keeps `scale` branch-free.
#[cfg(feature = "non_rectified")]
static RADIAL_TABLE_UNUSED: [[f64; 2]; RADIAL_TABLE_KNOTS] = [[1.0, 0.0]; RADIAL_TABLE_KNOTS];

/// Interpolation-error budget for [`RadialInverseTable`], in **normalized** units.
///
/// `1e-7` normalized is ~7e-5 px at the shipped hub focal lengths: three orders below any
/// corner accuracy this pipeline has ever measured, and three orders inside
/// [`MAX_UNDISTORT_RESIDUAL`]. The build *verifies* this rather than assuming it, at the knot
/// midpoints where Hermite interpolation is worst, and shrinks its domain until it holds.
///
/// Public because it is observable contract, not a private tuning knob: it is the bound
/// [`RadialInverseTable::build_in`] guarantees and the value
/// [`RadialInverseTable::verified_error`] is meaningful against.
#[cfg(feature = "non_rectified")]
pub const RADIAL_TABLE_BUDGET: f64 = 1e-7;

/// Domain shrink factor applied when a built table misses [`RADIAL_TABLE_BUDGET`].
#[cfg(feature = "non_rectified")]
const RADIAL_TABLE_BACKOFF: f64 = 0.85;

/// How many times [`RadialInverseTable::build_in`] may shrink its domain before giving up and
/// returning a disabled table, which sends every lookup to the iterative solve.
#[cfg(feature = "non_rectified")]
const RADIAL_TABLE_BUILD_ATTEMPTS: usize = 6;

/// Radius at which the monotone-branch probe starts.
///
/// See [`RADIAL_PROBE_MAX`] for why the ceiling doubles as an asymptotic-reach sample.
#[cfg(feature = "non_rectified")]
const RADIAL_PROBE_START: f64 = 0.25;
/// Growth factor of the monotone-branch probe. See [`RADIAL_PROBE_START`].
#[cfg(feature = "non_rectified")]
const RADIAL_PROBE_GROWTH: f64 = 1.5;
/// Ceiling of the monotone-branch probe.
///
/// It doubles as the "asymptotic reach" sample: a model whose `g'` never turns non-positive
/// still has a finite supremum. Kannala-Brandt is the case in point — `g' = dθ_d/dθ · 1/(1+r²)`
/// decays but stays positive, while `g(r) → θ_d(π/2)`, so no distorted radius beyond that
/// exists and `g` at the ceiling is that bound to well inside [`RADIAL_TABLE_BUDGET`].
#[cfg(feature = "non_rectified")]
const RADIAL_PROBE_MAX: f64 = 1.0e4;

/// Bisection steps used to pin the end of a model's monotone branch.
#[cfg(feature = "non_rectified")]
const RADIAL_BISECT_ITERS: usize = 64;
/// Newton budget for one knot solve. Continuation seeding makes two or three steps typical; the
/// budget exists so a pathological model truncates the table instead of spinning.
#[cfg(feature = "non_rectified")]
const RADIAL_SOLVE_ITERS: usize = 32;
/// Relative step tolerance for a knot solve, orders tighter than [`RADIAL_TABLE_BUDGET`] so the
/// stored knots are exact and the verified error is interpolation alone.
///
/// Not sufficient on its own — see [`RADIAL_SOLVE_RESIDUAL_ULPS`]. A Newton *step* is
/// `(g - r_d) / g'`, so where `g'` is small the step's own round-off floor exceeds this
/// tolerance and the iteration can never report convergence however correct its answer is.
#[cfg(feature = "non_rectified")]
const RADIAL_SOLVE_REL_TOL: f64 = 1e-15;

/// Residual tolerance for a knot solve, in ulps of the radius being matched.
///
/// The companion to [`RADIAL_SOLVE_REL_TOL`], and the one that actually binds at the fisheye
/// periphery. Kannala-Brandt has `g'(r) = dθ_d/dθ · 1/(1+r²)`, which decays as `r = tan θ`
/// grows, so around `r ≈ 14` the step `(g - r_d)/g'` has a round-off floor of ~7e-14 against a
/// step tolerance of 1.4e-14: `solve_radial` burned all 32 iterations and returned `None`, the
/// fill reported `Truncated`, and the table silently shrank its domain to 90.4 % of the frame
/// radius — 2.45 % of a 1920x1080 lattice permanently falling back to the per-point solve,
/// precisely where that solve is most expensive. Nothing signalled it: `is_enabled()` stayed
/// true and `verified_error()` reported 1.7e-11 over the *shrunk* domain.
///
/// A residual test has none of that trouble: it asks whether `g(r)` actually equals `r_d`,
/// which is the question, in units that mean something. 8 ulps leaves room for the handful of
/// roundings in a `radial_forward` evaluation. Same correction as the pose LM's step-gate →
/// function-tolerance change in #341, and the same class as the sub-ulp absolute tolerance
/// fixed in the Brown-Conrady inverter in #443 — third instance, so prefer a residual or
/// relative-cost test over a step test by default.
#[cfg(feature = "non_rectified")]
const RADIAL_SOLVE_RESIDUAL_ULPS: f64 = 8.0;

/// Tangential fixed-point iterations used by [`RadialInverseTable::undistort_checked`].
///
/// The iteration is `u ← radial_inverse(xd − tangential(u))`, whose contraction factor is
/// `O(|∂tangential/∂u| / g')` — at OpenCV-scale tangential coefficients (`|p| ~ 1e-4`) about
/// `1e-3` per step, so two steps take the residual from `~3e-4` to `~1e-10` normalized. A
/// forward evaluation then verifies it, so a calibration with implausibly large tangential
/// terms degrades to the iterative solve instead of returning a non-preimage.
#[cfg(feature = "non_rectified")]
const RADIAL_TABLE_TANGENTIAL_ITERS: usize = 2;

/// Tabulated inverse of a camera model's radial forward map, built once per frame.
///
/// # Why this exists
/// Straight-space quad extraction unprojects **every contour point of every candidate** —
/// measured at 83,623 points per frame on the Brown-Conrady hub, 29 % of that stage's CPU time
/// at 122.7 ns per call. The cost was almost entirely iteration and division: the per-point
/// solve ran up to eight radial Newton steps, then a 2-D polish, then a verification
/// evaluation — around nine `f64` divisions in all, each ~20 cycles.
///
/// The radial sub-problem `g(r_u) = r_d` is one-dimensional and monotone, so it can be solved
/// once per frame and read per point. Because `g` is odd, `scale(s) = r_u / r_d` is a *smooth*
/// function of `s = r_d²` — so the table is indexed by `s` directly and a lookup needs **no
/// square root and no division**, just a multiply, a truncation and a cubic Hermite
/// evaluation. Uniform knots in `s` also place resolution where the map bends most, since
/// `ds = 2 r dr`. Tangential terms are then removed by a fixed point that reuses the same
/// table, again without a division.
///
/// # Why it is safer than the iterative solve
/// Knots are filled outward from `s = 0` with each solve seeded from its predecessor, so every
/// stored value lies on the *same monotone branch* as the origin. A lookup therefore cannot
/// land on the mirrored far branch `g(−r) = −g(r)`, which an unguarded polish converges onto
/// and which re-distorts to a tiny residual — a true-but-wrong preimage that a round-trip gate
/// accepts. Here branch safety is structural rather than a guard.
///
/// Outside the verified domain — past the branch limit, past the model's asymptotic reach, or
/// a non-finite input — lookups fall back to [`CameraModel::undistort_checked`], so no
/// behaviour is lost; only the fast path is tabulated.
#[cfg(feature = "non_rectified")]
#[derive(Clone, Copy, Debug)]
pub struct RadialInverseTable<'a> {
    /// `1 / Δs` for the uniform knot grid in `s = r_d²`.
    inv_step: f64,
    /// Largest `s` this table may be read at. Zero disables it.
    s_max: f64,
    /// `[scale, dscale/ds · Δs]` per knot; the derivative is pre-scaled to the unit interval so
    /// the Hermite evaluation needs no extra multiply. Lives in the caller's frame arena, which
    /// is what keeps a 16 KiB table off the stack and out of the system allocator.
    ///
    /// A fixed-size array reference, not a slice: the length *is* the compile-time constant
    /// `RADIAL_TABLE_KNOTS`, and spelling it that way is what lets the compiler discharge both
    /// index checks in `scale` from `i <= KNOTS - 2`. Through a slice the length is opaque, so
    /// every one of the ~83,600 lookups per frame carried two compare-and-panic branches in
    /// the loop whose entire purpose is to be branch-free.
    knots: &'a [[f64; 2]; RADIAL_TABLE_KNOTS],
    /// Worst interpolation error measured at the knot midpoints during the build, in normalized
    /// units. Reported so callers and tests can assert the accuracy actually achieved.
    verified_error: f64,
}

/// Outcome of one knot-filling pass.
#[cfg(feature = "non_rectified")]
enum RadialFill {
    /// Every knot solved; carries the error measured at the knot midpoints.
    Filled(f64),
    /// The monotone branch ended inside the requested domain. Carries the largest `s` whose
    /// knot solved, which *is* the branch limit, so the retry needs no search.
    Truncated(f64),
}

#[cfg(feature = "non_rectified")]
impl<'a> RadialInverseTable<'a> {
    /// Build a table covering normalized distorted radii in `[0, r_d_max]`, with its knots in
    /// `arena`.
    ///
    /// `r_d_max` should be the largest radius the frame can produce, with a margin for the
    /// off-image taps gradient sampling reaches for. The domain is additionally clamped to the
    /// model's monotone branch and to its asymptotic reach, and shrunk until the build-time
    /// verification meets [`RADIAL_TABLE_BUDGET`].
    #[must_use]
    pub fn build_in<C: CameraModel>(
        arena: &'a bumpalo::Bump,
        camera: &C,
        r_d_max: f64,
    ) -> RadialInverseTable<'a> {
        if !r_d_max.is_finite() || r_d_max <= 0.0 {
            return Self::disabled();
        }
        let knots = arena.alloc_slice_fill_copy(RADIAL_TABLE_KNOTS, [0.0_f64; 2]);
        // Exact by construction; `try_into` is how that is spelled without an `unwrap`.
        let Ok(knots) = <&mut [[f64; 2]; RADIAL_TABLE_KNOTS]>::try_from(knots) else {
            return Self::disabled();
        };
        let reach = Self::branch_reach(camera);
        // `reach` is a supremum that is only attained in the limit, so stay strictly inside it.
        let mut r_max = r_d_max.min(reach * (1.0 - f64::EPSILON.sqrt()));
        if !r_max.is_finite() || r_max <= 0.0 {
            return Self::disabled();
        }
        let mut s_max = r_max * r_max;
        for _ in 0..RADIAL_TABLE_BUILD_ATTEMPTS {
            match Self::fill(camera, s_max, knots) {
                RadialFill::Filled(error) => {
                    if error <= RADIAL_TABLE_BUDGET {
                        return Self {
                            inv_step: (RADIAL_TABLE_KNOTS - 1) as f64 / s_max,
                            s_max,
                            knots,
                            verified_error: error,
                        };
                    }
                    s_max *= RADIAL_TABLE_BACKOFF;
                },
                RadialFill::Truncated(s_ok) => {
                    if s_ok.is_nan() || s_ok <= 0.0 {
                        return Self::disabled();
                    }
                    s_max = s_ok;
                },
            }
            r_max = s_max.sqrt();
            if r_max <= 0.0 {
                break;
            }
        }
        Self::disabled()
    }

    /// A table that covers nothing, so every lookup defers to the iterative solve.
    #[must_use]
    const fn disabled() -> RadialInverseTable<'static> {
        RadialInverseTable {
            inv_step: 0.0,
            s_max: 0.0,
            knots: &RADIAL_TABLE_UNUSED,
            verified_error: f64::INFINITY,
        }
    }

    /// Largest distorted radius the model's monotone branch reaches.
    ///
    /// Probes outward for the first radius where `g'` turns non-positive and bisects it; a
    /// model with no such radius still has a finite supremum, sampled at the probe ceiling.
    fn branch_reach<C: CameraModel>(camera: &C) -> f64 {
        let mut lo = 0.0_f64;
        let mut hi = RADIAL_PROBE_START;
        let mut bounded = false;
        // One predicate, used by both the probe and the bisection below. A model whose `g`
        // overflows before `g'` turns over (a large `k3` will do it) ends its usable branch at
        // the overflow, so bisecting on `g' > 0` alone would test a condition that holds across
        // the whole bracket, converge `lo` onto `hi`, and return a non-finite reach — which
        // `f64::min` would then silently discard, voiding the clamp in exactly the case it
        // exists for.
        let on_branch = |r: f64| {
            let (g, dg) = camera.radial_forward(r);
            dg > 0.0 && g.is_finite()
        };
        while hi <= RADIAL_PROBE_MAX {
            if !on_branch(hi) {
                bounded = true;
                break;
            }
            lo = hi;
            hi *= RADIAL_PROBE_GROWTH;
        }
        if !bounded {
            return camera.radial_forward(RADIAL_PROBE_MAX).0;
        }
        for _ in 0..RADIAL_BISECT_ITERS {
            let mid = 0.5 * (lo + hi);
            if on_branch(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let reach = camera.radial_forward(lo).0;
        if reach.is_finite() { reach } else { 0.0 }
    }

    /// Solve `g(r_u) = r_d` on the branch containing `seed`, marching only outward.
    ///
    /// `None` when the branch ends before reaching `r_d`, which is what makes the fill
    /// self-truncating at the true branch limit.
    fn solve_radial<C: CameraModel>(camera: &C, r_d: f64, seed: f64) -> Option<f64> {
        let mut r = seed.max(r_d);
        for _ in 0..RADIAL_SOLVE_ITERS {
            let (g, dg) = camera.radial_forward(r);
            if dg.is_nan() || dg <= 0.0 {
                return None;
            }
            // Converged when the *residual* is at round-off, whatever the step does. See
            // `RADIAL_SOLVE_RESIDUAL_ULPS`: testing the step alone makes convergence
            // unreportable wherever `g'` is small, which is the whole fisheye periphery.
            let residual = g - r_d;
            if residual.abs()
                <= RADIAL_SOLVE_RESIDUAL_ULPS * f64::EPSILON * g.abs().max(r_d).max(1.0)
            {
                return Some(r);
            }
            let step = residual / dg;
            let next = r - step;
            if !next.is_finite() {
                return None;
            }
            r = next.max(0.0);
            if step.abs() <= RADIAL_SOLVE_REL_TOL * r.max(1.0) {
                return Some(r);
            }
        }
        // Neither test was met within budget: refuse rather than store an unconverged knot, so
        // the fill truncates here and the caller falls back to the per-point solve.
        None
    }

    /// Fill `knots` over `[0, s_max]` and measure the error at the knot midpoints.
    fn fill<C: CameraModel>(
        camera: &C,
        s_max: f64,
        knots: &mut [[f64; 2]; RADIAL_TABLE_KNOTS],
    ) -> RadialFill {
        let intervals = RADIAL_TABLE_KNOTS - 1;
        let step = s_max / intervals as f64;
        // `scale(0) = 1 / g'(0)`: the forward map's linearisation at the optical axis.
        let dg0 = camera.radial_forward(NEAR_AXIS_RADIUS).1;
        if dg0.is_nan() || dg0 <= 0.0 {
            return RadialFill::Truncated(0.0);
        }
        knots[0] = [1.0 / dg0, 0.0];
        let mut seed = 0.0_f64;
        for (i, knot) in knots.iter_mut().enumerate().skip(1) {
            let s = step * i as f64;
            let r_d = s.sqrt();
            let Some(r_u) = Self::solve_radial(camera, r_d, seed) else {
                return RadialFill::Truncated(step * (i - 1) as f64);
            };
            seed = r_u;
            let dg = camera.radial_forward(r_u).1;
            if dg.is_nan() || dg <= 0.0 {
                return RadialFill::Truncated(step * (i - 1) as f64);
            }
            // scale = r_u/r_d, dr_u/dr_d = 1/g'(r_u) and ds = 2·r_d·dr_d, so
            // dscale/ds = ((1/g')·r_d − r_u) / r_d² / (2·r_d), pre-scaled by Δs.
            *knot = [
                r_u / r_d,
                (((1.0 / dg) * r_d - r_u) / (r_d * r_d) / (2.0 * r_d)) * step,
            ];
        }
        // Three-point one-sided derivative at the axis knot, in the same pre-scaled units.
        knots[0][1] = 0.5 * (-3.0 * knots[0][0] + 4.0 * knots[1][0] - knots[2][0]);
        let probe = RadialInverseTable {
            inv_step: intervals as f64 / s_max,
            s_max,
            knots,
            verified_error: 0.0,
        };
        let mut worst = 0.0_f64;
        let mut vseed = 0.0_f64;
        for i in 0..intervals {
            let s = step * (i as f64 + 0.5);
            let r_d = s.sqrt();
            let Some(r_u) = Self::solve_radial(camera, r_d, vseed) else {
                return RadialFill::Truncated(step * i as f64);
            };
            vseed = r_u;
            // Compared in normalized distorted units, which is what the residual gate uses.
            let err = (probe.scale(s) * r_d - r_u).abs();
            if err > worst {
                worst = err;
            }
        }
        RadialFill::Filled(worst)
    }

    /// `r_u / r_d` at `s = r_d²`, by cubic Hermite interpolation.
    ///
    /// Callers must have established `s <= self.s_max`, which also implies a non-empty table.
    #[inline]
    #[must_use]
    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "`s` is a sum of squares and `inv_step` is positive, so `t >= 0`; callers have also established `s <= s_max`, which bounds `t` by the interval count, and the `min` below clamps the index regardless"
    )]
    fn scale(&self, s: f64) -> f64 {
        let t = s * self.inv_step;
        // `s <= s_max` bounds `t` by the interval count; the clamp keeps the top interval in
        // range when `t` lands exactly on it, and costs a `min` against a *constant* rather
        // than a branch — which is also what lets the two indexes below compile without bounds
        // checks.
        let i = (t as usize).min(RADIAL_TABLE_KNOTS - 2);
        let u = t - i as f64;
        let [v0, m0] = self.knots[i];
        let [v1, m1] = self.knots[i + 1];
        let u2 = u * u;
        let u3 = u2 * u;
        (2.0 * u3 - 3.0 * u2 + 1.0) * v0
            + (u3 - 2.0 * u2 + u) * m0
            + (-2.0 * u3 + 3.0 * u2) * v1
            + (u3 - u2) * m1
    }

    /// Worst interpolation error measured when this table was built, in normalized units.
    ///
    /// `f64::INFINITY` for a disabled table.
    #[must_use]
    pub const fn verified_error(&self) -> f64 {
        self.verified_error
    }

    /// Whether this table covers anything at all.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.s_max > 0.0
    }

    /// Largest `s = r_d²` this table answers directly. Beyond it, lookups fall back.
    #[must_use]
    pub const fn s_max(&self) -> f64 {
        self.s_max
    }

    /// Tabulated equivalent of [`CameraModel::undistort_checked`].
    ///
    /// Falls back to the model's own iterative solve outside the verified domain, so the
    /// contract is identical: `Some` only for a point that really is a preimage.
    #[inline]
    #[must_use]
    pub fn undistort_checked<C: CameraModel>(
        &self,
        camera: &C,
        xd: f64,
        yd: f64,
    ) -> Option<[f64; 2]> {
        let s = xd * xd + yd * yd;
        if s < NEAR_AXIS_RADIUS * NEAR_AXIS_RADIUS {
            return Some([xd, yd]);
        }
        // Phrased so the tabulated path is the *positive* branch: a NaN radius then falls
        // through to the iterative solve, which rejects it, with no extra comparison.
        if s <= self.s_max {
            let scale = self.scale(s);
            if camera.is_purely_radial() {
                // `g` is then the entire forward map, so the radial solve is exact and was
                // verified at build time; nothing is left to check.
                return Some([xd * scale, yd * scale]);
            }
            let mut xu = xd * scale;
            let mut yu = yd * scale;
            let mut on_table = true;
            for _ in 0..RADIAL_TABLE_TANGENTIAL_ITERS {
                let [tx, ty] = camera.tangential(xu, yu);
                // `xd − tangential(u)` is parallel to `u` with magnitude `g(|u|)`, so the
                // radial table inverts it directly — no Jacobian, no division.
                let ax = xd - tx;
                let ay = yd - ty;
                let sa = ax * ax + ay * ay;
                if sa > self.s_max {
                    on_table = false;
                    break;
                }
                let sc = self.scale(sa);
                xu = ax * sc;
                yu = ay * sc;
            }
            if on_table {
                let [cx, cy] = camera.distort(xu, yu);
                let dx = cx - xd;
                let dy = cy - yd;
                let residual_sq = dx * dx + dy * dy;
                if residual_sq.is_finite()
                    && residual_sq <= MAX_UNDISTORT_RESIDUAL * MAX_UNDISTORT_RESIDUAL
                {
                    return Some([xu, yu]);
                }
            }
        }
        camera.undistort_checked(xd, yd)
    }
}

/// Radius below which every model in this module is the identity to within f64
/// precision, so the inversions can return their input. Shared by both models and
/// both directions, because it states one geometric fact: a point on the optical axis
/// has no radial direction to distort along.
#[cfg(feature = "non_rectified")]
const NEAR_AXIS_RADIUS: f64 = 1e-8;

/// Iteration cap of the Brown-Conrady radial Newton solve. Quadratic convergence from
/// `r_u = r_d` reaches f64 precision in 3-5 steps across the calibrations we ship.
#[cfg(feature = "non_rectified")]
const BC_RADIAL_MAX_ITERS: usize = 8;

/// **Relative** step tolerance of that solve. Relative, not absolute: at the wide-angle
/// radii a fisheye reaches (`r_u > 4.5`) an absolute 1e-15 sits below 1 ulp and can never
/// be met, so the loop would always burn its full iteration budget.
#[cfg(feature = "non_rectified")]
const BC_RADIAL_REL_TOL: f64 = 1e-14;

/// Iteration cap of the 2-D tangential polish (2-3 steps in practice).
#[cfg(feature = "non_rectified")]
const BC_POLISH_MAX_ITERS: usize = 4;

/// Squared-residual floor of that polish, in normalized units. 1e-24 is 1e-12 normalized,
/// i.e. ~1.4e-9 px at the shipped hub focal length — five orders below the
/// `quad::MAX_UNDISTORT_RESIDUAL` gate this feeds, and one polish step cheaper than
/// chasing the last few ulps.
#[cfg(feature = "non_rectified")]
const BC_POLISH_RESIDUAL_SQ: f64 = 1e-24;

/// Determinant floor below which the polish's 2x2 solve is treated as singular.
#[cfg(feature = "non_rectified")]
const BC_POLISH_MIN_DET: f64 = 1e-12;

#[cfg(feature = "non_rectified")]
impl BrownConradyModel {
    /// The radial map `g(r) = r·(1 + k1·r² + k2·r⁴ + k3·r⁶)` and its derivative.
    #[inline]
    fn radial_and_derivative(&self, r: f64) -> (f64, f64) {
        let r2 = r * r;
        let r4 = r2 * r2;
        let r6 = r2 * r4;
        (
            r * (1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6),
            1.0 + 3.0 * self.k1 * r2 + 5.0 * self.k2 * r4 + 7.0 * self.k3 * r6,
        )
    }

    /// Forward map and its Jacobian in one pass, sharing the radial monomials.
    ///
    /// The Jacobian is **symmetric** — `∂xd/∂yn` and `∂yd/∂xn` are the same expression —
    /// so it is returned as `[a, b, d]` for `[[a, b], [b, d]]` and callers can use
    /// `det = a·d − b²`. Used by the polish, which would otherwise evaluate `r²/r⁴/r⁶`
    /// and `radial` twice per step.
    #[inline]
    fn distort_with_jacobian(&self, xn: f64, yn: f64) -> ([f64; 2], [f64; 3]) {
        let r2 = xn * xn + yn * yn;
        let r4 = r2 * r2;
        let r6 = r2 * r4;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        // ∂radial/∂r² = k1 + 2·k2·r² + 3·k3·r⁴
        let d_radial_dr2 = self.k1 + 2.0 * self.k2 * r2 + 3.0 * self.k3 * r4;
        let two_xy_dr = 2.0 * xn * yn * d_radial_dr2;
        (
            [
                xn * radial + 2.0 * self.p1 * xn * yn + self.p2 * (r2 + 2.0 * xn * xn),
                yn * radial + self.p1 * (r2 + 2.0 * yn * yn) + 2.0 * self.p2 * xn * yn,
            ],
            [
                radial + 2.0 * xn * xn * d_radial_dr2 + 2.0 * self.p1 * yn + 6.0 * self.p2 * xn,
                two_xy_dr + 2.0 * self.p1 * xn + 2.0 * self.p2 * yn,
                radial + 2.0 * yn * yn * d_radial_dr2 + 6.0 * self.p1 * yn + 2.0 * self.p2 * xn,
            ],
        )
    }
}

#[cfg(feature = "non_rectified")]
impl CameraModel for BrownConradyModel {
    const IS_RECTIFIED: bool = false;

    fn distort(&self, xn: f64, yn: f64) -> [f64; 2] {
        let r2 = xn * xn + yn * yn;
        let r4 = r2 * r2;
        let r6 = r2 * r4;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        let xd = xn * radial + 2.0 * self.p1 * xn * yn + self.p2 * (r2 + 2.0 * xn * xn);
        let yd = yn * radial + self.p1 * (r2 + 2.0 * yn * yn) + 2.0 * self.p2 * xn * yn;
        [xd, yd]
    }

    fn undistort(&self, xd: f64, yd: f64) -> [f64; 2] {
        // Two-stage inversion.
        //
        // Stage 1 solves the *radial* sub-problem, which is one-dimensional and strictly
        // monotone on `[0, r_turn)`, where `r_turn` is the first positive root of
        // `g'(r)`: find `r_u` with `g(r_u) = r_d`. Newton from `r_u = r_d` converges
        // quadratically, and is **exact** when `p1 = p2 = 0`, which is why stage 2 is
        // skipped entirely in that (common) case.
        //
        // Stage 2 is a 2-D Newton polish against the full forward map, absorbing the
        // tangential terms stage 1 ignores. It runs only if stage 1 converged.
        //
        // Both stages replace a 5-iteration *fixed point* (`xu ← (xd − dx(xu))/radial(xu)`,
        // only linearly convergent). At the coefficients of the shipped Brown-Conrady hub
        // dataset (`k1 = -0.28`, `k2 = 0.08`) that iteration had not converged after 5
        // steps and left **0.42 px** of round-trip error at the image corner — above
        // `quad.rs`'s `MAX_UNDISTORT_RESIDUAL`, so it silently discarded every candidate
        // touching ~1.8 % of the frame.
        //
        // **Out-of-domain inputs must not be polished.** Beyond `g(r_turn)` the radius
        // `r_d` has no preimage on the invertible branch, but the *full* map still has
        // one: the mirrored far branch, `g(-r) = -g(r)`. A polish started past the
        // turning point converges onto it and re-distorts to ~1e-14, so the caller's
        // residual gate would ACCEPT a point at several times the correct radius and the
        // wrong sign (`k1 = -0.1`, `r_d = 1.335` returns `r_u = -3.69`). Stage 1
        // therefore restores the last on-branch iterate and reports non-convergence,
        // which keeps a large residual and makes the gate reject — the behaviour the
        // superseded fixed point had by accident.
        let r_d = (xd * xd + yd * yd).sqrt();
        if r_d < NEAR_AXIS_RADIUS {
            return [xd, yd];
        }

        let mut r_u = r_d;
        let mut last_on_branch = r_d;
        let mut converged = false;
        for _ in 0..BC_RADIAL_MAX_ITERS {
            let (g, dg) = self.radial_and_derivative(r_u);
            if dg <= 0.0 {
                r_u = last_on_branch;
                break;
            }
            last_on_branch = r_u;
            let step = (g - r_d) / dg;
            r_u = (r_u - step).max(0.0);
            if step.abs() <= BC_RADIAL_REL_TOL * r_u.max(1.0) {
                converged = true;
                break;
            }
        }

        let scale = r_u / r_d;
        let mut xu = xd * scale;
        let mut yu = yd * scale;

        // Stage 1 is the exact inverse without tangential terms, and must not run when it
        // did not converge (see above).
        if !converged || (self.p1 == 0.0 && self.p2 == 0.0) {
            return [xu, yu];
        }

        for _ in 0..BC_POLISH_MAX_ITERS {
            let ([fx, fy], [a, b, d]) = self.distort_with_jacobian(xu, yu);
            let rx = fx - xd;
            let ry = fy - yd;
            if rx * rx + ry * ry < BC_POLISH_RESIDUAL_SQ {
                break;
            }
            // Symmetric Jacobian: det = a·d − b².
            let det = a * d - b * b;
            if det.abs() < BC_POLISH_MIN_DET {
                break;
            }
            xu -= (d * rx - b * ry) / det;
            yu -= (a * ry - b * rx) / det;
        }
        [xu, yu]
    }

    #[inline]
    fn radial_forward(&self, r: f64) -> (f64, f64) {
        self.radial_and_derivative(r)
    }

    #[inline]
    fn tangential(&self, xn: f64, yn: f64) -> [f64; 2] {
        let r2 = xn * xn + yn * yn;
        [
            2.0 * self.p1 * xn * yn + self.p2 * (r2 + 2.0 * xn * xn),
            self.p1 * (r2 + 2.0 * yn * yn) + 2.0 * self.p2 * xn * yn,
        ]
    }

    #[inline]
    fn is_purely_radial(&self) -> bool {
        self.p1 == 0.0 && self.p2 == 0.0
    }

    fn distort_jacobian(&self, xn: f64, yn: f64) -> [[f64; 2]; 2] {
        // One expression, one place: `distort_with_jacobian` returns the symmetric
        // `[a, b, d]`; this expands it to the trait's dense 2x2.
        let (_, [a, b, d]) = self.distort_with_jacobian(xn, yn);
        [[a, b], [b, d]]
    }
}

#[cfg(feature = "non_rectified")]
/// Kannala-Brandt equidistant fisheye camera model.
///
/// Projection formula (operating on normalized coordinates):
/// ```text
/// r     = √(xn² + yn²)
/// θ     = atan(r)
/// θ_d   = θ·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
/// xd    = (θ_d / r) · xn
/// yd    = (θ_d / r) · yn
/// ```
///
/// Coefficient ordering: `[k1, k2, k3, k4]`.
#[derive(Clone, Copy, Debug)]
pub struct KannalaBrandtModel {
    /// Fisheye distortion coefficient k1.
    pub k1: f64,
    /// Fisheye distortion coefficient k2.
    pub k2: f64,
    /// Fisheye distortion coefficient k3.
    pub k3: f64,
    /// Fisheye distortion coefficient k4.
    pub k4: f64,
}

/// Upper clamp on the recovered incidence angle θ before `tan(θ)` in KB `undistort`,
/// so a Newton iterate that diverges past the calibrated fisheye envelope produces a
/// large-but-finite radius (caught by the round-trip residual gate) rather than ±∞.
#[cfg(feature = "non_rectified")]
const MAX_KB_THETA: f64 = core::f64::consts::FRAC_PI_2 - 1e-3;

#[cfg(feature = "non_rectified")]
impl KannalaBrandtModel {
    /// Construct from a flat coefficient slice `[k1, k2, k3, k4]`.
    ///
    /// # Errors
    /// Returns an error string if `coeffs.len() != 4`.
    pub fn from_coeffs(coeffs: &[f64]) -> Result<Self, &'static str> {
        if coeffs.len() != 4 {
            return Err("KannalaBrandt requires exactly 4 coefficients: [k1, k2, k3, k4]");
        }
        Ok(Self {
            k1: coeffs[0],
            k2: coeffs[1],
            k3: coeffs[2],
            k4: coeffs[3],
        })
    }

    /// Evaluate the angle polynomial and its derivative at θ.
    /// Returns `(θ_d, dθ_d/dθ)`.
    #[inline]
    fn angle_poly(&self, theta: f64) -> (f64, f64) {
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let t6 = t2 * t4;
        let t8 = t4 * t4;
        let theta_d = theta * (1.0 + self.k1 * t2 + self.k2 * t4 + self.k3 * t6 + self.k4 * t8);
        let dtheta_d =
            1.0 + 3.0 * self.k1 * t2 + 5.0 * self.k2 * t4 + 7.0 * self.k3 * t6 + 9.0 * self.k4 * t8;
        (theta_d, dtheta_d)
    }
}

#[cfg(feature = "non_rectified")]
impl CameraModel for KannalaBrandtModel {
    const IS_RECTIFIED: bool = false;

    fn distort(&self, xn: f64, yn: f64) -> [f64; 2] {
        let r = (xn * xn + yn * yn).sqrt();
        if r < NEAR_AXIS_RADIUS {
            return [xn, yn];
        }
        // theta = atan(r) since the point is at depth z=1 in normalized coords
        let theta = r.atan();
        let (theta_d, _) = self.angle_poly(theta);
        let scale = theta_d / r;
        [xn * scale, yn * scale]
    }

    fn undistort(&self, xd: f64, yd: f64) -> [f64; 2] {
        let r_d = (xd * xd + yd * yd).sqrt();
        if r_d < NEAR_AXIS_RADIUS {
            return [xd, yd];
        }
        // Invert θ_d = poly(θ) via Newton's method to recover θ, then r = tan(θ).
        let mut theta = r_d; // initial guess: identity
        for _ in 0..10 {
            let (theta_d, d_theta_d) = self.angle_poly(theta);
            let f = theta_d - r_d;
            if f.abs() < 1e-12 {
                break;
            }
            theta -= f / d_theta_d.max(1e-8);
            theta = theta.max(0.0);
        }
        // r_undistorted = tan(θ). Clamp θ just below π/2 so a Newton iterate driven
        // into the divergent tail (extreme fisheye beyond the calibrated envelope)
        // cannot make `tan(θ)` blow up to ±∞; the resulting large-but-finite radius
        // is then caught by the undistort round-trip residual gate.
        let r_undist = theta.min(MAX_KB_THETA).tan();
        let scale = r_undist / r_d;
        [xd * scale, yd * scale]
    }

    #[inline]
    fn radial_forward(&self, r: f64) -> (f64, f64) {
        // g(r) = theta_d(atan(r)); g'(r) = (d theta_d / d theta) * d(atan r)/dr.
        let theta = r.atan();
        let (theta_d, dtheta_d) = self.angle_poly(theta);
        (theta_d, dtheta_d / (1.0 + r * r))
    }

    #[expect(
        clippy::similar_names,
        reason = "partial-derivative vars (dxd_dxn/dxd_dyn/dyd_dxn/dyd_dyn) match the Jacobian math notation"
    )]
    fn distort_jacobian(&self, xn: f64, yn: f64) -> [[f64; 2]; 2] {
        let r2 = xn * xn + yn * yn;
        let r = r2.sqrt();
        if r < NEAR_AXIS_RADIUS {
            // Near the optical axis, the equidistant model approaches identity.
            return [[1.0, 0.0], [0.0, 1.0]];
        }

        let theta = r.atan();
        let (theta_d, dtheta_d_dtheta) = self.angle_poly(theta);

        // scale = θ_d / r
        // ∂scale/∂xn = (∂θ_d/∂xn · r - θ_d · ∂r/∂xn) / r²
        //
        // ∂r/∂xn = xn / r
        // ∂θ/∂r  = 1 / (1 + r²)
        // ∂θ/∂xn = (xn / r) / (1 + r²)
        // ∂θ_d/∂xn = dθ_d_dθ · ∂θ/∂xn = dθ_d_dθ · xn / (r · (1 + r²))

        let one_plus_r2 = 1.0 + r2;
        let dthetad_dxn = dtheta_d_dtheta * xn / (r * one_plus_r2);
        let dthetad_dyn = dtheta_d_dtheta * yn / (r * one_plus_r2);

        let dscale_dxn = (dthetad_dxn * r - theta_d * (xn / r)) / r2;
        let dscale_dyn = (dthetad_dyn * r - theta_d * (yn / r)) / r2;

        // xd = scale · xn  →  ∂xd/∂xn = scale + xn · ∂scale/∂xn
        let dxd_dxn = theta_d / r + xn * dscale_dxn;
        let dxd_dyn = xn * dscale_dyn;
        let dyd_dxn = yn * dscale_dxn;
        let dyd_dyn = theta_d / r + yn * dscale_dyn;

        [[dxd_dxn, dxd_dyn], [dyd_dxn, dyd_dyn]]
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn pinhole_is_identity() {
        let m = PinholeModel;
        let [xd, yd] = m.distort(0.3, -0.2);
        assert!((xd - 0.3).abs() < f64::EPSILON);
        assert!((yd - (-0.2)).abs() < f64::EPSILON);
        let [xu, yu] = m.undistort(0.3, -0.2);
        assert!((xu - 0.3).abs() < f64::EPSILON);
        assert!((yu - (-0.2)).abs() < f64::EPSILON);
    }

    #[cfg(feature = "non_rectified")]
    mod non_rectified {
        use super::*;

        /// Round-trip identity: distort then undistort should recover the original point.
        fn check_roundtrip<C: CameraModel>(model: &C, xn: f64, yn: f64, tol: f64) {
            let [xd, yd] = model.distort(xn, yn);
            let [xu, yu] = model.undistort(xd, yd);
            assert!((xu - xn).abs() < tol, "xn round-trip failed: {xu} vs {xn}");
            assert!((yu - yn).abs() < tol, "yn round-trip failed: {yu} vs {yn}");
        }

        /// Numerical Jacobian check via **central** differences, with the step
        /// scaled by the operating radius so the check stays meaningful at the
        /// fisheye periphery (where `r_u = tan(θ)` reaches ~11.7 on the hub
        /// dataset, far outside the `r ≤ 0.5` box the spot-checks below use).
        #[allow(clippy::similar_names)]
        fn check_jacobian<C: CameraModel>(model: &C, xn: f64, yn: f64) {
            let eps = 1e-6 * xn.hypot(yn).max(1.0);
            let jac = model.distort_jacobian(xn, yn);

            let [xp, yp_x] = model.distort(xn + eps, yn);
            let [xm, ym_x] = model.distort(xn - eps, yn);
            let [xp_y, yp] = model.distort(xn, yn + eps);
            let [xm_y, ym] = model.distort(xn, yn - eps);

            let num_dxd_dxn = (xp - xm) / (2.0 * eps);
            let num_dxd_dyn = (xp_y - xm_y) / (2.0 * eps);
            let num_dyd_dxn = (yp_x - ym_x) / (2.0 * eps);
            let num_dyd_dyn = (yp - ym) / (2.0 * eps);

            assert!(
                (jac[0][0] - num_dxd_dxn).abs() < 1e-5,
                "∂xd/∂xn: analytic={} numeric={num_dxd_dxn}",
                jac[0][0]
            );
            assert!(
                (jac[0][1] - num_dxd_dyn).abs() < 1e-5,
                "∂xd/∂yn: analytic={} numeric={num_dxd_dyn}",
                jac[0][1]
            );
            assert!(
                (jac[1][0] - num_dyd_dxn).abs() < 1e-5,
                "∂yd/∂xn: analytic={} numeric={num_dyd_dxn}",
                jac[1][0]
            );
            assert!(
                (jac[1][1] - num_dyd_dyn).abs() < 1e-5,
                "∂yd/∂yn: analytic={} numeric={num_dyd_dyn}",
                jac[1][1]
            );
        }

        #[test]
        fn brown_conrady_roundtrip() {
            let m = BrownConradyModel {
                k1: -0.3,
                k2: 0.1,
                p1: 0.001,
                p2: -0.002,
                k3: 0.0,
            };
            for &(xn, yn) in &[(0.1, 0.2), (-0.3, 0.15), (0.0, 0.4)] {
                check_roundtrip(&m, xn, yn, 1e-7);
            }
        }

        #[test]
        fn brown_conrady_jacobian() {
            let m = BrownConradyModel {
                k1: -0.3,
                k2: 0.1,
                p1: 0.001,
                p2: -0.002,
                k3: 0.0,
            };
            for &(xn, yn) in &[(0.1, 0.2), (-0.3, 0.15), (0.05, -0.05)] {
                check_jacobian(&m, xn, yn);
            }
        }

        #[test]
        fn kannala_brandt_roundtrip() {
            let m = KannalaBrandtModel {
                k1: 0.1,
                k2: -0.01,
                k3: 0.001,
                k4: 0.0,
            };
            for &(xn, yn) in &[(0.1, 0.2), (-0.3, 0.15), (0.5, 0.5)] {
                check_roundtrip(&m, xn, yn, 1e-7);
            }
        }

        #[test]
        fn kannala_brandt_jacobian() {
            let m = KannalaBrandtModel {
                k1: 0.1,
                k2: -0.01,
                k3: 0.001,
                k4: 0.0,
            };
            for &(xn, yn) in &[(0.1, 0.2), (-0.3, 0.15), (0.3, -0.3)] {
                check_jacobian(&m, xn, yn);
            }
        }

        #[test]
        fn kannala_brandt_undistort_stays_finite_at_extreme_radius() {
            // A large distorted radius drives the Newton iterate toward θ→π/2, where
            // an unclamped `tan(θ)` would blow up to ±∞. The MAX_KB_THETA clamp must
            // keep undistort finite (the round-trip residual gate then rejects it).
            let m = KannalaBrandtModel {
                k1: 0.1,
                k2: 0.01,
                k3: 0.0,
                k4: 0.0,
            };
            for &(xd, yd) in &[(5.0, 5.0), (50.0, 0.0), (0.0, 100.0)] {
                let out = m.undistort(xd, yd);
                assert!(
                    out[0].is_finite() && out[1].is_finite(),
                    "undistort must stay finite at extreme radius, got {out:?} for ({xd}, {yd})"
                );
            }
        }

        // ── The shipped hub datasets' own operating point ──────────────────
        //
        // `aprilgrid_distortion_{brown_conrady,kannala_brandt}_v1_1920x1080`.
        // The spot-checks above all sit inside `r ≤ 0.5`; the regression suite
        // runs a 1920×1080 frame whose corner reaches `r_d = 0.80` (Brown-Conrady)
        // and `θ = 85.1°` (Kannala-Brandt). Pinning the inverters *there* is what
        // the earlier benign points could not do.

        // Every literal below is read from the dataset, not chosen: `distortion_coeffs`,
        // `k_matrix` and `resolution` of the first TAG record of
        // `tests/data/hub_cache/aprilgrid_distortion_{brown_conrady,kannala_brandt}_v1_1920x1080/rich_truth.json`,
        // the subsets pinned by `xtask/datasets.toml` (`[hub]`, `revision`). Both K
        // matrices are `fx == fy` with the principal point at the exact frame centre, so
        // one `focal` is the camera, not an approximation. If a re-render or a pin bump
        // moves the operating point, update these together with the pin.
        const HUB_CX: f64 = 960.0;
        const HUB_CY: f64 = 540.0;
        const HUB_WIDTH: u32 = 1920;
        const HUB_HEIGHT: u32 = 1080;
        const HUB_BC_FOCAL: f64 = 1_371.022_086_472_430_1;
        const HUB_KB_FOCAL: f64 = 742.454_608_851_902_9;

        /// `distortion_coeffs` of the Brown-Conrady hub subset.
        fn hub_brown_conrady() -> BrownConradyModel {
            BrownConradyModel {
                k1: -0.28,
                k2: 0.08,
                p1: 0.0002,
                p2: 0.0001,
                k3: 0.0,
            }
        }

        /// `distortion_coeffs` of the Kannala-Brandt hub subset.
        fn hub_kannala_brandt() -> KannalaBrandtModel {
            KannalaBrandtModel {
                k1: -0.0035,
                k2: 0.0015,
                k3: -0.0003,
                k4: 0.0001,
            }
        }

        /// Worst `distort(undistort(u)) − u` over the frame, in **pixels**.
        fn max_round_trip_residual_px<C: CameraModel>(model: &C, focal: f64, stride: u32) -> f64 {
            let mut worst = 0.0_f64;
            let mut py = 0;
            while py <= HUB_HEIGHT {
                let mut px = 0;
                while px <= HUB_WIDTH {
                    let xd = (f64::from(px) - HUB_CX) / focal;
                    let yd = (f64::from(py) - HUB_CY) / focal;
                    if xd.hypot(yd) > 1e-9 {
                        let [xu, yu] = model.undistort(xd, yd);
                        let [xc, yc] = model.distort(xu, yu);
                        worst = worst.max((xc - xd).hypot(yc - yd) * focal);
                    }
                    px += stride;
                }
                py += stride;
            }
            worst
        }

        #[test]
        fn brown_conrady_round_trip_is_exact_over_the_hub_frame() {
            let worst = max_round_trip_residual_px(&hub_brown_conrady(), HUB_BC_FOCAL, 8);
            // Budget, not a round number: the polish stops at `BC_POLISH_RESIDUAL_SQ`
            // (1e-24 squared-normalized = 1e-12 normalized), which at this focal length is
            // 1.4e-9 px. The superseded fixed point scored 4.2e-1 px here, so this bound
            // is still eight orders of margin away from a convergence failure.
            assert!(
                worst < 1e-8,
                "Brown-Conrady round trip is {worst:.3e} px at worst over the hub frame; \
                 above 1e-8 px means the inverter stopped converging"
            );
        }

        #[test]
        fn kannala_brandt_round_trip_is_exact_over_the_hub_frame() {
            let worst = max_round_trip_residual_px(&hub_kannala_brandt(), HUB_KB_FOCAL, 8);
            assert!(
                worst < 1e-6,
                "Kannala-Brandt round trip is {worst:.3e} px at worst over the hub frame"
            );
        }

        /// Table domain bound for the hub frames, mirroring `quad.rs::frame_radius_bound`.
        fn hub_radius_bound(focal: f64) -> f64 {
            let dx = HUB_CX.max(f64::from(HUB_WIDTH) - HUB_CX) + 8.0;
            let dy = HUB_CY.max(f64::from(HUB_HEIGHT) - HUB_CY) + 8.0;
            (dx / focal).hypot(dy / focal)
        }

        /// Worst disagreement between the tabulated inverse and the model's own iterative
        /// solve, in **pixels**, over the whole frame on a `stride`-pixel lattice.
        fn max_table_disagreement_px<C: CameraModel>(
            model: &C,
            focal: f64,
            stride: u32,
        ) -> (f64, usize, usize) {
            let arena = bumpalo::Bump::new();
            let table = RadialInverseTable::build_in(&arena, model, hub_radius_bound(focal));
            let mut worst = 0.0_f64;
            let mut tabulated = 0;
            let mut fell_back = 0;
            let mut py = 0;
            while py <= HUB_HEIGHT {
                let mut px = 0;
                while px <= HUB_WIDTH {
                    let xd = (f64::from(px) - HUB_CX) / focal;
                    let yd = (f64::from(py) - HUB_CY) / focal;
                    let exact = model.undistort_checked(xd, yd);
                    let fast = table.undistort_checked(model, xd, yd);
                    assert_eq!(
                        exact.is_some(),
                        fast.is_some(),
                        "table and iterative solve disagree on invertibility at ({px}, {py}): \
                         exact={exact:?} fast={fast:?}"
                    );
                    if let (Some(e), Some(f)) = (exact, fast) {
                        worst = worst.max((e[0] - f[0]).hypot(e[1] - f[1]) * focal);
                        if xd.hypot(yd).powi(2) <= table.s_max() {
                            tabulated += 1;
                        } else {
                            fell_back += 1;
                        }
                    } else {
                        fell_back += 1;
                    }
                    px += stride;
                }
                py += stride;
            }
            (worst, tabulated, fell_back)
        }

        #[test]
        fn radial_table_agrees_with_the_iterative_solve_on_the_brown_conrady_hub() {
            let (worst, tabulated, fell_back) =
                max_table_disagreement_px(&hub_brown_conrady(), HUB_BC_FOCAL, 4);
            // The table's own verification bounds the radial interpolation error at
            // `RADIAL_TABLE_BUDGET` normalized; the tangential fixed point converges well
            // below that. 1e-3 px leaves an order of margin and is still two orders under
            // any corner accuracy this pipeline reports.
            assert!(
                worst < 1e-3,
                "tabulated inverse differs from the iterative solve by {worst:.3e} px"
            );
            assert_eq!(
                fell_back,
                0,
                "the Brown-Conrady hub frame should be fully tabulated, but {fell_back} of \
                 {} samples fell back to the iterative solve",
                tabulated + fell_back
            );
        }

        #[test]
        fn radial_table_agrees_with_the_iterative_solve_on_the_kannala_brandt_hub() {
            let (worst, tabulated, fell_back) =
                max_table_disagreement_px(&hub_kannala_brandt(), HUB_KB_FOCAL, 4);
            assert!(
                worst < 1e-3,
                "tabulated inverse differs from the iterative solve by {worst:.3e} px"
            );
            // Kannala-Brandt cannot produce a distorted radius beyond `θ_d(π/2)`, so the
            // extreme frame corners legitimately have no preimage and are left to the
            // iterative path. Guard the share, so a build regression that silently disables
            // the table (and with it the whole optimisation) fails here.
            let total = tabulated + fell_back;
            assert!(
                tabulated * 100 >= total * 95,
                "only {tabulated}/{total} samples were tabulated; the table should cover \
                 at least 95 % of the Kannala-Brandt hub frame"
            );
        }

        #[test]
        fn radial_table_meets_its_verified_error_budget() {
            let arena = bumpalo::Bump::new();
            for (name, err) in [
                (
                    "brown_conrady",
                    RadialInverseTable::build_in(
                        &arena,
                        &hub_brown_conrady(),
                        hub_radius_bound(HUB_BC_FOCAL),
                    )
                    .verified_error(),
                ),
                (
                    "kannala_brandt",
                    RadialInverseTable::build_in(
                        &arena,
                        &hub_kannala_brandt(),
                        hub_radius_bound(HUB_KB_FOCAL),
                    )
                    .verified_error(),
                ),
            ] {
                assert!(
                    err <= RADIAL_TABLE_BUDGET,
                    "{name}: verified error {err:.3e} exceeds the budget \
                     {RADIAL_TABLE_BUDGET:.3e}"
                );
            }
        }

        #[test]
        fn radial_table_cannot_reach_the_mirrored_far_branch() {
            // `g(r) = r(1 − 0.1 r²)` turns over at r = 1.826 with `g(r_turn) = 1.217`, and the
            // *full* map still has a preimage past it on the mirrored branch `g(−r) = −g(r)`:
            // an unguarded 2-D polish converges to `r_u = −3.69`, which re-distorts to ~1e-14
            // and so passes a round-trip gate. The table is filled outward from the origin
            // along one branch, so it can only ever refuse.
            let model = BrownConradyModel {
                k1: -0.1,
                k2: 0.0,
                p1: 0.0,
                p2: 0.0,
                k3: 0.0,
            };
            let arena = bumpalo::Bump::new();
            let table = RadialInverseTable::build_in(&arena, &model, 4.0);
            for &r_d in &[1.3, 1.335, 1.5, 2.0, 3.0] {
                let got = table.undistort_checked(&model, r_d, 0.0);
                assert!(
                    got.is_none_or(|[xu, _]| (0.0..=1.9).contains(&xu)),
                    "r_d = {r_d} inverted to {got:?}, which is off the monotone branch \
                     (0 ≤ r_u ≤ 1.826)"
                );
            }
        }

        #[test]
        fn radial_table_rejects_non_finite_input() {
            let model = hub_kannala_brandt();
            let arena = bumpalo::Bump::new();
            let table =
                RadialInverseTable::build_in(&arena, &model, hub_radius_bound(HUB_KB_FOCAL));
            for &(xd, yd) in &[
                (f64::NAN, 0.0),
                (0.0, f64::NAN),
                (f64::INFINITY, 0.0),
                (f64::NEG_INFINITY, 1.0),
            ] {
                assert!(
                    table.undistort_checked(&model, xd, yd).is_none(),
                    "({xd}, {yd}) must not invert"
                );
            }
        }

        #[test]
        fn radial_table_build_degrades_instead_of_failing() {
            // A degenerate request must produce a disabled table, not a panic or a table that
            // silently returns garbage: every lookup then defers to the iterative solve.
            let model = hub_brown_conrady();
            let arena = bumpalo::Bump::new();
            for bound in [0.0, -1.0, f64::NAN, f64::INFINITY] {
                let table = RadialInverseTable::build_in(&arena, &model, bound);
                assert!(
                    !table.is_enabled(),
                    "bound {bound} should disable the table"
                );
                // Still answers correctly, via the fallback.
                let exact = model.undistort_checked(0.1, 0.05);
                let got = table.undistort_checked(&model, 0.1, 0.05);
                assert_eq!(exact.is_some(), got.is_some());
            }
        }

        #[test]
        fn jacobians_hold_at_the_hub_frame_periphery() {
            let bc = hub_brown_conrady();
            // r_u = 1.0055 at the image corner (field angle 45.2°).
            let r = 1.0055 / core::f64::consts::SQRT_2;
            check_jacobian(&bc, r, r);
            check_jacobian(&bc, 1.0055, 0.0);

            let kb = hub_kannala_brandt();
            // θ = 85.1° at the image corner, i.e. r_u = tan θ = 11.68.
            for theta_deg in [45.0_f64, 75.0, 85.1] {
                let r = theta_deg.to_radians().tan();
                check_jacobian(
                    &kb,
                    r / core::f64::consts::SQRT_2,
                    r / core::f64::consts::SQRT_2,
                );
            }
        }

        #[test]
        fn brown_conrady_distort_jacobian_is_symmetric() {
            // `distort_with_jacobian` returns `[a, b, d]` on the strength of this; the
            // polish then uses `det = a·d − b²`.
            let m = hub_brown_conrady();
            for &(xn, yn) in &[(0.31, -0.17), (0.9, 0.42), (-0.6, 0.6), (1.0055, 0.0)] {
                let j = m.distort_jacobian(xn, yn);
                assert!(
                    (j[0][1] - j[1][0]).abs() < f64::EPSILON,
                    "off-diagonals differ at ({xn}, {yn}): {} vs {}",
                    j[0][1],
                    j[1][0]
                );
            }
        }

        /// A radius past the model's radial turning point has no preimage on the
        /// invertible branch, but the full map still has the mirrored one
        /// (`g(-r) = -g(r)`), which re-distorts exactly. `undistort` must not return it:
        /// `quad.rs` only re-distorts and compares against `MAX_UNDISTORT_RESIDUAL`, so a
        /// far-branch point would be **accepted** at several times the correct radius and
        /// the wrong sign.
        #[test]
        fn brown_conrady_undistort_stays_on_the_invertible_branch() {
            const GATE: f64 = 2e-4; // quad::MAX_UNDISTORT_RESIDUAL

            // g(r) = r − 0.1·r³ turns at r = 1.826, where g = 1.217.
            let m = BrownConradyModel {
                k1: -0.1,
                k2: 0.0,
                p1: 0.0,
                p2: 0.0,
                k3: 0.0,
            };

            for &r_d in &[1.25_f64, 1.335, 1.5, 2.0, 5.0] {
                let [xu, yu] = m.undistort(r_d, 0.0);
                let [xc, yc] = m.distort(xu, yu);
                let residual = (xc - r_d).hypot(yc);
                assert!(
                    xu.is_finite() && yu.is_finite(),
                    "undistort must stay finite at r_d = {r_d}, got ({xu}, {yu})"
                );
                assert!(
                    xu >= 0.0,
                    "undistort returned the mirrored branch at r_d = {r_d}: xu = {xu}"
                );
                assert!(
                    residual > GATE,
                    "r_d = {r_d} is past the turning point, so the round-trip residual \
                     must exceed the caller's gate ({GATE:e}) and reject the point; \
                     got {residual:.3e} for r_u = {xu}"
                );
                assert!(
                    m.undistort_checked(r_d, 0.0).is_none(),
                    "undistort_checked must report failure past the turning point \
                     (r_d = {r_d}); every geometry consumer depends on it"
                );
            }

            // Just inside the branch it must still invert exactly.
            for &r_d in &[0.2_f64, 0.8, 1.2] {
                let [xu, yu] = m.undistort(r_d, 0.0);
                let [xc, yc] = m.distort(xu, yu);
                assert!(
                    (xc - r_d).hypot(yc) < 1e-12,
                    "r_d = {r_d} is inside the invertible branch and must round-trip"
                );
                assert!(
                    m.undistort_checked(r_d, 0.0).is_some(),
                    "undistort_checked must accept r_d = {r_d}, inside the branch"
                );
            }
        }

        #[test]
        fn brown_conrady_undistort_stays_finite_at_extreme_radius() {
            let m = hub_brown_conrady();
            for &(xd, yd) in &[(5.0, 5.0), (50.0, 0.0), (0.0, 100.0), (1e8, -1e8)] {
                let out = m.undistort(xd, yd);
                assert!(
                    out[0].is_finite() && out[1].is_finite(),
                    "undistort must stay finite at extreme radius, got {out:?} for ({xd}, {yd})"
                );
            }
        }

        #[test]
        fn brown_conrady_from_coeffs_validates_length() {
            assert!(BrownConradyModel::from_coeffs(&[0.0; 4]).is_err());
            assert!(BrownConradyModel::from_coeffs(&[0.0; 5]).is_ok());
            assert!(BrownConradyModel::from_coeffs(&[0.0; 6]).is_err());
        }

        #[test]
        fn kannala_brandt_from_coeffs_validates_length() {
            assert!(KannalaBrandtModel::from_coeffs(&[0.0; 3]).is_err());
            assert!(KannalaBrandtModel::from_coeffs(&[0.0; 4]).is_ok());
            assert!(KannalaBrandtModel::from_coeffs(&[0.0; 5]).is_err());
        }
    }
}
