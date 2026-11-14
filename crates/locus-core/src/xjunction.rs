//! Model fit of an X-junction: the corner where a marker's black border touches a black
//! connector square diagonally (AprilGrid boards).
//!
//! Near the junction the pattern is two straight lines crossing at `c`, dark where the signed
//! distances to both lines have the same sign. The image is modelled as
//!
//! ```text
//! μ(p) = A + B · erf(n₁·(p − c) / s) · erf(n₂·(p − c) / s)
//! ```
//!
//! with line normals `nᵢ = (−sin tᵢ, cos tᵢ)`, a blur scale `s` (`√2·σ` of a Gaussian edge) and
//! the two levels `A ± B`. The seven parameters are fitted to the pixels of a window around the
//! junction by weighted Levenberg–Marquardt. The pixel values are used as they are, so no
//! interpolation phase enters the estimate.
//!
//! **The pixel's aperture.** A pixel integrates the light over its square; it does not sample
//! the image at its centre. Where an edge runs near a pixel axis, the box blurs it far from the
//! Gaussian that `s` can absorb, and the fitted junction moves by about a hundredth of a pixel,
//! three at a 0.2 px PSF. The fit therefore converges with the model sampled at each pixel's
//! centre, which is cheap, and is then polished by one Gauss–Newton step of the model averaged
//! over the pixel ([`Junction::polish`]): each line's profile exactly ([`ApertureLine`]), and
//! the covariance of the two profiles over the pixel to first order
//! ([`ApertureModel`]).
//!
//! **Lens curvature.** The model's lines are straight, and a lens images a marker's sides as
//! arcs. Across the window a side of curvature `κ` leaves its tangent by `½κτ²`, which moves
//! the fitted junction by about `κR²/6` for a window of reach `R`: 0.02 px at `κ = 0.005 /px`
//! and `R = 4.75 px`, several times the Cramér–Rao bound of a 20 px tag. The polish bends each
//! line by the curvature that the marker's four fits measure themselves ([`side_curvatures`]).
//! Fitting a curvature per line instead was measured and dropped: it trades off with `c`, and
//! the median error rose 60 %.
//!
//! **Measured.** AprilGrids rendered through a 195° fisheye with exact image formation (each
//! pixel the box ⊗ Gaussian-PSF integral of the board, with sensor noise), as the median
//! corner error over the Cramér–Rao bound, for the centre-sampled fit, the polish with
//! straight lines, and the polish:
//!
//! | 24 frames per case | centre | aperture | aperture and curvature |
//! |---|---|---|---|
//! | 20 px tags at the image centre | 1.80 | 1.47 | 1.20 |
//! | 0.2 px PSF | 4.81 | 1.73 | 1.51 |
//! | 12 px tags | 1.71 | 1.16 | 1.17 |
//! | 32 px tags | 3.51 | 3.23 | 1.73 |
//! | 50° incidence | 1.93 | 1.64 | 1.32 |
//! | 40° off axis | 2.79 | 2.56 | 1.52 |
//! | 70° off axis | 4.12 | 4.02 | 1.86 |
//! | without noise | 1.53 | 1.07 | 0.71 |
//! | geometric mean of 13 cases | 2.30 | 1.79 | 1.32 |
//!
//! Recall is the same in every case, since the fit that the gates judge is unchanged. On
//! pinhole renders (the hub AprilGrid, 5 128 tags) the median corner error falls from 0.0085
//! to 0.0079 px and its 95th percentile from 0.063 to 0.046 px, all of it from the aperture:
//! there is no curvature to find, and looking for it moves the median by 0.0001 px. A
//! quadrature of the model over the pixel (3 × 3 Gauss–Legendre points) reaches 2.11 at the
//! 0.2 px PSF, where its points are farther apart than the blur, at 18 `erf` per pixel
//! against 8. The earlier finding that box-integration changes nothing came from renders whose
//! blur was a Gaussian sampled at whole pixels, which is not a box aperture.
//!
//! **Why it is unbiased without the true blur.** The product of two odd functions is point
//! symmetric about `c`, and so is the image of any X-junction under any point-symmetric PSF and
//! any per-pixel tone curve. The least-squares cost is then even in the corner offset, so its
//! minimum stays at the junction however wrong the edge profile is. The product form ignores the
//! correlation the blur introduces between the two lines when they are not orthogonal; it is
//! point symmetric too, and measured equal to the exact bivariate-normal model on rendered
//! AprilGrids at up to 70° incidence. A window the frame clips is not point symmetric, so there
//! a wrong edge profile can bias the fit; it is still consistent, and beats the seed.

use crate::image::ImageView;
use crate::simd::math::{F64x4, erf_and_gauss_x4, wrap_angle};
use nalgebra::{SMatrix, SVector};

/// Number of fitted parameters: `c` (2), `t₁`, `t₂`, `s`, `A`, `B`.
const PARAMS: usize = 7;
/// Largest half-extent (px) of the window's bounding box: bounds the stack buffers.
const MAX_EXTENT_PX: f64 = 10.0;
/// Side of the largest pixel window: the bounding box plus a pixel.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
const MAX_SIDE: usize = 2 * (MAX_EXTENT_PX as usize + 1) + 1;
/// Smallest sine of the angle between the two edges: nearly parallel edges have no junction
/// to locate.
const MIN_SKEW_SINE: f64 = 0.2;
/// Window capacity, a lane multiple: the loop reads whole lanes and the zeroed tail has weight 0.
const MAX_SAMPLES: usize = (MAX_SIDE * MAX_SIDE).next_multiple_of(LANES);
/// Distance (px) from both lines past which a pixel is left out of the window: there the
/// pattern is flat and informs only the two levels. Measured on rendered AprilGrids: 2 px is
/// as accurate as the whole window (median error 0.0226 px either way) at 15-30 % less fit
/// time; 1.5 px is no better and 1.0 px is worse (0.040), since the edge profiles reach that
/// far.
const FLAT_PX: f64 = 2.0;
/// Fewest window pixels per fitted parameter. A 12 px Kalibr tag's window holds 21-30 pixels;
/// at 4 per parameter its corners kept their seeds (rendered 12 px tags: 6.8x the bound, 1.5 %
/// of tags 0.1 px inward); at 3 they are fitted (4.6x, 0.8 %), and 2 adds nothing.
const MIN_SAMPLES_PER_PARAM: usize = 3;
/// Width (px) of the window's soft rim: weights fall from 1 to 0 over it, so a pixel entering or
/// leaving the window does not make the cost jump.
const RIM_PX: f64 = 1.0;
/// Initial blur scale: `√2 · 0.6 px`, a sharp camera.
const INITIAL_SCALE: f64 = 0.85;
/// Iteration cap, and the relative cost decrease under which an accepted step ends the fit.
const MAX_ITER: usize = 30;
const COST_TOLERANCE: f64 = 1e-4;
/// Initial Levenberg–Marquardt damping, relative to the normal matrix's diagonal.
const INITIAL_DAMPING: f64 = 1e-3;
/// Narrowest box of a pixel's projection on a line's normal, relative to the wider one
/// ([`ApertureLine`]). The exact average divides by the product of the two widths, which
/// vanishes for an edge along a pixel axis. The narrow box enters at second order, so holding
/// it at 1 % of the wide one changes the model by under 10⁻⁴ of the contrast.
const MIN_APERTURE_RATIO: f64 = 0.01;
/// What the centre-sampled fit's `s²` holds of the pixel's box, px². Along any direction the
/// box has variance `(n.x² + n.y²)/12 = 1/12`, and an edge profile `erf(d/s)` is a Gaussian
/// blur of variance `s²/2`.
const APERTURE_SCALE_SQ: f64 = 1.0 / 6.0;
/// Rejections of a fit: largest move from the start (px; or half the window's shorter reach,
/// if larger: a seed under the cell floor sits up to ~1.1 px off, and the window still holds
/// the junction), largest turn of either line (rad), the blur scale's range (px), and the
/// largest weighted residual RMS over the half contrast `|B|` (a model that does not describe
/// the window, such as an L-corner).
const MAX_MOVE_PX: f64 = 1.0;
const MAX_TURN_RAD: f64 = 0.35;
const MIN_SCALE_PX: f64 = 0.2;
const MAX_RELATIVE_RMS: f64 = 0.35;

/// A refused fit that landed farther than this (px) from its window's centre gets one refit
/// over a window centred where it landed (`fit_x_junction`).
const RECENTRE_PX: f64 = 0.5;

/// The window: each pixel's centre relative to the window centre, its value and its weight.
///
/// One window serves every fit and polish of a marker ([`Window::empty`]), each of which
/// refills it: clearing its 17 KB per fit took as long as half a pass of the kernel over it.
pub(crate) struct Window {
    dx: [f64; MAX_SAMPLES],
    dy: [f64; MAX_SAMPLES],
    value: [f64; MAX_SAMPLES],
    weight: [f64; MAX_SAMPLES],
    len: usize,
    /// Shorter reach (px) after any shrink to the buffers: what the window actually holds.
    min_reach: f64,
}

impl Window {
    /// A window holding no pixel yet.
    pub(crate) fn empty() -> Self {
        Self {
            dx: [0.0; MAX_SAMPLES],
            dy: [0.0; MAX_SAMPLES],
            value: [0.0; MAX_SAMPLES],
            weight: [0.0; MAX_SAMPLES],
            len: 0,
            min_reach: 0.0,
        }
    }

    /// Pixels whose centres lie in the ellipse `{c + α·u + β·v : (α/a)² + (β/b)² ≤ 1}` for unit
    /// edge directions `axes = [u, v]` and `reach = [a, b]` px: the image of a disk on the
    /// marker's plane, so a foreshortened border gets a window as narrow as the border. Pixels
    /// farther than `FLAT_PX` from both lines are left out. `false`, leaving the window
    /// unusable, when the edges are nearly parallel or the window lies outside the image.
    fn gather(
        &mut self,
        img: &ImageView,
        centre: [f64; 2],
        axes: [[f64; 2]; 2],
        reach: [f64; 2],
    ) -> bool {
        self.len = 0;
        let [along_u, along_v] = axes;
        let skew = along_u[0] * along_v[1] - along_u[1] * along_v[0];
        if skew.abs() < MIN_SKEW_SINE {
            return false;
        }
        // Bounding-box half-extents; a window wider than the buffers shrinks, keeping its shape.
        let half_x = (reach[0] * along_u[0]).hypot(reach[1] * along_v[0]);
        let half_y = (reach[0] * along_u[1]).hypot(reach[1] * along_v[1]);
        let shrink = (MAX_EXTENT_PX / half_x.max(half_y)).min(1.0);
        // Distances to the lines over these give the normalised elliptic radius (α/a, β/b).
        let inv_reach = reach.map(|r| 1.0 / (r * shrink * skew));
        let (x0, x1) = (
            (centre[0] - half_x * shrink).floor(),
            (centre[0] + half_x * shrink).ceil(),
        );
        let (y0, y1) = (
            (centre[1] - half_y * shrink).floor(),
            (centre[1] + half_y * shrink).ceil(),
        );
        // A window the image clips keeps the pixels it has. The fit needs no symmetric window
        // to be consistent, only for its robustness to a wrong edge profile, and the corners
        // it serves (within a few pixels of the frame, at the fisheye rim) otherwise keep
        // seeds 0.4-2.8 px off.
        // `clamp`, unlike `max`/`min`, keeps a NaN bound (a degenerate seed or reach), which the
        // ordering test then refuses instead of scanning the whole image.
        let (x0, y0) = (x0.clamp(0.0, f64::INFINITY), y0.clamp(0.0, f64::INFINITY));
        let (x1, y1) = (
            x1.clamp(f64::NEG_INFINITY, img.width as f64 - 1.0),
            y1.clamp(f64::NEG_INFINITY, img.height as f64 - 1.0),
        );
        if !(x0 <= x1 && y0 <= y1) {
            return false;
        }
        // Non-negative, inside the image and ordered: clamped and checked just above.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let (x0, x1, y0, y1) = (x0 as usize, x1 as usize, y0 as usize, y1 as usize);
        self.min_reach = reach[0].min(reach[1]) * shrink;
        // Rim width in units of the normalised radius, along the shorter axis.
        let rim = self.min_reach / RIM_PX;
        for y in y0..=y1 {
            let row = &img.data[y * img.stride + x0..=y * img.stride + x1];
            let dy = y as f64 + 0.5 - centre[1];
            for (x, &value) in (x0..).zip(row) {
                let dx = x as f64 + 0.5 - centre[0];
                // Signed distances to the lines along v and along u.
                let from_v = dx * along_v[1] - dy * along_v[0];
                let from_u = along_u[0] * dy - along_u[1] * dx;
                // Flat there: such a pixel informs only the levels, which the rest already fix.
                if from_u.abs().min(from_v.abs()) > FLAT_PX || self.len == MAX_SAMPLES {
                    continue;
                }
                let (along_a, along_b) = (from_v * inv_reach[0], from_u * inv_reach[1]);
                let radius2 = along_a * along_a + along_b * along_b;
                if radius2 >= 1.0 {
                    continue;
                }
                // sin² taper over the rim: the cost stays smooth as pixels cross it.
                let taper = (1.0 - radius2.sqrt()) * rim;
                let weight = if taper < 1.0 {
                    let rise = (0.5 * core::f64::consts::PI * taper).sin();
                    rise * rise
                } else {
                    1.0
                };
                self.dx[self.len] = dx;
                self.dy[self.len] = dy;
                self.value[self.len] = f64::from(value);
                self.weight[self.len] = weight;
                self.len += 1;
            }
        }
        // The kernels read whole lanes: the tail past `len` carries weight 0.
        for k in self.len..self.len.next_multiple_of(LANES) {
            (self.dx[k], self.dy[k], self.value[k], self.weight[k]) = (0.0, 0.0, 0.0, 0.0);
        }
        true
    }
}

/// Lanes of the window loop.
const LANES: usize = F64x4::LANES;

/// Parameter vector `[cx, cy, t₁, t₂, s, A, B]`; `c` is relative to the window centre.
type Params = SVector<f64, PARAMS>;
const CX: usize = 0;
const CY: usize = 1;
const T1: usize = 2;
const T2: usize = 3;
const SCALE: usize = 4;
const LEVEL: usize = 5;
const CONTRAST: usize = 6;

/// `JᵀWJ`, `JᵀWr` and the cost `Σ w·r²` at `params`, the model sampled at each pixel's centre.
///
/// The AVX2+FMA kernel when the CPU has it, else the portable one.
fn normal_equations(win: &Window, params: &Params) -> NormalEquations {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        #[allow(
            unsafe_code,
            reason = "a target_feature function needs its features, detected just above"
        )]
        // SAFETY: AVX2 and FMA were detected on this CPU just above, which is the only
        // requirement of the `target_feature` function.
        let fast = unsafe { avx2::normal_equations(win, params) };
        return fast;
    }
    normal_equations_portable(win, params)
}

/// [`normal_equations`] with the model averaged over each pixel's aperture and each line bent
/// by `curvature[i]` (1/px, toward its left normal): the model of [`Junction::polish`].
fn normal_equations_aperture(
    win: &Window,
    params: &Params,
    curvature: [f64; 2],
) -> NormalEquations {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        #[allow(
            unsafe_code,
            reason = "a target_feature function needs its features, detected just above"
        )]
        // SAFETY: AVX2 and FMA were detected on this CPU just above, which is the only
        // requirement of the `target_feature` function.
        let fast = unsafe { avx2::normal_equations_aperture(win, params, curvature) };
        return fast;
    }
    normal_equations_aperture_portable(win, params, curvature)
}

/// `JᵀWJ`, `JᵀWr` and the cost.
type NormalEquations = (SMatrix<f64, PARAMS, PARAMS>, SVector<f64, PARAMS>, f64);

/// `1/√π`.
const INV_SQRT_PI: f64 = 0.5 * core::f64::consts::FRAC_2_SQRT_PI;

/// One line of the aperture model at a parameter vector: what is constant over the window.
///
/// Along the line's normal `n`, the unit pixel projects to the convolution of two boxes of
/// widths `|n.x|` and `|n.y|` (a trapezoid). The edge profile averaged over the pixel is `erf`
/// convolved with both, which has a closed form in the second antiderivative of `erf`,
/// `F₂(x) = ½·((x² + ½)·erf x + x·e^(−x²)/√π)`:
///
/// ```text
/// E(h) = (F₂(h+a+b) − F₂(h+a−b) − F₂(h−a+b) + F₂(h−a−b)) / (4ab)
/// ```
///
/// with half-widths `a = |n.x|/2s` and `b = |n.y|/2s`. It is exact at any blur, where a
/// quadrature over the pixel is not once `s` is under the spacing of its points.
#[derive(Clone, Copy)]
struct ApertureLine {
    /// Unit normal `(−sin t, cos t)`.
    normal: [f64; 2],
    /// `[a, b]`, and their derivatives in `t`.
    half: [f64; 2],
    half_dt: [f64; 2],
    /// Half the line's curvature (1/px): its distance from a pixel at `e` is
    /// `n·e − ½κ·(u·e)²`, with `u` the line's direction.
    half_curvature: f64,
}

impl ApertureLine {
    fn new(angle: f64, inv_scale: f64, curvature: f64) -> Self {
        let normal = normal(angle);
        let width = [normal[0].abs(), normal[1].abs()];
        // d|n.x|/dt = −sign(n.x)·n.y and d|n.y|/dt = sign(n.y)·n.x.
        let width_dt = [
            -normal[0].signum() * normal[1],
            normal[1].signum() * normal[0],
        ];
        let half = 0.5 * inv_scale;
        let side = |axis: usize| {
            let other = 1 - axis;
            if width[axis] >= MIN_APERTURE_RATIO * width[other] {
                (half * width[axis], half * width_dt[axis])
            } else {
                (
                    half * MIN_APERTURE_RATIO * width[other],
                    half * MIN_APERTURE_RATIO * width_dt[other],
                )
            }
        };
        let ((a, a_dt), (b, b_dt)) = (side(0), side(1));
        Self {
            normal,
            half: [a, b],
            half_dt: [a_dt, b_dt],
            half_curvature: 0.5 * curvature,
        }
    }
}

/// A line's averaged profile `E` at four-lane `h`, and its derivatives: `eh`, `ehh` in `h`;
/// `ea`, `eb` in the half-widths; `eha`, `ehb` those of `eh`.
struct Profile<V> {
    e: V,
    eh: V,
    ehh: V,
    ea: V,
    eb: V,
    eha: V,
    ehb: V,
}

/// The aperture model at a parameter vector: `A + B·(E₁·E₂ + ρ·E₁′·E₂′)` of the two lines'
/// averaged profiles ([`ApertureLine`]).
///
/// `ρ = n₁·n₂ / 12s²` is the covariance of the two lines' coordinates over a pixel. The pixel
/// average of `erf(h₁)·erf(h₂)` is the product of the averaged profiles plus their covariance
/// over the pixel, which to first order is `ρ·E₁′·E₂′`. It vanishes for orthogonal lines;
/// without it a junction seen at 50° incidence is fitted worse than by a quadrature over the
/// pixel (rendered AprilGrids: 1.78 over the Cramér–Rao bound, against 1.71 with it and 1.70
/// for the quadrature).
struct ApertureModel {
    lines: [ApertureLine; 2],
    /// `ρ`, and its derivatives in `t₁` and `t₂`.
    covariance: f64,
    covariance_dt: [f64; 2],
    inv_scale: f64,
    contrast: f64,
}

impl ApertureModel {
    fn new(params: &Params, curvature: [f64; 2]) -> Self {
        let inv_scale = 1.0 / params[SCALE];
        let lines = [
            ApertureLine::new(params[T1], inv_scale, curvature[0]),
            ApertureLine::new(params[T2], inv_scale, curvature[1]),
        ];
        let [n1, n2] = [lines[0].normal, lines[1].normal];
        let twelfth = inv_scale * inv_scale / 12.0;
        Self {
            lines,
            covariance: twelfth * (n1[0] * n2[0] + n1[1] * n2[1]),
            // d n/d t = (−n.y, n.x).
            covariance_dt: [
                twelfth * (n1[0] * n2[1] - n1[1] * n2[0]),
                twelfth * (n1[1] * n2[0] - n1[0] * n2[1]),
            ],
            inv_scale,
            contrast: params[CONTRAST],
        }
    }
}

/// Portable kernel: branch-free over `LANES` pixels at a time ([`F64x4`]); the window is
/// zero-padded to a lane multiple with weight 0. The reference for the AVX2 kernel.
fn normal_equations_portable(win: &Window, params: &Params) -> NormalEquations {
    const TWO_OVER_SQRT_PI: f64 = core::f64::consts::FRAC_2_SQRT_PI;
    let (n1, n2) = (normal(params[T1]), normal(params[T2]));
    // d nᵢ/d tᵢ = −uᵢ.
    let (dn1, dn2) = ([-n1[1], n1[0]], [-n2[1], n2[0]]);
    let inv_scale = 1.0 / params[SCALE];
    let (level, contrast) = (params[LEVEL], params[CONTRAST]);
    let slope = contrast * TWO_OVER_SQRT_PI * inv_scale;
    let (cx, cy) = (params[CX], params[CY]);

    let padded = win.len.next_multiple_of(LANES);
    let (dx, dy, value, weight) = (
        &win.dx[..padded],
        &win.dy[..padded],
        &win.value[..padded],
        &win.weight[..padded],
    );
    let splat = F64x4::splat;
    // One 4-lane partial sum per entry: the lanes vectorize, and no single sum is reassociated.
    let mut tri = [splat(0.0); PARAMS * (PARAMS + 1) / 2];
    let mut grad = [splat(0.0); PARAMS];
    let mut cost = splat(0.0);
    for base in (0..padded).step_by(LANES) {
        let lanes = base..base + LANES;
        let ex = F64x4::load(&dx[lanes.clone()]) - splat(cx);
        let ey = F64x4::load(&dy[lanes.clone()]) - splat(cy);
        let h1 = (splat(n1[0]) * ex + splat(n1[1]) * ey) * splat(inv_scale);
        let h2 = (splat(n2[0]) * ex + splat(n2[1]) * ey) * splat(inv_scale);
        let (erf1, gauss1) = erf_and_gauss_x4(h1);
        let (erf2, gauss2) = erf_and_gauss_x4(h2);
        let product = erf1 * erf2;
        let res = F64x4::load(&value[lanes.clone()]) - (splat(level) + splat(contrast) * product);
        let w = F64x4::load(&weight[lanes]);
        // ∂μ/∂(distance to each line).
        let d1 = splat(slope) * gauss1 * erf2;
        let d2 = splat(slope) * gauss2 * erf1;
        let jac = [
            splat(0.0) - (d1 * splat(n1[0]) + d2 * splat(n2[0])),
            splat(0.0) - (d1 * splat(n1[1]) + d2 * splat(n2[1])),
            d1 * (splat(dn1[0]) * ex + splat(dn1[1]) * ey),
            d2 * (splat(dn2[0]) * ex + splat(dn2[1]) * ey),
            splat(0.0) - (d1 * h1 + d2 * h2),
            splat(1.0),
            product,
        ];
        let w_res = w * res;
        cost = cost + w_res * res;
        let mut at = 0;
        for row in 0..PARAMS {
            grad[row] = grad[row] + jac[row] * w_res;
            let w_row = w * jac[row];
            for &col in &jac[..=row] {
                tri[at] = tri[at] + w_row * col;
                at += 1;
            }
        }
    }
    assemble(tri.map(F64x4::sum), grad.map(F64x4::sum), cost.sum())
}

/// Entries of the packed lower triangle of `JᵀWJ`, row by row.
const TRIANGLE: usize = PARAMS * (PARAMS + 1) / 2;

/// [`Profile`] of `line` at the line coordinate `coord`, portable.
fn profile_x4(coord: F64x4, line: &ApertureLine) -> Profile<F64x4> {
    let splat = F64x4::splat;
    let [half_x, half_y] = line.half;
    let (sum, diff) = (splat(half_x + half_y), splat(half_x - half_y));
    let points = [coord + sum, coord + diff, coord - diff, coord - sum];
    // erf, its antiderivative F₁ = x·erf + e^(−x²)/√π, and x·F₁ (F₂ = ½·(x·F₁ + ½·erf)).
    let mut erf = [splat(0.0); 4];
    let mut first = [splat(0.0); 4];
    let mut second = [splat(0.0); 4];
    for k in 0..4 {
        let (value, gauss) = erf_and_gauss_x4(points[k]);
        erf[k] = value;
        first[k] = points[k] * value + gauss * splat(INV_SQRT_PI);
        second[k] = points[k] * first[k];
    }
    // The second difference over both boxes, and its derivatives in each half-width.
    let both = |v: &[F64x4; 4]| (v[0] - v[1]) - (v[2] - v[3]);
    let through_x = |v: &[F64x4; 4]| (v[0] - v[1]) + (v[2] - v[3]);
    let through_y = |v: &[F64x4; 4]| (v[0] + v[1]) - (v[2] + v[3]);
    let inv_area = splat(0.25 / (half_x * half_y));
    let (per_x, per_y) = (splat(1.0 / half_x), splat(1.0 / half_y));
    let e = inv_area * splat(0.5) * (both(&second) + splat(0.5) * both(&erf));
    let eh = inv_area * both(&first);
    Profile {
        e,
        eh,
        ehh: inv_area * both(&erf),
        ea: inv_area * through_x(&first) - e * per_x,
        eb: inv_area * through_y(&first) - e * per_y,
        eha: inv_area * through_x(&erf) - eh * per_x,
        ehb: inv_area * through_y(&erf) - eh * per_y,
    }
}

/// The aperture model's Jacobian at four pixels offset `(ex, ey)` from the junction, in
/// parameter order; its `CONTRAST` entry is the model's `E₁·E₂ + ρ·E₁′·E₂′` itself. Portable.
fn aperture_jacobian_x4(model: &ApertureModel, ex: F64x4, ey: F64x4) -> [F64x4; PARAMS] {
    let splat = F64x4::splat;
    let inv_scale = splat(model.inv_scale);
    // Per line: distance to the straight line, coordinate along it, and `h`.
    let geometry = |line: &ApertureLine| {
        let [nx, ny] = line.normal;
        let dist = splat(nx) * ex + splat(ny) * ey;
        let along = splat(ny) * ex - splat(nx) * ey;
        let coord = (dist - splat(line.half_curvature) * along * along) * inv_scale;
        (dist, along, coord)
    };
    let [first, second] = &model.lines;
    let (dist1, along1, h1) = geometry(first);
    let (dist2, along2, h2) = geometry(second);
    let (p1, p2) = (profile_x4(h1, first), profile_x4(h2, second));
    let rho = splat(model.covariance);
    let slopes = p1.eh * p2.eh;
    // ∂(product)/∂hᵢ, and its derivatives through each line's two half-widths.
    let d1 = p1.eh * p2.e + rho * p1.ehh * p2.eh;
    let d2 = p1.e * p2.eh + rho * p1.eh * p2.ehh;
    let (a1, b1) = (
        p1.ea * p2.e + rho * p1.eha * p2.eh,
        p1.eb * p2.e + rho * p1.ehb * p2.eh,
    );
    let (a2, b2) = (
        p1.e * p2.ea + rho * p1.eh * p2.eha,
        p1.e * p2.eb + rho * p1.eh * p2.ehb,
    );
    // s·∂hᵢ/∂c = −nᵢ + κᵢ·(uᵢ·e)·uᵢ with uᵢ = (nᵢ.y, −nᵢ.x), and
    // s·∂hᵢ/∂tᵢ = −(uᵢ·e)·(1 + κᵢ·nᵢ·e).
    let bend1 = splat(2.0 * first.half_curvature) * along1;
    let bend2 = splat(2.0 * second.half_curvature) * along2;
    let [n1, n2] = [first.normal, second.normal];
    let dc = |axis: usize| {
        let (u1, u2) = if axis == 0 {
            (n1[1], n2[1])
        } else {
            (-n1[0], -n2[0])
        };
        d1 * (bend1 * splat(u1) - splat(n1[axis])) + d2 * (bend2 * splat(u2) - splat(n2[axis]))
    };
    let turn = |line: &ApertureLine, slope: F64x4, along: F64x4, dist: F64x4| {
        splat(0.0) - slope * along * (splat(1.0) + splat(2.0 * line.half_curvature) * dist)
    };
    let contrast = splat(model.contrast);
    let dt = |line: &ApertureLine, turned: F64x4, wide: F64x4, narrow: F64x4, rho_dt: f64| {
        contrast
            * (inv_scale * turned
                + splat(line.half_dt[0]) * wide
                + splat(line.half_dt[1]) * narrow
                + splat(rho_dt) * slopes)
    };
    let through_scale = h1 * d1
        + h2 * d2
        + splat(first.half[0]) * a1
        + splat(first.half[1]) * b1
        + splat(second.half[0]) * a2
        + splat(second.half[1]) * b2
        + splat(2.0 * model.covariance) * slopes;
    let per_scale = contrast * inv_scale;
    [
        per_scale * dc(0),
        per_scale * dc(1),
        dt(
            first,
            turn(first, d1, along1, dist1),
            a1,
            b1,
            model.covariance_dt[0],
        ),
        dt(
            second,
            turn(second, d2, along2, dist2),
            a2,
            b2,
            model.covariance_dt[1],
        ),
        splat(0.0) - per_scale * through_scale,
        splat(1.0),
        p1.e * p2.e + rho * slopes,
    ]
}

/// [`normal_equations_portable`] for the [`ApertureModel`] along bent lines. The reference for
/// the AVX2 aperture kernel.
fn normal_equations_aperture_portable(
    win: &Window,
    params: &Params,
    curvature: [f64; 2],
) -> NormalEquations {
    let model = ApertureModel::new(params, curvature);
    let (level, contrast) = (params[LEVEL], params[CONTRAST]);
    let (cx, cy) = (params[CX], params[CY]);

    let padded = win.len.next_multiple_of(LANES);
    let (dx, dy, value, weight) = (
        &win.dx[..padded],
        &win.dy[..padded],
        &win.value[..padded],
        &win.weight[..padded],
    );
    let splat = F64x4::splat;
    // One 4-lane partial sum per entry, as in the centre kernel.
    let mut tri = [splat(0.0); TRIANGLE];
    let mut grad = [splat(0.0); PARAMS];
    let mut cost = splat(0.0);
    for base in (0..padded).step_by(LANES) {
        let span = base..base + LANES;
        let ex = F64x4::load(&dx[span.clone()]) - splat(cx);
        let ey = F64x4::load(&dy[span.clone()]) - splat(cy);
        let jac = aperture_jacobian_x4(&model, ex, ey);
        let res =
            F64x4::load(&value[span.clone()]) - (splat(level) + splat(contrast) * jac[CONTRAST]);
        let w = F64x4::load(&weight[span]);
        let w_res = w * res;
        cost = cost + w_res * res;
        let mut at = 0;
        for row in 0..PARAMS {
            grad[row] = grad[row] + jac[row] * w_res;
            let w_row = w * jac[row];
            for &col in &jac[..=row] {
                tri[at] = tri[at] + w_row * col;
                at += 1;
            }
        }
    }
    assemble(tri.map(F64x4::sum), grad.map(F64x4::sum), cost.sum())
}

/// The symmetric `JᵀWJ` from its packed lower triangle, with `JᵀWr` and the cost.
fn assemble(tri: [f64; TRIANGLE], grad: [f64; PARAMS], cost: f64) -> NormalEquations {
    let mut jtj = SMatrix::<f64, PARAMS, PARAMS>::zeros();
    let mut at = 0;
    for row in 0..PARAMS {
        for col in 0..=row {
            jtj[(row, col)] = tri[at];
            jtj[(col, row)] = tri[at];
            at += 1;
        }
    }
    (jtj, SVector::from(grad), cost)
}

/// The kernels on AVX2+FMA: [`normal_equations_portable`] and
/// [`normal_equations_aperture_portable`] with each [`F64x4`] in one register. LLVM leaves the
/// portable versions scalar (measured: no `ymm` arithmetic in their AVX2 clone), and these
/// kernels are most of the fit's time.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{
        ApertureLine, ApertureModel, CONTRAST, CX, CY, INV_SQRT_PI, LANES, LEVEL, NormalEquations,
        PARAMS, Params, Profile, SCALE, T1, T2, TRIANGLE, Window, assemble, normal,
    };
    use crate::simd::math::avx2::{erf_and_gauss, sum};
    use core::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_fnmadd_pd,
        _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_sub_pd,
    };

    /// [`super::normal_equations_portable`] on AVX2+FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) fn normal_equations(win: &Window, params: &Params) -> NormalEquations {
        const TWO_OVER_SQRT_PI: f64 = core::f64::consts::FRAC_2_SQRT_PI;
        let (n1, n2) = (normal(params[T1]), normal(params[T2]));
        let inv_scale = 1.0 / params[SCALE];
        let splat = _mm256_set1_pd;
        let (n1x, n1y, n2x, n2y) = (splat(n1[0]), splat(n1[1]), splat(n2[0]), splat(n2[1]));
        // −nᵢ, by line then axis.
        let minus = [
            [splat(-n1[0]), splat(-n1[1])],
            [splat(-n2[0]), splat(-n2[1])],
        ];
        let (level, contrast) = (splat(params[LEVEL]), splat(params[CONTRAST]));
        let slope = splat(params[CONTRAST] * TWO_OVER_SQRT_PI * inv_scale);
        let (cx, cy, inv_s) = (splat(params[CX]), splat(params[CY]), splat(inv_scale));
        let mut tri = [_mm256_setzero_pd(); TRIANGLE];
        let mut grad = [_mm256_setzero_pd(); PARAMS];
        let mut cost = _mm256_setzero_pd();
        let padded = win.len.next_multiple_of(LANES);
        for base in (0..padded).step_by(LANES) {
            let load = |s: &[f64; super::MAX_SAMPLES]| {
                let lanes = &s[base..base + LANES];
                #[allow(unsafe_code, reason = "std::arch loads take a raw pointer")]
                // SAFETY: `lanes` is a bounds-checked slice of exactly four `f64`, which is
                // what the unaligned load reads.
                unsafe {
                    _mm256_loadu_pd(lanes.as_ptr())
                }
            };
            let ex = _mm256_sub_pd(load(&win.dx), cx);
            let ey = _mm256_sub_pd(load(&win.dy), cy);
            let h1 = _mm256_mul_pd(_mm256_fmadd_pd(n1x, ex, _mm256_mul_pd(n1y, ey)), inv_s);
            let h2 = _mm256_mul_pd(_mm256_fmadd_pd(n2x, ex, _mm256_mul_pd(n2y, ey)), inv_s);
            let (erf1, gauss1) = erf_and_gauss(h1);
            let (erf2, gauss2) = erf_and_gauss(h2);
            let product = _mm256_mul_pd(erf1, erf2);
            let res = _mm256_sub_pd(load(&win.value), _mm256_fmadd_pd(contrast, product, level));
            let w = load(&win.weight);
            // ∂μ/∂(distance to each line).
            let d1 = _mm256_mul_pd(_mm256_mul_pd(slope, gauss1), erf2);
            let d2 = _mm256_mul_pd(_mm256_mul_pd(slope, gauss2), erf1);
            // d nᵢ/d tᵢ = −uᵢ = (−nᵢ.y, nᵢ.x).
            let along1 = _mm256_fmadd_pd(n1x, ey, _mm256_mul_pd(minus[0][1], ex));
            let along2 = _mm256_fmadd_pd(n2x, ey, _mm256_mul_pd(minus[1][1], ex));
            let jac = [
                _mm256_fmadd_pd(d1, minus[0][0], _mm256_mul_pd(d2, minus[1][0])),
                _mm256_fmadd_pd(d1, minus[0][1], _mm256_mul_pd(d2, minus[1][1])),
                _mm256_mul_pd(d1, along1),
                _mm256_mul_pd(d2, along2),
                _mm256_sub_pd(
                    _mm256_setzero_pd(),
                    _mm256_fmadd_pd(d1, h1, _mm256_mul_pd(d2, h2)),
                ),
                splat(1.0),
                product,
            ];
            let w_res = _mm256_mul_pd(w, res);
            cost = _mm256_fmadd_pd(w_res, res, cost);
            let mut at = 0;
            for row in 0..PARAMS {
                grad[row] = _mm256_fmadd_pd(jac[row], w_res, grad[row]);
                let w_row = _mm256_mul_pd(w, jac[row]);
                for col in jac.iter().take(row + 1) {
                    tri[at] = _mm256_fmadd_pd(w_row, *col, tri[at]);
                    at += 1;
                }
            }
        }
        assemble(tri.map(|v| sum(v)), grad.map(|v| sum(v)), sum(cost))
    }

    /// [`super::profile_x4`] on AVX2+FMA.
    #[target_feature(enable = "avx2,fma")]
    fn profile(coord: __m256d, line: &ApertureLine) -> Profile<__m256d> {
        let splat = _mm256_set1_pd;
        let [half_x, half_y] = line.half;
        let (sum, diff) = (splat(half_x + half_y), splat(half_x - half_y));
        let points = [
            _mm256_add_pd(coord, sum),
            _mm256_add_pd(coord, diff),
            _mm256_sub_pd(coord, diff),
            _mm256_sub_pd(coord, sum),
        ];
        let mut erf = [_mm256_setzero_pd(); 4];
        let mut first = [_mm256_setzero_pd(); 4];
        let mut second = [_mm256_setzero_pd(); 4];
        for k in 0..4 {
            let (value, gauss) = erf_and_gauss(points[k]);
            erf[k] = value;
            first[k] = _mm256_fmadd_pd(points[k], value, _mm256_mul_pd(gauss, splat(INV_SQRT_PI)));
            second[k] = _mm256_mul_pd(points[k], first[k]);
        }
        let both =
            |v: &[__m256d; 4]| _mm256_sub_pd(_mm256_sub_pd(v[0], v[1]), _mm256_sub_pd(v[2], v[3]));
        let through_x =
            |v: &[__m256d; 4]| _mm256_add_pd(_mm256_sub_pd(v[0], v[1]), _mm256_sub_pd(v[2], v[3]));
        let through_y =
            |v: &[__m256d; 4]| _mm256_sub_pd(_mm256_add_pd(v[0], v[1]), _mm256_add_pd(v[2], v[3]));
        let inv_area = splat(0.25 / (half_x * half_y));
        let (per_x, per_y) = (splat(1.0 / half_x), splat(1.0 / half_y));
        let e = _mm256_mul_pd(
            _mm256_mul_pd(inv_area, splat(0.5)),
            _mm256_fmadd_pd(splat(0.5), both(&erf), both(&second)),
        );
        let eh = _mm256_mul_pd(inv_area, both(&first));
        Profile {
            e,
            eh,
            ehh: _mm256_mul_pd(inv_area, both(&erf)),
            ea: _mm256_fnmadd_pd(e, per_x, _mm256_mul_pd(inv_area, through_x(&first))),
            eb: _mm256_fnmadd_pd(e, per_y, _mm256_mul_pd(inv_area, through_y(&first))),
            eha: _mm256_fnmadd_pd(eh, per_x, _mm256_mul_pd(inv_area, through_x(&erf))),
            ehb: _mm256_fnmadd_pd(eh, per_y, _mm256_mul_pd(inv_area, through_y(&erf))),
        }
    }

    /// [`super::aperture_jacobian_x4`] on AVX2+FMA.
    #[target_feature(enable = "avx2,fma")]
    fn aperture_jacobian(model: &ApertureModel, ex: __m256d, ey: __m256d) -> [__m256d; PARAMS] {
        let splat = _mm256_set1_pd;
        let (inv_scale, one) = (splat(model.inv_scale), splat(1.0));
        // Per line: distance to the straight line, coordinate along it, and `h`.
        let geometry = |line: &ApertureLine| {
            let [nx, ny] = line.normal;
            let dist = _mm256_fmadd_pd(splat(nx), ex, _mm256_mul_pd(splat(ny), ey));
            let along = _mm256_fnmadd_pd(splat(nx), ey, _mm256_mul_pd(splat(ny), ex));
            let bent = _mm256_fnmadd_pd(
                _mm256_mul_pd(splat(line.half_curvature), along),
                along,
                dist,
            );
            (dist, along, _mm256_mul_pd(bent, inv_scale))
        };
        let [first, second] = &model.lines;
        let (dist1, along1, h1) = geometry(first);
        let (dist2, along2, h2) = geometry(second);
        let (p1, p2) = (profile(h1, first), profile(h2, second));
        let rho = splat(model.covariance);
        let slopes = _mm256_mul_pd(p1.eh, p2.eh);
        // x·y + ρ·z·w.
        let pair = |x: __m256d, y: __m256d, z: __m256d, w: __m256d| {
            _mm256_fmadd_pd(_mm256_mul_pd(rho, z), w, _mm256_mul_pd(x, y))
        };
        let d1 = pair(p1.eh, p2.e, p1.ehh, p2.eh);
        let d2 = pair(p1.e, p2.eh, p1.eh, p2.ehh);
        let (a1, b1) = (
            pair(p1.ea, p2.e, p1.eha, p2.eh),
            pair(p1.eb, p2.e, p1.ehb, p2.eh),
        );
        let (a2, b2) = (
            pair(p1.e, p2.ea, p1.eh, p2.eha),
            pair(p1.e, p2.eb, p1.eh, p2.ehb),
        );
        let bend1 = _mm256_mul_pd(splat(2.0 * first.half_curvature), along1);
        let bend2 = _mm256_mul_pd(splat(2.0 * second.half_curvature), along2);
        let [n1, n2] = [first.normal, second.normal];
        let dc = |axis: usize| {
            let (u1, u2) = if axis == 0 {
                (n1[1], n2[1])
            } else {
                (-n1[0], -n2[0])
            };
            _mm256_fmadd_pd(
                d1,
                _mm256_fmsub_pd(bend1, splat(u1), splat(n1[axis])),
                _mm256_mul_pd(d2, _mm256_fmsub_pd(bend2, splat(u2), splat(n2[axis]))),
            )
        };
        // −slope·(u·e)·(1 + κ·n·e).
        let turn = |line: &ApertureLine, slope: __m256d, along: __m256d, dist: __m256d| {
            let lever = _mm256_fmadd_pd(splat(2.0 * line.half_curvature), dist, one);
            _mm256_sub_pd(
                _mm256_setzero_pd(),
                _mm256_mul_pd(_mm256_mul_pd(slope, along), lever),
            )
        };
        let contrast = splat(model.contrast);
        let dt =
            |line: &ApertureLine, turned: __m256d, wide: __m256d, narrow: __m256d, rho_dt: f64| {
                let shape = _mm256_fmadd_pd(
                    splat(line.half_dt[0]),
                    wide,
                    _mm256_fmadd_pd(
                        splat(line.half_dt[1]),
                        narrow,
                        _mm256_mul_pd(splat(rho_dt), slopes),
                    ),
                );
                _mm256_mul_pd(contrast, _mm256_fmadd_pd(inv_scale, turned, shape))
            };
        let mut through_scale = _mm256_fmadd_pd(h1, d1, _mm256_mul_pd(h2, d2));
        for (factor, term) in [
            (first.half[0], a1),
            (first.half[1], b1),
            (second.half[0], a2),
            (second.half[1], b2),
            (2.0 * model.covariance, slopes),
        ] {
            through_scale = _mm256_fmadd_pd(splat(factor), term, through_scale);
        }
        let per_scale = _mm256_mul_pd(contrast, inv_scale);
        [
            _mm256_mul_pd(per_scale, dc(0)),
            _mm256_mul_pd(per_scale, dc(1)),
            dt(
                first,
                turn(first, d1, along1, dist1),
                a1,
                b1,
                model.covariance_dt[0],
            ),
            dt(
                second,
                turn(second, d2, along2, dist2),
                a2,
                b2,
                model.covariance_dt[1],
            ),
            _mm256_sub_pd(_mm256_setzero_pd(), _mm256_mul_pd(per_scale, through_scale)),
            one,
            _mm256_fmadd_pd(rho, slopes, _mm256_mul_pd(p1.e, p2.e)),
        ]
    }

    /// [`super::normal_equations_aperture_portable`] on AVX2+FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) fn normal_equations_aperture(
        win: &Window,
        params: &Params,
        curvature: [f64; 2],
    ) -> NormalEquations {
        let model = ApertureModel::new(params, curvature);
        let splat = _mm256_set1_pd;
        let (level, contrast) = (splat(params[LEVEL]), splat(params[CONTRAST]));
        let (cx, cy) = (splat(params[CX]), splat(params[CY]));
        let mut tri = [_mm256_setzero_pd(); TRIANGLE];
        let mut grad = [_mm256_setzero_pd(); PARAMS];
        let mut cost = _mm256_setzero_pd();
        let padded = win.len.next_multiple_of(LANES);
        for base in (0..padded).step_by(LANES) {
            let load = |s: &[f64; super::MAX_SAMPLES]| {
                let span = &s[base..base + LANES];
                #[allow(unsafe_code, reason = "std::arch loads take a raw pointer")]
                // SAFETY: `span` is a bounds-checked slice of exactly four `f64`, which is
                // what the unaligned load reads.
                unsafe {
                    _mm256_loadu_pd(span.as_ptr())
                }
            };
            let ex = _mm256_sub_pd(load(&win.dx), cx);
            let ey = _mm256_sub_pd(load(&win.dy), cy);
            let jac = aperture_jacobian(&model, ex, ey);
            let res = _mm256_sub_pd(
                load(&win.value),
                _mm256_fmadd_pd(contrast, jac[CONTRAST], level),
            );
            let w = load(&win.weight);
            let w_res = _mm256_mul_pd(w, res);
            cost = _mm256_fmadd_pd(w_res, res, cost);
            let mut at = 0;
            for row in 0..PARAMS {
                grad[row] = _mm256_fmadd_pd(jac[row], w_res, grad[row]);
                let w_row = _mm256_mul_pd(w, jac[row]);
                for col in jac.iter().take(row + 1) {
                    tri[at] = _mm256_fmadd_pd(w_row, *col, tri[at]);
                    at += 1;
                }
            }
        }
        assemble(tri.map(|v| sum(v)), grad.map(|v| sum(v)), sum(cost))
    }
}

/// Unit normal `(−sin t, cos t)` of a line with direction angle `t`.
fn normal(t: f64) -> [f64; 2] {
    let (sin, cos) = t.sin_cos();
    [-sin, cos]
}

/// A fitted X-junction: where it is, how its lines run, and what [`Junction::polish`] needs to
/// refine it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Junction {
    /// The junction (Locus convention), from the centre-sampled fit.
    pub(crate) corner: [f64; 2],
    /// Direction angles (rad) of the two lines, along `toward[0]` and `toward[1]`: each is the
    /// line's tangent at the junction, since the window is symmetric about it.
    angles: [f64; 2],
    /// The fit's window shape and converged parameters.
    axes: [[f64; 2]; 2],
    reach: [f64; 2],
    params: Params,
}

/// The X-junction near `start` (Locus convention, pixel centres at +0.5) whose two lines run
/// along `toward[0]` and `toward[1]` (the marker's two edges from the corner, any length),
/// fitted over the window reaching `reach[i]` px along each (`Window::gather`), which `win`
/// lends its buffers to.
///
/// A fit refused only by the move limit or the residual gate, after landing more than
/// `RECENTRE_PX` from its start, is refitted once over a window centred where it landed, and
/// only that refit is judged. A window off-centre by half a pixel or more holds the junction
/// but describes it badly: its far side is mostly one wedge, so the fit fails the residual gate
/// (a split piece's seed pushed 1 px into the neck), or the move limit (foreshortened tags at
/// 50° incidence start 1.5-2 px off). A fit whose lines turned or whose wedge is bright found
/// some other structure, and is not a seed.
///
/// `None` when the fit, or its refit, is refused.
pub(crate) fn fit_x_junction(
    img: &ImageView,
    start: [f64; 2],
    toward: [[f64; 2]; 2],
    reach: [f64; 2],
    win: &mut Window,
) -> Option<Junction> {
    let (fit, verdict) = fit_once(img, start, toward, reach, win)?;
    match verdict {
        Verdict::Accepted => Some(fit),
        Verdict::OffCentre(moved) if moved > RECENTRE_PX => {
            let (refit, verdict) = fit_once(img, fit.corner, toward, reach, win)?;
            matches!(verdict, Verdict::Accepted).then_some(refit)
        },
        _ => None,
    }
}

/// How a fit's landing fared at the gates (`fit_once`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Verdict {
    Accepted,
    /// The lines, blur scale and dark wedge are this marker's junction, but the fit moved past
    /// the limit or left a residual over the gate, as a window off-centre does. Carries the move
    /// (px).
    OffCentre(f64),
    Refused,
}

/// One fit over the window at `start`: where it landed and its verdict.
///
/// The gates refuse a result that is not this marker's X-junction: it moved more than
/// `MAX_MOVE_PX` (or half the window's shorter reach, after any shrink to the buffers), a line
/// turned more than `MAX_TURN_RAD`, the blur scale left `[MIN_SCALE_PX, max(reach)]`, the
/// wedge between the edges is not the dark one, or the residual RMS exceeds
/// `MAX_RELATIVE_RMS` of `|B|`.
///
/// `None`, with no landing point to refit from, when the window lies outside the image or
/// holds too few pixels, or the fit does not converge or leaves the image.
fn fit_once(
    img: &ImageView,
    start: [f64; 2],
    toward: [[f64; 2]; 2],
    reach: [f64; 2],
    win: &mut Window,
) -> Option<(Junction, Verdict)> {
    let unit = |d: [f64; 2]| {
        let len = d[0].hypot(d[1]);
        (len > 0.0).then(|| [d[0] / len, d[1] / len])
    };
    let axes = [unit(toward[0])?, unit(toward[1])?];
    if !win.gather(img, start, axes, reach) || win.len < MIN_SAMPLES_PER_PARAM * PARAMS {
        return None;
    }
    let win = &*win;
    let angles = axes.map(|d| d[1].atan2(d[0]));
    // Levels by linear least squares at the initial geometry: with `A = B = 0` the residual is
    // the pixel value, so the kernel's (A, B) block is exactly those normal equations.
    let mut params = Params::from([0.0, 0.0, angles[0], angles[1], INITIAL_SCALE, 0.0, 0.0]);
    let (jtj, grad, _) = normal_equations(win, &params);
    let (sw, se, see) = (
        jtj[(LEVEL, LEVEL)],
        jtj[(LEVEL, CONTRAST)],
        jtj[(CONTRAST, CONTRAST)],
    );
    let det = sw * see - se * se;
    if det <= f64::EPSILON * sw * sw {
        return None;
    }
    let contrast = (sw * grad[CONTRAST] - se * grad[LEVEL]) / det;
    params[LEVEL] = (grad[LEVEL] - contrast * se) / sw;
    params[CONTRAST] = contrast;
    // The wedge between the edges is the marker's border, so it must be dark: `B` must have
    // the sign opposite to `erf·erf` there.
    let inward = [axes[0][0] + axes[1][0], axes[0][1] + axes[1][1]];
    let side = |n: [f64; 2]| (n[0] * inward[0] + n[1] * inward[1]).signum();
    let dark_sign = -side(normal(angles[0])) * side(normal(angles[1]));
    if contrast * dark_sign <= 0.0 {
        return None;
    }

    let (mut jtj, mut grad, mut cost) = normal_equations(win, &params);
    let mut lambda = INITIAL_DAMPING;
    let mut converged = false;
    for _ in 0..MAX_ITER {
        let mut damped = jtj;
        for k in 0..PARAMS {
            damped[(k, k)] += lambda * jtj[(k, k)].max(f64::MIN_POSITIVE);
        }
        let mut trial = params + damped.cholesky()?.solve(&grad);
        trial[SCALE] = trial[SCALE].max(MIN_SCALE_PX);
        // The trial's normal equations carry its cost, so an accepted step (the usual case)
        // costs one pass over the window, not two.
        let (trial_jtj, trial_grad, trial_cost) = normal_equations(win, &trial);
        if trial_cost <= cost {
            let decrease = cost - trial_cost;
            (params, jtj, grad, cost) = (trial, trial_jtj, trial_grad, trial_cost);
            lambda = (lambda / 3.0).max(1e-12);
            if decrease <= COST_TOLERANCE * cost {
                converged = true;
                break;
            }
        } else {
            lambda *= 10.0;
            // No step lowers the cost: a minimum to working precision.
            if lambda > 1e8 {
                converged = true;
                break;
            }
        }
    }
    if !converged {
        return None;
    }
    let sum_w: f64 = win.weight[..win.len].iter().sum();
    let rms = (cost / (sum_w - PARAMS as f64).max(1.0)).sqrt();
    let shaped = wrap_angle(params[T1] - angles[0]).abs() <= MAX_TURN_RAD
        && wrap_angle(params[T2] - angles[1]).abs() <= MAX_TURN_RAD
        && (MIN_SCALE_PX..=reach[0].max(reach[1])).contains(&params[SCALE])
        && params[CONTRAST] * dark_sign > 0.0;
    let described = rms <= MAX_RELATIVE_RMS * params[CONTRAST].abs();
    let corner = [start[0] + params[CX], start[1] + params[CY]];
    // A clipped window may let the fit run off the frame, where no pixel holds the junction
    // and the downstream probes read clamped edge pixels.
    let inside = inside_image(img, corner);
    // The window's own reach, not the requested one: a shrunk window cannot vouch for a
    // junction it does not hold.
    let moved = params[CX].hypot(params[CY]);
    let held = moved <= MAX_MOVE_PX.max(0.5 * win.min_reach);
    let verdict = match (shaped, described && held) {
        (true, true) => Verdict::Accepted,
        (true, false) => Verdict::OffCentre(moved),
        (false, _) => Verdict::Refused,
    };
    let junction = Junction {
        corner,
        angles: [params[T1], params[T2]],
        axes,
        reach,
        params,
    };
    inside.then_some((junction, verdict))
}

/// Whether `point` lies on the image.
fn inside_image(img: &ImageView, point: [f64; 2]) -> bool {
    (0.0..img.width as f64).contains(&point[0]) && (0.0..img.height as f64).contains(&point[1])
}

impl Junction {
    /// The junction under the aperture model with its lines bent by `curvature[i]` (1/px,
    /// toward line `i`'s left normal): one Gauss–Newton step from the centre-sampled fit.
    ///
    /// The step starts at the fit with `s²` lowered by `APERTURE_SCALE_SQ`, the part of the
    /// blur that was the pixel's box, which leaves every parameter within the linear range of
    /// its optimum. One step from there is as accurate as three verified Levenberg–Marquardt
    /// steps (hub AprilGrid: median corner error 0.0079 px either way, 95th percentile 0.0460
    /// against 0.0462), so it is taken unverified and the polish is one pass over the window.
    /// Without the lowered start one step is not enough: `s` must fall by 40 % at a 0.2 px
    /// PSF, which is not a linear change.
    ///
    /// The centre-sampled corner stands when the window cannot be gathered, the normal
    /// equations do not factor, or the step moves the corner more than `RECENTRE_PX` or off
    /// the image. The polish corrects tenths of a pixel at most (measured: 0.33 px on the
    /// smallest hub tags, 0.03 px at the 99th percentile on 20 px tags through a fisheye), and
    /// a step of half a pixel is not that correction.
    pub(crate) fn polish(
        &self,
        img: &ImageView,
        curvature: [f64; 2],
        win: &mut Window,
    ) -> [f64; 2] {
        // A window centred on the fit, not where the fit started: from a window up to a pixel
        // off-centre the same polish leaves 3 % more error (rendered AprilGrids, 1.36 against
        // 1.32 over the bound).
        if !win.gather(img, self.corner, self.axes, self.reach) {
            return self.corner;
        }
        let mut params = self.params;
        (params[CX], params[CY]) = (0.0, 0.0);
        params[SCALE] = (params[SCALE] * params[SCALE] - APERTURE_SCALE_SQ)
            .max(MIN_SCALE_PX * MIN_SCALE_PX)
            .sqrt();
        let (jtj, grad, _) = normal_equations_aperture(win, &params, curvature);
        let Some(step) = jtj.cholesky().map(|factor| factor.solve(&grad)) else {
            return self.corner;
        };
        let corner = [self.corner[0] + step[CX], self.corner[1] + step[CY]];
        // `<=` is false for a NaN step.
        if step[CX].hypot(step[CY]) <= RECENTRE_PX && inside_image(img, corner) {
            corner
        } else {
            self.corner
        }
    }
}

/// Curvature (1/px) of each side of the marker whose corners are `junctions`, each fitted with
/// `toward = [previous corner, next corner]`: side `j` runs from corner `j` to corner `j + 1`,
/// and its curvature is positive where it turns toward the left normal of that direction.
///
/// A lens images a straight side as an arc. The junction fits measure its tangent at both
/// ends (each window is symmetric about its junction, so a straight line fitted to an arc is
/// parallel to the tangent there), and the turn between the two tangents over the chord is the
/// side's mean curvature. No pixel is read for it.
///
/// Only the part that opposite sides do not share is kept: half the difference of their
/// curvatures along the contour. A lens bends two parallel sides the same way, which along the
/// contour is opposite turns. What the sides share is not the lens:
/// - ink spread staggers each line at the junction, which the straight-line model fits as a
///   scissoring of its two lines, the same at all four corners. That reads as every side
///   bowing alike, by more than the lens bends a 20 px side. On rendered 20 px tags with
///   0.4 mm of spread, the error over the Cramér–Rao bound is 1.77 with straight lines, 3.76
///   with the whole turn and 1.47 with the difference;
/// - the change of the lens's bend across the marker, which is second order in its size.
///
/// The estimate is calibrated: scaling it by 0.8 or 1.2 is worse than using it as it is
/// (32 px tags: 1.76 and 1.87 against 1.72).
fn side_curvatures(junctions: &[Junction; 4]) -> [f64; 4] {
    let raw: [f64; 4] = core::array::from_fn(|j| {
        let (from, to) = (&junctions[j], &junctions[(j + 1) % 4]);
        let chord = (to.corner[0] - from.corner[0]).hypot(to.corner[1] - from.corner[1]);
        // The side leaves corner `j` along its line 1 and arrives against the next corner's
        // line 0, which points back.
        wrap_angle(to.angles[0] + core::f64::consts::PI - from.angles[1]) / chord
    });
    core::array::from_fn(|j| 0.5 * (raw[j] - raw[(j + 2) % 4]))
}

/// The four corners of a marker from its `junctions` (each fitted with
/// `toward = [previous corner, next corner]`), polished over the pixel's aperture along sides
/// bent as the junctions' own tangents measure ([`side_curvatures`], [`Junction::polish`]).
pub(crate) fn polish_marker(
    img: &ImageView,
    junctions: &[Junction; 4],
    win: &mut Window,
) -> [[f64; 2]; 4] {
    let curvature = side_curvatures(junctions);
    core::array::from_fn(|j| {
        // Line 0 runs back along the previous side, so that side's turn is reversed on it.
        junctions[j].polish(img, [-curvature[(j + 3) % 4], curvature[j]], win)
    })
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;

    /// An X-junction at `c` with line directions `t1`, `t2`, dark (40) where both signed
    /// distances share a sign and bright (200) elsewhere, box-integrated over each pixel
    /// (8 × 8 samples), blurred by a 3-tap kernel.
    fn render(c: [f64; 2], t1: f64, t2: f64, side: usize) -> Vec<u8> {
        let (n1, n2) = (normal(t1), normal(t2));
        let mut irr = vec![0.0f64; side * side];
        for y in 0..side {
            for x in 0..side {
                let mut acc = 0.0;
                for sy in 0..8u8 {
                    for sx in 0..8u8 {
                        let px = x as f64 + (f64::from(sx) + 0.5) / 8.0 - c[0];
                        let py = y as f64 + (f64::from(sy) + 0.5) / 8.0 - c[1];
                        let a = n1[0] * px + n1[1] * py;
                        let b = n2[0] * px + n2[1] * py;
                        acc += if (a > 0.0) == (b > 0.0) { 40.0 } else { 200.0 };
                    }
                }
                irr[y * side + x] = acc / 64.0;
            }
        }
        let k = [0.2, 0.6, 0.2];
        let mut tmp = irr.clone();
        for y in 0..side {
            for x in 1..side - 1 {
                tmp[y * side + x] = (0..3).map(|d| k[d] * irr[y * side + x + d - 1]).sum();
            }
        }
        let mut out = vec![0u8; side * side];
        for y in 1..side - 1 {
            for x in 0..side {
                let v: f64 = (0..3).map(|d| k[d] * tmp[(y + d - 1) * side + x]).sum();
                out[y * side + x] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    }

    #[test]
    fn recovers_a_skewed_junction_from_an_offset_start() {
        let side = 32;
        let c = [16.27, 15.61];
        let (t1, t2) = (0.3, 0.3 + 1.9);
        let data = render(c, t1, t2, side);
        let img = ImageView::new(&data, side, side, side).unwrap();
        // The dark wedges are where both signed distances agree: the one between `d1` and
        // `−d2` is a marker border's corner.
        let (d1, d2) = ([t1.cos(), t1.sin()], [t2.cos(), t2.sin()]);
        let fit = fit_x_junction(
            &img,
            [c[0] + 0.4, c[1] - 0.3],
            [d1, [-d2[0], -d2[1]]],
            [4.5; 2],
            &mut Window::empty(),
        )
        .expect("fit")
        .corner;
        let err = (fit[0] - c[0]).hypot(fit[1] - c[1]);
        assert!(err < 0.02, "error {err}");
    }

    /// A junction 2 px from the frame: the window is clipped, and the fit still places it.
    #[test]
    fn fits_a_junction_the_frame_clips() {
        let side = 32;
        let c = [2.37, 16.21];
        let (t1, t2) = (0.2, 0.2 + 1.7);
        let data = render(c, t1, t2, side);
        let img = ImageView::new(&data, side, side, side).unwrap();
        let (d1, d2) = ([t1.cos(), t1.sin()], [t2.cos(), t2.sin()]);
        let (start, reach) = ([c[0] + 0.3, c[1] - 0.2], 4.5);
        // The premise: the window's bounding box crosses the left frame edge.
        assert!(start[0] - reach * d1[0].hypot(d2[0]) < 0.0);
        let win = &mut Window::empty();
        let fit =
            fit_x_junction(&img, start, [d1, [-d2[0], -d2[1]]], [reach; 2], win).expect("fit");
        let err = (fit.corner[0] - c[0]).hypot(fit.corner[1] - c[1]);
        assert!(err < 0.05, "error {err}");
        // The polish keeps a clipped window's junction too.
        let polished = fit.polish(&img, [0.0; 2], win);
        let err = (polished[0] - c[0]).hypot(polished[1] - c[1]);
        assert!(err < 0.05, "polished error {err}");
    }

    /// A seed past one window's move limit: the first fit lands at the junction but cannot
    /// vouch for it; the refit centred there can.
    #[test]
    fn refits_a_seed_past_the_move_limit() {
        let side = 32;
        let c = [16.27, 15.61];
        let (t1, t2) = (0.3, 0.3 + 1.9);
        let data = render(c, t1, t2, side);
        let img = ImageView::new(&data, side, side, side).unwrap();
        let (d1, d2) = ([t1.cos(), t1.sin()], [t2.cos(), t2.sin()]);
        let toward = [d1, [-d2[0], -d2[1]]];
        let (start, reach) = ([c[0] + 1.3, c[1] - 1.0], [3.0; 2]);
        // The premise: one fit converges but moves past its limit (1.5 px here).
        assert!(matches!(
            fit_once(&img, start, toward, reach, &mut Window::empty()),
            Some((_, Verdict::OffCentre(_)))
        ));
        let fit = fit_x_junction(&img, start, toward, reach, &mut Window::empty())
            .expect("fit")
            .corner;
        let err = (fit[0] - c[0]).hypot(fit[1] - c[1]);
        assert!(err < 0.05, "error {err}");
    }

    #[test]
    fn rejects_an_l_corner() {
        // A dark square's corner: one dark quadrant only.
        let side = 32;
        let mut data = vec![200u8; side * side];
        for y in 16..side {
            for x in 16..side {
                data[y * side + x] = 40;
            }
        }
        let img = ImageView::new(&data, side, side, side).unwrap();
        let toward = [[1.0, 0.0], [0.0, 1.0]];
        assert!(
            fit_x_junction(&img, [16.2, 15.8], toward, [4.5; 2], &mut Window::empty()).is_none()
        );
    }

    /// The AVX2 kernel agrees with the portable one on a real window.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_kernel_matches_the_portable_one() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            return;
        }
        let side = 32;
        let data = render([16.27, 15.61], 0.3, 2.2, side);
        let img = ImageView::new(&data, side, side, side).unwrap();
        let axes = [
            [0.3f64.cos(), 0.3f64.sin()],
            [-(2.2f64.cos()), -(2.2f64.sin())],
        ];
        let mut win = Window::empty();
        assert!(win.gather(&img, [16.5, 15.4], axes, [4.5; 2]));
        let params = Params::from([0.1, -0.2, 0.32, 2.21, 0.9, 120.0, -80.0]);
        let (a, ga, ca) = normal_equations_portable(&win, &params);
        #[allow(unsafe_code, reason = "test of the target_feature kernel")]
        // SAFETY: AVX2 and FMA were detected above.
        let (b, gb, cb) = unsafe { avx2::normal_equations(&win, &params) };
        let close = |x: f64, y: f64| (x - y).abs() <= 1e-9 * x.abs().max(1.0);
        assert!(a.iter().zip(b.iter()).all(|(x, y)| close(*x, *y)));
        assert!(ga.iter().zip(gb.iter()).all(|(x, y)| close(*x, *y)));
        assert!(close(ca, cb));
    }

    /// A window of the skewed junction of [`render`], and parameters near its fit.
    fn skewed_window() -> (Vec<u8>, [[f64; 2]; 2], Params) {
        let data = render([16.27, 15.61], 0.3, 2.2, 32);
        let axes = [
            [0.3f64.cos(), 0.3f64.sin()],
            [-(2.2f64.cos()), -(2.2f64.sin())],
        ];
        let params = Params::from([0.1, -0.2, 0.32, 2.21, 0.9, 120.0, -80.0]);
        (data, axes, params)
    }

    /// The AVX2 aperture kernel agrees with the portable one on a real window, bent lines
    /// included.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_aperture_kernel_matches_the_portable_one() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            return;
        }
        let (data, axes, params) = skewed_window();
        let img = ImageView::new(&data, 32, 32, 32).unwrap();
        let mut win = Window::empty();
        assert!(win.gather(&img, [16.5, 15.4], axes, [4.5; 2]));
        let close = |x: f64, y: f64| (x - y).abs() <= 1e-9 * x.abs().max(1.0);
        for curvature in [[0.0; 2], [0.004, -0.007]] {
            let (a, ga, ca) = normal_equations_aperture_portable(&win, &params, curvature);
            #[allow(unsafe_code, reason = "test of the target_feature kernel")]
            // SAFETY: AVX2 and FMA were detected above.
            let (b, gb, cb) = unsafe { avx2::normal_equations_aperture(&win, &params, curvature) };
            assert!(a.iter().zip(b.iter()).all(|(x, y)| close(*x, *y)));
            assert!(ga.iter().zip(gb.iter()).all(|(x, y)| close(*x, *y)));
            assert!(close(ca, cb));
            // The premise: the aperture and the bend each change the normal equations.
            let (c, _, _) = normal_equations_portable(&win, &params);
            assert!(a.iter().zip(c.iter()).any(|(x, y)| !close(*x, *y)));
        }
        let (straight, _, _) = normal_equations_aperture_portable(&win, &params, [0.0; 2]);
        let (bent, _, _) = normal_equations_aperture_portable(&win, &params, [0.004, -0.007]);
        assert!(
            straight
                .iter()
                .zip(bent.iter())
                .any(|(x, y)| !close(*x, *y))
        );
    }

    /// The aperture kernel's `JᵀWr` is the gradient of its cost: every analytic derivative of
    /// the model, checked by central differences. The cases are a skewed junction, bent lines,
    /// and a line within `MIN_APERTURE_RATIO` of a pixel axis, where the narrow box is held.
    ///
    /// The kernel differentiates the exact profile, while its `erf` is an approximation with
    /// 1.5·10⁻⁷ of ripple, whose slope the profile's second difference passes on: the two
    /// agree to 3·10⁻⁵ where the narrow box is held and to 5·10⁻⁶ elsewhere.
    #[test]
    fn aperture_kernel_gradient_matches_finite_differences() {
        let (data, axes, skewed) = skewed_window();
        let img = ImageView::new(&data, 32, 32, 32).unwrap();
        let mut win = Window::empty();
        assert!(win.gather(&img, [16.5, 15.4], axes, [4.5; 2]));
        let near_axis = Params::from([0.1, -0.2, 0.004, 1.5712, 0.45, 120.0, -80.0]);
        // The premise of the last case: both of its lines hold their narrow box.
        for t in [near_axis[T1], near_axis[T2]] {
            let n = normal(t);
            assert!(n[0].abs().min(n[1].abs()) < MIN_APERTURE_RATIO);
        }
        for (params, curvature) in [
            (skewed, [0.0; 2]),
            (skewed, [0.004, -0.007]),
            (near_axis, [0.0; 2]),
            (near_axis, [-0.003, 0.005]),
        ] {
            let (_, grad, _) = normal_equations_aperture_portable(&win, &params, curvature);
            for k in 0..PARAMS {
                let step = 1e-6 * params[k].abs().max(1.0);
                let cost = |sign: f64| {
                    let mut moved = params;
                    moved[k] += sign * step;
                    normal_equations_aperture_portable(&win, &moved, curvature).2
                };
                // cost = Σ w·r² with r = value − μ, so ∂cost/∂p = −2·JᵀWr.
                let numeric = (cost(1.0) - cost(-1.0)) / (2.0 * step);
                let analytic = -2.0 * grad[k];
                assert!(
                    (numeric - analytic).abs() <= 1e-4 * numeric.abs().max(1.0),
                    "parameter {k}: {numeric} against {analytic} ({params:?}, {curvature:?})"
                );
            }
        }
    }

    /// The model's value at one pixel, read off the kernel: with `A = 0`, `B = 1` and a pixel
    /// of value 0, `JᵀWr`'s level entry is minus the model.
    fn aperture_value(angles: [f64; 2], scale: f64, offset: [f64; 2]) -> f64 {
        let mut win = Window::empty();
        (win.dx[0], win.dy[0], win.weight[0], win.len) = (offset[0], offset[1], 1.0, 1);
        let params = Params::from([0.0, 0.0, angles[0], angles[1], scale, 0.0, 1.0]);
        -normal_equations_aperture_portable(&win, &params, [0.0; 2]).1[LEVEL]
    }

    /// `erf·erf` averaged over the unit pixel at `offset`, by 256 × 256 midpoints.
    fn pixel_average(angles: [f64; 2], scale: f64, offset: [f64; 2]) -> f64 {
        const FINE: u16 = 256;
        let (n1, n2) = (normal(angles[0]), normal(angles[1]));
        let mut acc = 0.0;
        for sy in 0..FINE {
            for sx in 0..FINE {
                let px = offset[0] + (f64::from(sx) + 0.5) / f64::from(FINE) - 0.5;
                let py = offset[1] + (f64::from(sy) + 0.5) / f64::from(FINE) - 0.5;
                acc += libm::erf((n1[0] * px + n1[1] * py) / scale)
                    * libm::erf((n2[0] * px + n2[1] * py) / scale);
            }
        }
        acc / f64::from(FINE).powi(2)
    }

    /// The aperture model against the integral it stands for, over the 25 pixels around a
    /// junction at a 0.2 px PSF (`s = 0.28`), the sharpest the model is asked to describe.
    /// Sampling at the centre is off by 0.30-0.34 of the contrast there. The model:
    /// - is the integral for orthogonal lines along the pixel axes, where the two profiles are
    ///   independent over the pixel (measured 1.5·10⁻⁵, the `erf` approximation);
    /// - stays within 0.025 of the contrast for lines at any angle, 50° to 90° apart, at the
    ///   one or two pixels that hold the junction: there the profiles depend on each other
    ///   beyond the covariance the model carries.
    #[test]
    fn aperture_model_matches_the_integral_over_the_pixel() {
        let scale = 0.28;
        let worst = |angles: [f64; 2]| {
            let mut worst = (0.0f64, 0.0f64);
            for iy in -2..=2 {
                for ix in -2..=2 {
                    let offset = [f64::from(ix) + 0.13, f64::from(iy) - 0.31];
                    let exact = pixel_average(angles, scale, offset);
                    let centre = libm::erf(
                        (normal(angles[0])[0] * offset[0] + normal(angles[0])[1] * offset[1])
                            / scale,
                    ) * libm::erf(
                        (normal(angles[1])[0] * offset[0] + normal(angles[1])[1] * offset[1])
                            / scale,
                    );
                    worst.0 = worst
                        .0
                        .max((aperture_value(angles, scale, offset) - exact).abs());
                    worst.1 = worst.1.max((centre - exact).abs());
                }
            }
            worst
        };
        let quarter = core::f64::consts::FRAC_PI_2;
        let (model, centre) = worst([0.0, quarter]);
        assert!(model < 1e-4, "axis-aligned: {model}");
        // The premise: sampling at the centre is far from the integral here.
        assert!(centre > 0.25, "axis-aligned, centre-sampled: {centre}");
        for (t1, apart) in [
            (0.2, 90.0),
            (0.4, 90.0),
            (0.785, 90.0),
            (0.3, 70.0),
            (0.3, 50.0),
        ] {
            let (model, centre) = worst([t1, t1 + f64::to_radians(apart)]);
            assert!(model < 0.03, "{t1} rad, {apart}° apart: {model}");
            assert!(
                centre > 10.0 * model,
                "{t1} rad, {apart}° apart, centre-sampled: {centre}"
            );
        }
    }

    /// An exact X-junction image: orthogonal lines, so `erf·erf` is the image of a Gaussian
    /// PSF of `sigma` px, box-integrated over each pixel by 16 × 16 points and rounded to 8 bits.
    /// Only the pixel aperture and the rounding differ from the centre-sampled model. Each line
    /// is bent by `curvature[i]` toward its left normal.
    fn render_exact(
        centre: [f64; 2],
        angle: f64,
        sigma: f64,
        curvature: [f64; 2],
        side: usize,
    ) -> Vec<u8> {
        const FINE: u8 = 16;
        let (n1, n2) = (normal(angle), normal(angle + core::f64::consts::FRAC_PI_2));
        let scale = core::f64::consts::SQRT_2 * sigma;
        let mut out = vec![0u8; side * side];
        for y in 0..side {
            for x in 0..side {
                let mut acc = 0.0;
                for sy in 0..FINE {
                    for sx in 0..FINE {
                        let px = x as f64 + (f64::from(sx) + 0.5) / f64::from(FINE) - centre[0];
                        let py = y as f64 + (f64::from(sy) + 0.5) / f64::from(FINE) - centre[1];
                        let bent = |n: [f64; 2], k: f64| {
                            let along = n[1] * px - n[0] * py;
                            (n[0] * px + n[1] * py - 0.5 * k * along * along) / scale
                        };
                        acc += 120.0
                            - 80.0
                                * libm::erf(bent(n1, curvature[0]))
                                * libm::erf(bent(n2, curvature[1]));
                    }
                }
                out[y * side + x] = (acc / f64::from(FINE).powi(2)).round() as u8;
            }
        }
        out
    }

    /// RMS error of the centre-sampled fit and of its polish over 100 junctions of
    /// [`render_exact`] at random sub-pixel phases, with lines within `tilt` of the pixel axes
    /// bent by `curvature`, which the polish is told.
    fn fit_and_polish_rms(sigma: f64, tilt: f64, curvature: [f64; 2]) -> (f64, f64) {
        use rand::prelude::*;
        let side = 24;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let (mut centre, mut polished) = (0.0, 0.0);
        let win = &mut Window::empty();
        let junctions = 100;
        for _ in 0..junctions {
            let half = 0.5 * side as f64;
            let c = [half + rng.random::<f64>(), half + rng.random::<f64>()];
            let t = (2.0 * rng.random::<f64>() - 1.0) * tilt;
            let data = render_exact(c, t, sigma, curvature, side);
            let img = ImageView::new(&data, side, side, side).unwrap();
            let t2 = t + core::f64::consts::FRAC_PI_2;
            // Line 1 is fitted along the reverse of its direction, which reverses its bend.
            let toward = [[t.cos(), t.sin()], [-t2.cos(), -t2.sin()]];
            let start = [
                c[0] + 0.3 * (rng.random::<f64>() - 0.5),
                c[1] + 0.3 * (rng.random::<f64>() - 0.5),
            ];
            let (fit, verdict) = fit_once(&img, start, toward, [4.75; 2], win).unwrap();
            assert_eq!(verdict, Verdict::Accepted);
            let refined = fit.polish(&img, [curvature[0], -curvature[1]], win);
            for (corner, sum) in [(fit.corner, &mut centre), (refined, &mut polished)] {
                *sum += (corner[0] - c[0]).powi(2) + (corner[1] - c[1]).powi(2);
            }
        }
        let rms = |sum: f64| (sum / f64::from(junctions)).sqrt();
        (rms(centre), rms(polished))
    }

    /// Edges within 1.5° of the pixel axes at σ = 0.3 px, the worst case for centre sampling:
    /// the polish over the aperture removes most of the error.
    #[test]
    fn the_polish_corrects_axis_aligned_edges() {
        let (centre, polished) = fit_and_polish_rms(0.3, 1.5f64.to_radians(), [0.0; 2]);
        // The premise: centre sampling is biased here.
        assert!(centre > 0.007, "centre-sampled RMS {centre}");
        assert!(
            polished < 0.003,
            "polished RMS {polished}, centre-sampled {centre}"
        );
    }

    /// Lines bent as by a lens (a 4.75 px window leaves an arc of curvature 0.006 /px by
    /// 0.07 px at its rim): the straight-line fit is pulled toward the inside of the bends,
    /// and the polish, told the curvatures, is not.
    #[test]
    fn the_polish_follows_bent_lines() {
        let (centre, polished) = fit_and_polish_rms(0.5, core::f64::consts::PI, [0.006, -0.004]);
        // The premise: straight lines misplace this junction.
        assert!(centre > 0.012, "centre-sampled RMS {centre}");
        assert!(
            polished < 0.004,
            "polished RMS {polished}, centre-sampled {centre}"
        );
    }

    /// A junction with only what [`side_curvatures`] reads.
    fn junction_at(corner: [f64; 2], angles: [f64; 2]) -> Junction {
        Junction {
            corner,
            angles,
            axes: [[1.0, 0.0], [0.0, 1.0]],
            reach: [1.0; 2],
            params: Params::zeros(),
        }
    }

    /// A 20 px square whose sides are arcs: its horizontal sides bent by 0.004 /px and its
    /// vertical ones by −0.003 /px, as a lens bends every line of one direction alike. The
    /// side curvatures are those bends, signed along the contour, and they do not move when
    /// every junction's lines scissor (ink spread) or every side bows alike.
    #[test]
    fn side_curvatures_are_the_bend_opposite_sides_do_not_share() {
        use core::f64::consts::PI;
        let (side, bend) = (20.0, [0.004, -0.003]);
        let corners = [[10.0f64, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]];
        // Along the contour, sides 0 and 1 run with their line's direction and 2 and 3 against.
        let lens = [bend[0], bend[1], -bend[0], -bend[1]];
        let junctions = |scissor: f64, shared: f64| -> [Junction; 4] {
            core::array::from_fn(|j| {
                let (prev, next) = ((j + 3) % 4, (j + 1) % 4);
                let chord = |from: usize, to: usize| {
                    (corners[to][1] - corners[from][1]).atan2(corners[to][0] - corners[from][0])
                };
                // An arc's tangent turns by κ·length, half before the chord's direction and
                // half after it.
                let leaving = chord(j, next) - 0.5 * side * (lens[j] + shared);
                let arriving = chord(prev, j) + 0.5 * side * (lens[prev] + shared);
                junction_at(corners[j], [arriving + PI - scissor, leaving + scissor])
            })
        };
        let close = |got: [f64; 4]| got.iter().zip(&lens).all(|(g, w)| (g - w).abs() < 1e-12);
        assert!(close(side_curvatures(&junctions(0.0, 0.0))));
        assert!(close(side_curvatures(&junctions(0.02, 0.0))));
        assert!(close(side_curvatures(&junctions(0.0, 0.002))));
    }

    /// A marker as a lens images it: a 24 px black square with a black connector square on
    /// each corner, so every corner is an X-junction, drawn through a map that bends its
    /// horizontal sides by 0.005 /px and its vertical ones by 0.004 /px (each side leaves its
    /// chord by a third of a pixel). The straight-line fits misplace the corners; the polish,
    /// with the curvatures the four fits measure themselves, does not. A wrong sign anywhere
    /// between the fits' angles and the kernel's bend doubles the error instead.
    #[test]
    fn a_marker_bent_by_a_lens_is_polished_to_its_corners() {
        const FINE: u8 = 8;
        let (side, half, connector, sigma) = (64usize, 12.0, 6.0, 0.5);
        let (bend_x, bend_y) = (-0.004, 0.005);
        let centre = [32.3, 31.6];
        // Board to image: a point at `q` lands at `q + ½·(bend_x·q.y², bend_y·q.x²)`.
        let image_of = |q: [f64; 2]| {
            [
                centre[0] + q[0] + 0.5 * bend_x * q[1] * q[1],
                centre[1] + q[1] + 0.5 * bend_y * q[0] * q[0],
            ]
        };
        let board_of = |p: [f64; 2]| {
            let mut q = [p[0] - centre[0], p[1] - centre[1]];
            for _ in 0..8 {
                q = [
                    p[0] - centre[0] - 0.5 * bend_x * q[1] * q[1],
                    p[1] - centre[1] - 0.5 * bend_y * q[0] * q[0],
                ];
            }
            q
        };
        // Ink coverage: the square and its four connectors, each blurred by the PSF.
        let step = |x: f64| 0.5 * (1.0 + libm::erf(x / (core::f64::consts::SQRT_2 * sigma)));
        let rect = |q: [f64; 2], lo: [f64; 2], hi: [f64; 2]| {
            step(q[0] - lo[0]) * step(hi[0] - q[0]) * step(q[1] - lo[1]) * step(hi[1] - q[1])
        };
        let ink = |q: [f64; 2]| {
            let mut ink = rect(q, [-half, -half], [half, half]);
            for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let (x, y) = (sx * half, sy * half);
                let far = [x + sx * connector, y + sy * connector];
                ink += rect(
                    q,
                    [x.min(far[0]), y.min(far[1])],
                    [x.max(far[0]), y.max(far[1])],
                );
            }
            ink
        };
        let mut data = vec![0u8; side * side];
        for y in 0..side {
            for x in 0..side {
                let mut acc = 0.0;
                for sy in 0..FINE {
                    for sx in 0..FINE {
                        let p = [
                            x as f64 + (f64::from(sx) + 0.5) / f64::from(FINE),
                            y as f64 + (f64::from(sy) + 0.5) / f64::from(FINE),
                        ];
                        acc += 200.0 - 160.0 * ink(board_of(p));
                    }
                }
                data[y * side + x] = (acc / f64::from(FINE).powi(2)).round() as u8;
            }
        }
        let img = ImageView::new(&data, side, side, side).unwrap();
        let truth = [[-half, -half], [half, -half], [half, half], [-half, half]].map(image_of);
        let win = &mut Window::empty();
        let junctions: [Junction; 4] = core::array::from_fn(|j| {
            let (prev, next) = (truth[(j + 3) % 4], truth[(j + 1) % 4]);
            let start = [truth[j][0] + 0.2, truth[j][1] - 0.15];
            let toward = [
                [prev[0] - start[0], prev[1] - start[1]],
                [next[0] - start[0], next[1] - start[1]],
            ];
            fit_x_junction(&img, start, toward, [4.75; 2], win).expect("fit")
        });
        let polished = polish_marker(&img, &junctions, win);
        let rms = |corners: [[f64; 2]; 4]| {
            let sum: f64 = corners
                .iter()
                .zip(&truth)
                .map(|(c, t)| (c[0] - t[0]).powi(2) + (c[1] - t[1]).powi(2))
                .sum();
            (sum / 4.0).sqrt()
        };
        let (straight, bent) = (rms(junctions.map(|j| j.corner)), rms(polished));
        // The premise: straight lines misplace these corners.
        assert!(straight > 0.015, "centre-sampled RMS {straight}");
        assert!(
            bent < 0.005,
            "polished RMS {bent}, centre-sampled {straight}"
        );
        // The curvatures the fits measured are the map's: along +x a side leaves its chord
        // toward +y, its left; along +y toward −x for a negative `bend_x`, its left again.
        let measured = side_curvatures(&junctions);
        for (got, want) in measured.iter().zip([bend_y, -bend_x, -bend_y, bend_x]) {
            assert!((got - want).abs() < 5e-4, "curvature {got} against {want}");
        }
    }
}
