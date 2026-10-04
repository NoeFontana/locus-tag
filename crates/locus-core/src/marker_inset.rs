//! Photometric calibration of a decoded marker's corners from the marker's own bit edges.
//!
//! A camera blurs in linear light and then applies its tone curve, so an edge estimator that
//! works on the recorded intensities finds every blurred edge shifted by some `δ` toward one
//! side. On sRGB images the shift is 0.3–0.5 px toward the dark side and grows with the blur.
//! Every corner estimator (ours, OpenCV's, aruco_nano's) therefore reports the marker's corners
//! 0.4–0.7 px inside the printed ones. On a single tag that inset is indistinguishable from
//! depth (translation errors of centimetres), and under perspective it is not even the image of
//! a smaller square, so it also becomes rotation error. The shift depends on the blur, which
//! varies with depth along an oblique edge, so it cannot be a constant of the detector.
//!
//! The decoded marker measures `δ` itself. Its bit boundaries come in both polarities, and every
//! one shifts toward its dark side by the same `δ`; the layout says where each boundary is. For
//! every boundary of the decoded bit pattern, the edge offset from where the detected corners'
//! homography puts it is modelled as
//!
//! `e = δ·σ + (s − 1)·(u − N/2)·c + shrink(u; ε) + bow(u, w; q)`,
//!
//! - `σ = ±1`: which side of the boundary is dark;
//! - `s`: a scale of the interior layout about the marker centre (printing or rendering
//!   differences between the border and the bits);
//! - `ε` per side: how far the detected corners sit inside the true outline, which moves the
//!   homography's boundaries linearly from `+ε_near` at one side to `−ε_far` at the other;
//! - `q`: lens distortion the pinhole model does not describe. A homography maps lines to
//!   lines, so distortion shows as a bow along each boundary, `4a(1 − a)` at fraction `a` of its
//!   length (zero at the corners), with an amplitude that may vary across the marker.
//!
//! Polarity separates `δ`; the outer boundaries, which carry no `s` term, pin `ε`. The fit is a
//! ten-parameter robust least-squares problem, and the corners move outward by `ε` per side. It
//! needs no tone curve, blur model or camera knowledge. Markers whose layout does not match
//! (for example a 2-bit Kalibr border decoded as a 1-bit tag) fail the measurement guards and
//! keep their corners.

use crate::decoder::Homography;
use crate::image::ImageView;
use nalgebra::{SMatrix, SVector};

/// Model parameters: `δ`, `s − 1`, `ε` for sides 0–3, and the bow's constant and linear
/// amplitudes for each measurement axis.
const PARAMS: usize = 10;
type Params = SVector<f64, PARAMS>;
/// Indices of the bow parameters.
const BOW_PARAMS: [usize; 4] = [6, 7, 8, 9];
/// Critical value for adding the bow: the F(4, ∞) 99 % point is 3.32; the margin covers the
/// robust weights being estimated from the same residuals.
const BOW_F_CRITICAL: f64 = 3.5;

/// Largest layout handled, in cells across including the border.
const MAX_CELLS: usize = 10;
/// Positions along each boundary segment (fractions of its cell) where the edge is measured.
/// The segment ends are left out: there the boundary meets the next one at a junction.
const STATIONS: [f64; 2] = [1.0 / 3.0, 2.0 / 3.0];
/// Profile sample spacing (px). With central differences over two samples this is the 1 px
/// derivative baseline of the edge fits in `refinement.rs`.
const PROFILE_STEP: f64 = 0.5;
/// Half-width cap (px) of the window around each expected edge. It covers the blur measured on
/// the benchmarks; the window also stays within half a cell so the neighbouring boundary is
/// outside it.
const MAX_WINDOW_PX: f64 = 4.0;
/// Profile buffer: the widest window (`2 · MAX_WINDOW_PX / PROFILE_STEP + 1` samples) plus one
/// sample each side for the central difference.
const MAX_PROFILE: usize = 19;
/// Most boundary measurements: two axes × `MAX_CELLS` lines × `MAX_CELLS + 1` boundaries ×
/// stations.
const MAX_MEASUREMENTS: usize = 2 * MAX_CELLS * (MAX_CELLS + 1) * STATIONS.len();
/// Smallest cell (px) calibrated: the same floor as `corner_subpix`. Below it a cell holds
/// little more than the blur, and neighbouring boundaries share a window.
const MIN_CELL_PX: f64 = 2.0 / 0.6;
/// Smallest dark-to-white contrast (grey levels) for the bit pattern to be read.
const MIN_CONTRAST: f64 = 20.0;
/// Fraction of the marker's contrast an edge must step through within its window to count as
/// found.
const MIN_EDGE_STEP: f64 = 0.5;
/// Fraction of the expected boundary stations that must be found. A layout mismatch puts the
/// expected boundaries where there are none.
const MIN_FOUND_FRACTION: f64 = 0.75;
/// Huber scale (px) of the robust fit, and its iterations.
const HUBER_PX: f64 = 0.3;
const IRLS_ITERATIONS: usize = 6;
/// Largest median absolute residual (px) of an accepted fit.
const MAX_RESIDUAL_SCALE_PX: f64 = 0.25;
/// Largest interior scale deviation `|s − 1|` accepted. Printed and rendered markers measure
/// within 0.016 (ICRA's rendered artwork; 0.000 on the Blender renders, 0.004 on EuRoC); a
/// marker read with the wrong layout (a 2-cell border taken for 1) fits about 0.16.
const MAX_LAYOUT_SCALE_DEVIATION: f64 = 0.05;
/// Largest corner inset (px) accepted: past it the fit is not measuring a photometric shift.
const MAX_INSET_PX: f64 = 1.5;

/// One boundary measurement: its regression row and measured offset.
#[derive(Clone, Copy, Default)]
struct Measurement {
    /// `σ`: +1 when the dark cell is on the far (+u) side of the boundary.
    polarity: f32,
    /// `(u − N/2)·c` for interior boundaries, 0 for the outline.
    scale: f32,
    /// Position of the boundary across the marker, `u / N`.
    f: f32,
    /// Position of the station along the boundary, as a fraction of its length.
    along: f32,
    /// Measured edge offset (px) along +u from where the homography puts the boundary.
    offset: f32,
    /// 0: a boundary `u = b`, offset along u (moved by sides 3 and 1); 1: `v = b` (sides 0, 2).
    axis: u8,
}

/// Nonzero entries of a regression row, in ascending parameter order: the plain model's four
/// (`δ`, `s − 1`, and the two sides that move the boundary), then the bow's two.
type Row = ([usize; 6], [f64; 6]);
/// Row entries used by the plain model and by the model with the bow.
const PLAIN_ENTRIES: usize = 4;
const BOW_ENTRIES: usize = 6;

impl Measurement {
    fn row(&self) -> Row {
        let (f, along) = (f64::from(self.f), f64::from(self.along));
        let axis = usize::from(self.axis);
        let shape = 4.0 * along * (1.0 - along);
        // Axis 0 boundaries move with sides 3 (at u = 0) and 1 (at u = N); axis 1 with sides
        // 0 (v = 0) and 2 (v = N). Entries stay in ascending parameter order.
        let (low, high) = if axis == 0 { (3, 5) } else { (2, 4) };
        let (v_low, v_high) = if axis == 0 {
            (f, -(1.0 - f))
        } else {
            (-(1.0 - f), f)
        };
        (
            [0, 1, low, high, 6 + 2 * axis, 7 + 2 * axis],
            [
                f64::from(self.polarity),
                f64::from(self.scale),
                v_low,
                v_high,
                shape,
                shape * (2.0 * f - 1.0),
            ],
        )
    }
}

/// Corners of a decoded marker with its photometric inset removed, or `None` when the marker
/// cannot be calibrated (small cells, low contrast, a layout that does not match, or an
/// inconsistent fit).
///
/// `corners` are the marker's outline corners in its canonical cyclic order and `cells` the
/// number of cells across the marker including its border.
pub(crate) fn calibrate_marker_corners(
    img: &ImageView,
    corners: [[f64; 2]; 4],
    cells: usize,
) -> Option<[[f64; 2]; 4]> {
    if !(3..=MAX_CELLS).contains(&cells) {
        return None;
    }
    let side = (0..4)
        .map(|j| {
            let (p, q) = (corners[j], corners[(j + 1) % 4]);
            (q[0] - p[0]).hypot(q[1] - p[1])
        })
        .sum::<f64>()
        * 0.25;
    let n = cells as f64;
    if side / n < MIN_CELL_PX {
        return None;
    }
    let h = Homography::square_to_quad(&corners)?;
    // Image point of layout coordinates (u, v) ∈ [0, N]².
    let at = |u: f64, v: f64| h.project([2.0 * u / n - 1.0, 2.0 * v / n - 1.0]);
    // Every sample lies within one cell of the layout square (the quiet-zone reads at half a
    // cell out, profiles reach half a cell past the outline). While the projective depth `w`
    // keeps one sign over that padded square (it is affine in the layout coordinates, so its
    // corners decide), the square's image is a convex quad and its corners bound every sample.
    // A vanishing line crossing the padding (a strongly foreshortened quad) breaks that, and so
    // does a non-finite projection; both take the checked sampler.
    let padded = [
        (-1.0, -1.0),
        (n + 1.0, -1.0),
        (n + 1.0, n + 1.0),
        (-1.0, n + 1.0),
    ];
    let depth = |(u, v): (f64, f64)| {
        let (x, y) = (2.0 * u / n - 1.0, 2.0 * v / n - 1.0);
        h.h[(2, 0)] * x + h.h[(2, 1)] * y + h.h[(2, 2)]
    };
    let convex = padded.iter().all(|&c| depth(c) > 0.0) || padded.iter().all(|&c| depth(c) < 0.0);
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for &(u, v) in &padded {
        let p = at(u, v);
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let inside = convex
        && lo[0] >= 1.0
        && lo[1] >= 1.0
        && hi[0] <= img.width as f64 - 2.0
        && hi[1] <= img.height as f64 - 2.0;
    if inside {
        #[allow(
            unsafe_code,
            reason = "the bounds of every sample are checked once per marker above, so the per-sample checks of the safe sampler are redundant on this hot path"
        )]
        // SAFETY: `w` keeps one sign over the padded layout square, so its image is the convex
        // quad of its four corners, which lie in `[1, width − 2] × [1, height − 2]` (checked
        // above, finite since NaN fails every comparison). Every sample point is inside that
        // quad, so after the sampler's −0.5 shift both bilinear taps are valid pixel indices.
        calibrate_with(
            &|x, y| unsafe { img.sample_bilinear_unchecked(x, y) },
            &at,
            corners,
            cells,
        )
    } else {
        calibrate_with(&|x, y| img.sample_bilinear(x, y), &at, corners, cells)
    }
}

/// [`calibrate_marker_corners`] with an image sampler `sample(x, y)` and the layout-to-image
/// map `at(u, v)`.
fn calibrate_with(
    sample_px: &impl Fn(f64, f64) -> f64,
    at: &impl Fn(f64, f64) -> [f64; 2],
    corners: [[f64; 2]; 4],
    cells: usize,
) -> Option<[[f64; 2]; 4]> {
    let n = cells as f64;
    let sample = |u: f64, v: f64| {
        let p = at(u, v);
        sample_px(p[0], p[1])
    };

    let bits = read_bits(cells, &sample)?;
    let contrast = bits.white - bits.dark;

    let mut meas = [Measurement::default(); MAX_MEASUREMENTS];
    let mut count = 0usize;
    let mut expected = 0usize;
    let mut profile = [0.0f64; MAX_PROFILE];
    // Axis 0 measures the boundaries `u = b` along each row; axis 1 the boundaries `v = b`.
    for axis in 0..2 {
        for line in 0..cells {
            for b in 0..=cells {
                // Cells on the near (b − 1) and far (b) side; index 0 and N + 1 are quiet zone.
                let (near_dark, far_dark) = if axis == 0 {
                    (bits.dark_at(line + 1, b), bits.dark_at(line + 1, b + 1))
                } else {
                    (bits.dark_at(b, line + 1), bits.dark_at(b + 1, line + 1))
                };
                if near_dark == far_dark {
                    continue;
                }
                for &t in &STATIONS {
                    expected += 1;
                    let (lb, lt) = (b as f64, line as f64 + t);
                    let (p0, p1) = if axis == 0 {
                        (at(lb, lt), at(lb + 1.0, lt))
                    } else {
                        (at(lt, lb), at(lt, lb + 1.0))
                    };
                    let Some(offset) =
                        measure_edge(sample_px, p0, p1, far_dark, contrast, &mut profile)
                    else {
                        continue;
                    };
                    let cell_px = (p1[0] - p0[0]).hypot(p1[1] - p0[1]);
                    let scale = if b == 0 || b == cells {
                        0.0
                    } else {
                        (lb - 0.5 * n) * cell_px
                    };
                    #[allow(clippy::cast_possible_truncation)]
                    let m = Measurement {
                        polarity: if far_dark { 1.0 } else { -1.0 },
                        scale: scale as f32,
                        f: (lb / n) as f32,
                        along: (lt / n) as f32,
                        offset: offset as f32,
                        axis: u8::from(axis == 1),
                    };
                    meas[count] = m;
                    count += 1;
                }
            }
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let found = count as f64 / expected.max(1) as f64;
    if count < 24 || found < MIN_FOUND_FRACTION {
        return None;
    }
    let meas = &meas[..count];
    let theta = select_fit(meas)?;
    if (theta[1]).abs() > MAX_LAYOUT_SCALE_DEVIATION {
        return None;
    }
    let insets = [theta[2], theta[3], theta[4], theta[5]];
    if insets
        .iter()
        .any(|e| !e.is_finite() || e.abs() > MAX_INSET_PX)
    {
        return None;
    }
    move_sides_outward(corners, insets)
}

/// Reads the bit pattern: the border ring's and the quiet zone's median levels set the
/// threshold. `None` for a marker without enough contrast.
fn read_bits(cells: usize, sample: &impl Fn(f64, f64) -> f64) -> Option<BitsView> {
    let n = cells as f64;
    let mut ring = [0.0f64; 4 * MAX_CELLS];
    let mut quiet = [0.0f64; 4 * MAX_CELLS];
    let mut values = [[0.0f64; MAX_CELLS]; MAX_CELLS];
    for (v, row) in values.iter_mut().enumerate().take(cells) {
        for (u, value) in row.iter_mut().enumerate().take(cells) {
            *value = sample(u as f64 + 0.5, v as f64 + 0.5);
        }
    }
    let mut k = 0;
    for (i, row) in values.iter().enumerate().take(cells) {
        let c = i as f64 + 0.5;
        ring[k] = values[0][i];
        ring[k + 1] = values[cells - 1][i];
        ring[k + 2] = row[0];
        ring[k + 3] = row[cells - 1];
        quiet[k] = sample(c, -0.5);
        quiet[k + 1] = sample(c, n + 0.5);
        quiet[k + 2] = sample(-0.5, c);
        quiet[k + 3] = sample(n + 0.5, c);
        k += 4;
    }
    let dark = median(&mut ring[..k]);
    let white = median(&mut quiet[..k]);
    if white - dark < MIN_CONTRAST {
        return None;
    }
    let threshold = 0.5 * (dark + white);
    let mut grid = [[false; MAX_CELLS + 2]; MAX_CELLS + 2];
    for v in 0..cells {
        for u in 0..cells {
            grid[v + 1][u + 1] = values[v][u] < threshold;
        }
    }
    Some(BitsView { grid, dark, white })
}

/// The decoded marker's bit pattern, read at the cell centres through the homography.
struct BitsView {
    /// `grid[v][u]`: dark cell, over the layout padded by one quiet-zone cell on each side.
    grid: [[bool; MAX_CELLS + 2]; MAX_CELLS + 2],
    /// Median grey levels of the border ring and of the quiet zone.
    dark: f64,
    white: f64,
}

impl BitsView {
    fn dark_at(&self, v: usize, u: usize) -> bool {
        self.grid[v][u]
    }
}

/// Median by selection (the upper median for an even count).
fn median(values: &mut [f64]) -> f64 {
    let m = values.len() / 2;
    *values.select_nth_unstable_by(m, f64::total_cmp).1
}

/// Offset (px, along `p0 → p1`) of the edge expected at `p0`: the gradient-weighted centroid
/// of the profile across the edge within `±min(cell/2, 4 px)`, taking only gradients of the
/// expected sign (1 px central differences of samples `PROFILE_STEP` apart). `None` if the
/// profile does not step through `MIN_EDGE_STEP` of the marker's contrast there.
fn measure_edge(
    sample: &impl Fn(f64, f64) -> f64,
    p0: [f64; 2],
    p1: [f64; 2],
    far_dark: bool,
    contrast: f64,
    profile: &mut [f64; MAX_PROFILE],
) -> Option<f64> {
    let (dx, dy) = (p1[0] - p0[0], p1[1] - p0[1]);
    let cell_px = dx.hypot(dy);
    if cell_px < 1e-9 {
        return None;
    }
    let (nx, ny) = (dx / cell_px, dy / cell_px);
    let window = (0.5 * cell_px).min(MAX_WINDOW_PX);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let steps = (window / PROFILE_STEP).floor() as usize;
    let reach = steps + 1;
    let samples = 2 * reach + 1;
    for (idx, slot) in profile[..samples].iter_mut().enumerate() {
        let o = (idx as f64 - reach as f64) * PROFILE_STEP;
        *slot = sample(p0[0] + o * nx, p0[1] + o * ny);
    }
    // Intensity falls along +u into a dark far cell, rises out of a dark near cell.
    let sign = if far_dark { -1.0 } else { 1.0 };
    let (mut sum_w, mut sum_wo) = (0.0, 0.0);
    for at in 1..samples - 1 {
        let g = (sign * (profile[at + 1] - profile[at - 1])).max(0.0);
        let o = (at as f64 - reach as f64) * PROFILE_STEP;
        sum_w += g;
        sum_wo += g * o;
    }
    // `sum_w` is the gradient summed over samples (a 2-step difference each): the step the
    // profile makes within the window is `sum_w / 2`.
    (0.5 * sum_w >= MIN_EDGE_STEP * contrast).then(|| sum_wo / sum_w)
}

/// Weighted normal equations of the first `entries` row entries (the plain model or the one
/// with the bow), accumulated sparsely on the upper triangle.
fn normal_equations(
    meas: &[Measurement],
    weights: &[f64],
    entries: usize,
) -> ([[f64; PARAMS]; PARAMS], [f64; PARAMS]) {
    let mut a = [[0.0f64; PARAMS]; PARAMS];
    let mut b = [0.0f64; PARAMS];
    for (m, &w) in meas.iter().zip(weights) {
        let (idx, val) = m.row();
        let y = f64::from(m.offset);
        for i in 0..entries {
            let wv = w * val[i];
            b[idx[i]] += wv * y;
            for j in i..entries {
                a[idx[i]][idx[j]] += wv * val[j];
            }
        }
    }
    (a, b)
}

/// Solves the normal equations of the first `entries` row entries. The plain model's six
/// parameters are solved as a 6×6 system and the bow stays at zero.
fn solve(a: &[[f64; PARAMS]; PARAMS], b: &[f64; PARAMS], entries: usize) -> Option<Params> {
    fn cholesky_solve<const K: usize>(
        a: &[[f64; PARAMS]; PARAMS],
        b: &[f64; PARAMS],
    ) -> Option<SVector<f64, K>> {
        let m = SMatrix::<f64, K, K>::from_fn(|i, j| a[i.min(j)][i.max(j)]);
        m.cholesky()
            .map(|c| c.solve(&SVector::<f64, K>::from_column_slice(&b[..K])))
    }
    let mut theta = Params::zeros();
    if entries == PLAIN_ENTRIES {
        theta
            .fixed_rows_mut::<{ BOW_PARAMS[0] }>(0)
            .copy_from(&cholesky_solve::<{ BOW_PARAMS[0] }>(a, b)?);
    } else {
        theta = cholesky_solve::<PARAMS>(a, b)?;
    }
    Some(theta)
}

/// Residuals of the first `entries` row entries at `theta` into `out`.
fn residuals(meas: &[Measurement], theta: &Params, entries: usize, out: &mut [f64]) {
    for (m, r) in meas.iter().zip(out.iter_mut()) {
        let (idx, val) = m.row();
        let fit: f64 = (0..entries).map(|i| val[i] * theta[idx[i]]).sum();
        *r = f64::from(m.offset) - fit;
    }
}

/// Huber IRLS from the current `weights`, which it updates; stops once no weight moves by more
/// than `1e-3`. Leaves the final residuals in `res`.
fn irls(
    meas: &[Measurement],
    entries: usize,
    weights: &mut [f64],
    res: &mut [f64],
) -> Option<Params> {
    let mut theta = Params::zeros();
    for _it in 0..IRLS_ITERATIONS {
        let (a, b) = normal_equations(meas, weights, entries);
        theta = solve(&a, &b, entries)?;
        residuals(meas, &theta, entries, res);
        let mut moved = 0.0f64;
        for (w, &r) in weights.iter_mut().zip(res.iter()) {
            let updated = if r.abs() <= HUBER_PX {
                1.0
            } else {
                HUBER_PX / r.abs()
            };
            moved = moved.max((updated - *w).abs());
            *w = updated;
        }
        if moved < 1e-3 {
            break;
        }
    }
    Some(theta)
}

/// The boundary model for these measurements: the plain model, or the one with the lens bow
/// when, at the plain fit's robust weights, the bow reduces the weighted residual sum of squares
/// by more than its four parameters would by chance (an F-test at `BOW_F_CRITICAL`). On a
/// rectified image the plain model wins and the bow costs no precision. `None` when the chosen
/// fit is singular or too inconsistent to be a calibration.
fn select_fit(meas: &[Measurement]) -> Option<Params> {
    let n = meas.len();
    let mut weights = [1.0f64; MAX_MEASUREMENTS];
    let mut res = [0.0f64; MAX_MEASUREMENTS];
    let (weights, res) = (&mut weights[..n], &mut res[..n]);
    let mut theta = irls(meas, PLAIN_ENTRIES, weights, res)?;

    let weighted_ss = |r: &[f64], w: &[f64]| r.iter().zip(w).map(|(r, w)| w * r * r).sum::<f64>();
    let plain_ss = weighted_ss(res, weights);
    let (a, b) = normal_equations(meas, weights, BOW_ENTRIES);
    if let Some(curved) = solve(&a, &b, BOW_ENTRIES) {
        let mut curved_res = [0.0f64; MAX_MEASUREMENTS];
        let curved_res = &mut curved_res[..n];
        residuals(meas, &curved, BOW_ENTRIES, curved_res);
        let curved_ss = weighted_ss(curved_res, weights);
        #[allow(clippy::cast_precision_loss)]
        let dof = n as f64 - PARAMS as f64;
        #[allow(clippy::cast_precision_loss)]
        let f = ((plain_ss - curved_ss) / BOW_PARAMS.len() as f64) / (curved_ss / dof);
        if dof > 0.0 && curved_ss > 0.0 && f > BOW_F_CRITICAL {
            theta = irls(meas, BOW_ENTRIES, weights, res)?;
        }
    }
    for r in res.iter_mut() {
        *r = r.abs();
    }
    (median(res) <= MAX_RESIDUAL_SCALE_PX).then_some(theta)
}

/// Moves each side of the quad outward by its inset (side `e` joins corners `e` and `e + 1`);
/// each corner goes to the intersection of its two moved sides.
fn move_sides_outward(corners: [[f64; 2]; 4], insets: [f64; 4]) -> Option<[[f64; 2]; 4]> {
    let centre = [
        corners.iter().map(|c| c[0]).sum::<f64>() * 0.25,
        corners.iter().map(|c| c[1]).sum::<f64>() * 0.25,
    ];
    // Unit normal of side `e`, pointing into the marker.
    let inward = |e: usize| {
        let (p, q) = (corners[e], corners[(e + 1) % 4]);
        let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
        let len = dx.hypot(dy);
        let mut nrm = [-dy / len, dx / len];
        if (centre[0] - p[0]) * nrm[0] + (centre[1] - p[1]) * nrm[1] < 0.0 {
            nrm = [-nrm[0], -nrm[1]];
        }
        nrm
    };
    let mut out = corners;
    for (j, slot) in out.iter_mut().enumerate() {
        let prev = (j + 3) % 4;
        let (a, b) = (inward(prev), inward(j));
        // Solve a·Δ = −ε_prev, b·Δ = −ε_j.
        let det = a[0] * b[1] - a[1] * b[0];
        if det.abs() < 1e-6 {
            return None;
        }
        let (ra, rb) = (-insets[prev], -insets[j]);
        slot[0] += (ra * b[1] - rb * a[1]) / det;
        slot[1] += (a[0] * rb - b[0] * ra) / det;
    }
    Some(out)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::cast_possible_wrap
)]
mod tests {
    use super::*;

    const SIZE: usize = 200;
    /// True outline of the synthetic marker: oblique, not a parallelogram.
    const OUTLINE: [[f64; 2]; 4] = [[42.3, 38.7], [163.1, 52.4], [151.8, 160.2], [35.6, 147.9]];

    /// Dark cells of an `n × n` layout: the border ring (`border` cells wide) and a fixed
    /// pseudo-random interior.
    fn layout(n: usize, border: usize) -> Vec<bool> {
        (0..n * n)
            .map(|i| {
                let (u, v) = (i % n, i / n);
                let ring = u < border || v < border || u >= n - border || v >= n - border;
                ring || (u * 7 + v * 13 + u * v) % 5 < 2
            })
            .collect()
    }

    /// Radial (barrel for `k1 < 0`) lens distortion about the image centre, in units of 100 px.
    fn distort(p: [f64; 2], k1: f64) -> [f64; 2] {
        let c = SIZE as f64 / 2.0;
        let (x, y) = ((p[0] - c) / 100.0, (p[1] - c) / 100.0);
        let g = 1.0 + k1 * (x * x + y * y);
        [c + 100.0 * x * g, c + 100.0 * y * g]
    }

    fn undistort(p: [f64; 2], k1: f64) -> [f64; 2] {
        let mut u = p;
        for _ in 0..20 {
            let d = distort(u, k1);
            u = [u[0] + p[0] - d[0], u[1] + p[1] - d[1]];
        }
        u
    }

    /// Renders the marker in linear light (supersampled), blurs it with a Gaussian of `sigma`
    /// px and encodes it with the sRGB tone curve (or keeps it linear).
    fn render(n: usize, border: usize, sigma: f64, srgb: bool) -> Vec<u8> {
        render_distorted(n, border, sigma, srgb, 0.0)
    }

    /// [`render`] through radial lens distortion `k1`.
    fn render_distorted(n: usize, border: usize, sigma: f64, srgb: bool, k1: f64) -> Vec<u8> {
        let dark = layout(n, border);
        let h = Homography::square_to_quad(&OUTLINE).unwrap();
        let inv = h.h.try_inverse().unwrap();
        let ss = 4;
        let mut lin = vec![0.0f64; SIZE * SIZE];
        for y in 0..SIZE {
            for x in 0..SIZE {
                let mut acc = 0.0;
                for sy in 0..ss {
                    for sx in 0..ss {
                        let px = x as f64 + (sx as f64 + 0.5) / ss as f64;
                        let py = y as f64 + (sy as f64 + 0.5) / ss as f64;
                        let [px, py] = undistort([px, py], k1);
                        let q = inv * nalgebra::Vector3::new(px, py, 1.0);
                        let (cu, cv) = (q[0] / q[2], q[1] / q[2]);
                        let (u, v) = ((cu + 1.0) * 0.5 * n as f64, (cv + 1.0) * 0.5 * n as f64);
                        let inside = (0.0..n as f64).contains(&u) && (0.0..n as f64).contains(&v);
                        let is_dark = inside && dark[v as usize * n + u as usize];
                        acc += if is_dark { 0.03 } else { 0.8 };
                    }
                }
                lin[y * SIZE + x] = acc / (ss * ss) as f64;
            }
        }
        let radius = (3.0 * sigma).ceil() as i64;
        let kernel: Vec<f64> = (-radius..=radius)
            .map(|k| (-(k * k) as f64 / (2.0 * sigma * sigma)).exp())
            .collect();
        let norm: f64 = kernel.iter().sum();
        let blur = |src: &[f64], horizontal: bool| -> Vec<f64> {
            let mut out = vec![0.0; SIZE * SIZE];
            for y in 0..SIZE as i64 {
                for x in 0..SIZE as i64 {
                    let mut acc = 0.0;
                    for (k, w) in (-radius..=radius).zip(&kernel) {
                        let (sx, sy) = if horizontal { (x + k, y) } else { (x, y + k) };
                        let sx = sx.clamp(0, SIZE as i64 - 1) as usize;
                        let sy = sy.clamp(0, SIZE as i64 - 1) as usize;
                        acc += w * src[sy * SIZE + sx];
                    }
                    out[y as usize * SIZE + x as usize] = acc / norm;
                }
            }
            out
        };
        let lin = blur(&blur(&lin, true), false);
        lin.iter()
            .map(|&l| {
                let e = if !srgb {
                    l
                } else if l <= 0.003_130_8 {
                    12.92 * l
                } else {
                    1.055 * l.powf(1.0 / 2.4) - 0.055
                };
                (e * 255.0).round().clamp(0.0, 255.0) as u8
            })
            .collect()
    }

    /// The outline moved inward by `d` px on every side.
    fn inset(d: f64) -> [[f64; 2]; 4] {
        move_sides_outward(OUTLINE, [-d; 4]).unwrap()
    }

    fn max_error(a: &[[f64; 2]; 4], b: &[[f64; 2]; 4]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(p, q)| (p[0] - q[0]).hypot(p[1] - q[1]))
            .fold(0.0, f64::max)
    }

    #[test]
    fn recovers_the_outline_from_inset_corners_on_srgb_blur() {
        let data = render(8, 1, 1.2, true);
        let img = ImageView::new(&data, SIZE, SIZE, SIZE).unwrap();
        // A corner estimator on these intensities reports the corners inside the outline.
        let seen = inset(0.5);
        let calibrated = calibrate_marker_corners(&img, seen, 8).expect("calibrated");
        assert!(max_error(&seen, &OUTLINE) > 0.5);
        let err = max_error(&calibrated, &OUTLINE);
        assert!(err < 0.12, "calibrated corners off by {err:.3} px");
    }

    #[test]
    fn leaves_exact_corners_of_a_linear_image_in_place() {
        let data = render(8, 1, 1.2, false);
        let img = ImageView::new(&data, SIZE, SIZE, SIZE).unwrap();
        let calibrated = calibrate_marker_corners(&img, OUTLINE, 8).expect("calibrated");
        let err = max_error(&calibrated, &OUTLINE);
        assert!(err < 0.08, "moved exact corners by {err:.3} px");
    }

    #[test]
    fn lens_distortion_does_not_corrupt_the_calibration() {
        // Barrel distortion bends every boundary of the marker; the corners are where the
        // distorted outline's corners land.
        let k1 = -0.06;
        let data = render_distorted(8, 1, 1.2, true, k1);
        let img = ImageView::new(&data, SIZE, SIZE, SIZE).unwrap();
        let truth = OUTLINE.map(|c| distort(c, k1));
        let seen = move_sides_outward(truth, [-0.5; 4]).unwrap();
        let calibrated = calibrate_marker_corners(&img, seen, 8).expect("calibrated");
        let err = max_error(&calibrated, &truth);
        assert!(
            err < 0.15,
            "calibrated corners off by {err:.3} px under distortion"
        );
    }

    #[test]
    fn declines_a_layout_that_does_not_match() {
        // A 2-cell (Kalibr) border read as a 1-cell layout.
        let data = render(10, 2, 1.2, true);
        let img = ImageView::new(&data, SIZE, SIZE, SIZE).unwrap();
        assert!(calibrate_marker_corners(&img, OUTLINE, 8).is_none());
    }

    #[test]
    fn declines_cells_smaller_than_the_floor() {
        let data = render(8, 1, 1.2, true);
        let img = ImageView::new(&data, SIZE, SIZE, SIZE).unwrap();
        assert!(calibrate_marker_corners(&img, OUTLINE, 40).is_none());
    }

    #[test]
    fn moving_sides_outward_inverts_an_inset() {
        let back = move_sides_outward(inset(0.7), [0.7; 4]).unwrap();
        assert!(max_error(&back, &OUTLINE) < 1e-3);
    }
}
