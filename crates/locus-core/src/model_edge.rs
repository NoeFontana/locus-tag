//! Model-based edge pose refinement (opt-in, Accurate-mode post-decode stage).
//!
//! A corner-based PnP pose is determined by only 4 points. A decoded tag exposes
//! its full bit grid: every boundary between a black and a white cell — plus the
//! 4 outer border edges — is a high-contrast straight edge with known geometry.
//! For an `NxN`-cell 36h11 tag that is ~40 internal edges, an order of magnitude
//! more pose constraints than 4 corners, distributed across the tag interior
//! where they pin orientation far better than the corners alone.
//!
//! This stage refines the pose by aligning those model edges to the image:
//!
//! 1. Sample every candidate grid line (internal cell boundaries + border) in the
//!    tag frame, project under the current pose, and measure the sub-pixel edge
//!    location by the **50%-intensity crossing** along each edge normal (unbiased
//!    for a symmetric PSF). Low-contrast samples (same-colour cell boundaries with
//!    no real edge) are discarded automatically — the image contrast selects the
//!    real edges, so no bit-pattern reconstruction is needed.
//! 2. `measure → fit` iterations: with the measured edge points held *fixed*,
//!    Levenberg-Marquardt refines the full 6-DoF pose to those targets, then the
//!    targets are re-measured at the improved pose. This gives an excellent
//!    **rotation** but a slightly noisy translation (1-D edge offsets weakly
//!    constrain absolute depth/scale).
//! 3. Re-anchor **translation** by a translation-only solve against the 4 trusted
//!    corners under the refined rotation. Edges → rotation, corners → translation.
//! 4. A no-worse gate rejects the refinement when it is untrustworthy (too few
//!    edges, or an implausibly large pose jump from the corner pose).
//!
//! Empirically (1080p render-tag, see the benchmarking note) this cuts rotation
//! p99 ~0.60° → ~0.32° — beating OpenCV's `apriltag` refinement — while keeping
//! Locus's best-in-class translation. See
//! `docs/engineering/benchmarking/model_edge_refinement_*`.

use nalgebra::{Matrix2, Matrix3, Vector3, Vector6};

use crate::image::ImageView;
use crate::pose::{
    BodyFrameNormalEquations, CameraIntrinsics, NielsenConfig, Pose, nielsen_lm,
    projection_and_gradient,
};

// Tuning constants. Deliberately not exposed as config knobs — they are internal
// numerics of the estimator, tuned once against render-tag and not corpus-specific.
// The *selection* gates (CONTRAST_MIN / OFFSET_MAX / MIN_EDGE_SAMPLES) are the
// corpus-sensitive ones; if real-camera (low-contrast / motion-blur) validation
// shows the stage silently declining, promote those three to `DetectorConfig`
// (the LM iteration counts should stay `const`).
/// `measure → fit` outer iterations (re-measure targets at the improved pose).
const MEASURE_ITERS: usize = 4;
/// Nielsen trust-region config for the inner edge fit against a *fixed* target
/// set. Reuses the vetted corner/board [`NielsenConfig::POSE`] core — same
/// body-frame Jacobian, damping, and grad/func convergence gates — so the edge
/// LM can never drift from the production pose solver on numerics.
const EDGE_LM: NielsenConfig = NielsenConfig::POSE;
const TRANS_ITERS: usize = 3;
/// Interior fractions along each cell-boundary segment (avoid the cell-corner
/// junctions, where a perpendicular scan would hit a T-junction and bias).
const SEG_FRACS: [f64; 3] = [0.25, 0.5, 0.75];
/// Minimum black/white contrast (0..255) for a scan to count as a real edge.
/// High so only strong, unambiguous black↔white boundaries are used (weak/partial
/// edges give incoherent measurements that swamp the pose signal).
const CONTRAST_MIN: f64 = 50.0;
/// Half-window (px) for the perpendicular intensity scan, clamped below to stay
/// well inside one cell so a neighbouring edge never contaminates a scan.
const SCAN_HALF_MAX: f64 = 2.0;
/// Reject a measurement whose sub-pixel offset exceeds this: with a good corner
/// init the true edge is <1px from the projection, so a large offset means the
/// scan latched onto a *different* boundary (a spurious outlier).
const OFFSET_MAX: f64 = 1.25;
const SCAN_STEP: f64 = 0.25;
/// No-worse gate: reject refinements that move the pose implausibly far from the
/// corner pose (these are decode/geometry outliers, not genuine corrections).
const MAX_ROT_CHANGE_RAD: f64 = 3.0 * std::f64::consts::PI / 180.0;
const MAX_TRANS_CHANGE_FRAC: f64 = 0.05;
/// Minimum number of high-contrast edge samples to trust the refinement.
const MIN_EDGE_SAMPLES: usize = 24;
/// Per-sample cost charged when a fixed target projects behind the camera. Large
/// enough to dominate any plausible in-front residual so an LM step that pushes
/// targets behind the camera is *rejected* rather than rewarded for dropping
/// terms (a variable-subset sum would otherwise fall spuriously).
const BEHIND_PENALTY: f64 = 1e6;
/// Edge-tangent finite-difference length (tag units, fraction of the tag
/// half-extent `h`). Small so the projected chord is the *local* tangent — a
/// full-cell step would be the chord across a whole cell, whose perspective
/// curvature tilts the normal at oblique views. Projection is analytic, so a
/// tiny step carries no measurement noise.
const TAN_EPS_FRAC: f64 = 1e-2;
/// Upper bound on stored edge samples: `(N+1) * N * |SEG_FRACS| * 2`. For the
/// current families N ≤ 8 (36h11/ArUco 6x6) → 9*8*3*2 = 432 < 640; a larger grid
/// simply caps its sample count here, which only trims the fit slightly.
const MAX_SAMPLES: usize = 640;

/// A fixed edge measurement: the model point (tag frame), the measured sub-pixel
/// edge point (image px), and the edge normal (image px). Held constant while the
/// inner LM adjusts the pose to fit these targets.
#[derive(Clone, Copy)]
struct Sample {
    p: Vector3<f64>,
    m: [f64; 2],
    n: [f64; 2],
}

impl Default for Sample {
    fn default() -> Self {
        Self {
            p: Vector3::zeros(),
            m: [0.0; 2],
            n: [0.0; 2],
        }
    }
}

/// Measure the sub-pixel edge offset along the normal `(nx, ny)` through the
/// image point `(px, py)` via the 50%-intensity crossing nearest the centre.
/// Returns `(signed_offset_px, contrast)` or `None` if no clear edge is present.
fn measure_crossing(
    img: &ImageView,
    px: f64,
    py: f64,
    nx: f64,
    ny: f64,
    half: f64,
) -> Option<(f64, f64)> {
    let mut prof = [0.0f64; 64];
    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "half is clamped >= 1.0 and SCAN_STEP > 0, so the quotient is a small positive count"
    )]
    let n = ((2.0 * half / SCAN_STEP) as usize + 1).min(64);
    if n < 5 {
        return None;
    }
    // Reject scans that leave the image: `sample_bilinear` silently *clamps*
    // out-of-bounds coordinates to the edge pixel, which would feed a flat
    // plateau (still high-contrast) into the crossing and bias the edge target.
    // Both endpoints in-bounds ⇒ the whole (straight) scan is in-bounds.
    let (xmax, ymax) = ((img.width - 1) as f64, (img.height - 1) as f64);
    let oob = |x: f64, y: f64| x < 0.0 || x > xmax || y < 0.0 || y > ymax;
    if oob(px - half * nx, py - half * ny) || oob(px + half * nx, py + half * ny) {
        return None;
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for (k, slot) in prof.iter_mut().enumerate().take(n) {
        let t = -half + (k as f64) * SCAN_STEP;
        let v = img.sample_bilinear(px + t * nx, py + t * ny);
        *slot = v;
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let contrast = hi - lo;
    if contrast < CONTRAST_MIN {
        return None;
    }
    let mid = 0.5 * (lo + hi);
    // Index of the projected point (t = 0): t = -half + k·SCAN_STEP ⇒ k = half/step.
    // (`n/2` is off by up to half a step for a continuous `half`, which biases the
    // nearest-crossing pick when the profile has multiple crossings.)
    let center = half / SCAN_STEP;
    let mut best: Option<(usize, f64)> = None;
    for k in 0..n - 1 {
        let a = prof[k] - mid;
        let b = prof[k + 1] - mid;
        if (a <= 0.0) != (b <= 0.0) {
            let frac = if (b - a).abs() > 1e-9 {
                -a / (b - a)
            } else {
                0.5
            };
            let dist = (k as f64 + frac - center).abs();
            if best.is_none_or(|(_, d)| dist < d) {
                best = Some((k, dist));
            }
        }
    }
    let (k, _) = best?;
    let a = prof[k] - mid;
    let b = prof[k + 1] - mid;
    let frac = if (b - a).abs() > 1e-9 {
        -a / (b - a)
    } else {
        0.5
    };
    Some((-half + (k as f64 + frac) * SCAN_STEP, contrast))
}

/// Project a tag-frame point + tangent under `pose`, returning the projected
/// image point and unit edge normal, or `None` if behind the camera / degenerate.
fn project_with_normal(
    intr: &CameraIntrinsics,
    pose: &Pose,
    p_tag: Vector3<f64>,
    tan_tag: Vector3<f64>,
) -> Option<([f64; 2], [f64; 2])> {
    let p_cam = pose.rotation * p_tag + pose.translation;
    if p_cam.z < 1e-4 {
        return None;
    }
    let [pu, pv] = intr.distort_normalized(p_cam.x / p_cam.z, p_cam.y / p_cam.z);
    let p2 = pose.rotation * (p_tag + tan_tag) + pose.translation;
    if p2.z < 1e-4 {
        return None;
    }
    let [pu2, pv2] = intr.distort_normalized(p2.x / p2.z, p2.y / p2.z);
    let (mut tx, mut ty) = (pu2 - pu, pv2 - pv);
    let tn = (tx * tx + ty * ty).sqrt();
    if tn < 1e-9 {
        return None;
    }
    tx /= tn;
    ty /= tn;
    Some(([pu, pv], [-ty, tx]))
}

/// Measure a fixed target set at `pose`: every high-contrast cell-boundary sample.
/// Writes into `out` and returns the count.
#[expect(
    clippy::too_many_arguments,
    reason = "grid geometry (n, cell, h), image, camera, pose, scan window and output buffer are distinct per-frame inputs; a struct only adds indirection"
)]
fn measure_samples(
    img: &ImageView,
    intr: &CameraIntrinsics,
    pose: &Pose,
    n: usize,
    cell: f64,
    h: f64,
    scan_half: f64,
    out: &mut [Sample; MAX_SAMPLES],
) -> usize {
    let mut ns = 0;
    for k in 0..=n {
        let g = -1.0 + (k as f64) * cell; // boundary canonical coord
        for j in 0..n {
            for frac in SEG_FRACS {
                let along = -1.0 + (j as f64 + frac) * cell;
                let eps = TAN_EPS_FRAC * h; // local tangent step (see const doc)
                // Vertical boundary (const x): edge along +y. Horizontal: along +x.
                for (p_tag, tan_tag) in [
                    (
                        Vector3::new(g * h, along * h, 0.0),
                        Vector3::new(0.0, eps, 0.0),
                    ),
                    (
                        Vector3::new(along * h, g * h, 0.0),
                        Vector3::new(eps, 0.0, 0.0),
                    ),
                ] {
                    if ns >= MAX_SAMPLES {
                        return ns;
                    }
                    let Some(([pu, pv], nrm)) = project_with_normal(intr, pose, p_tag, tan_tag)
                    else {
                        continue;
                    };
                    if let Some((off, _c)) =
                        measure_crossing(img, pu, pv, nrm[0], nrm[1], scan_half)
                    {
                        if off.abs() > OFFSET_MAX {
                            continue; // spurious: latched onto a different boundary
                        }
                        out[ns] = Sample {
                            p: p_tag,
                            m: [pu + off * nrm[0], pv + off * nrm[1]],
                            n: nrm,
                        };
                        ns += 1;
                    }
                }
            }
        }
    }
    ns
}

/// Sum of the per-sample squared 1-D edge cost `½·((measured−projected)·n)²` over
/// the fixed sample set at `pose`, or [`BEHIND_PENALTY`] for a target that projects
/// behind the camera. The penalty (rather than dropping the term) keeps the cost
/// over a *fixed* term count so LM acceptance cannot be fooled by a step that sheds
/// samples. The `½` matches [`edge_normal_equations`]'s cost so `‖grad‖ = ‖−jtr‖`
/// (the convention the shared Nielsen gain-ratio assumes).
///
/// Test-only: production reuses the cost returned by [`edge_normal_equations`]; this
/// standalone form exists so `jacobian_matches_finite_difference` can numerically
/// differentiate the cost independently of the analytic `jtr`.
#[cfg(test)]
fn fixed_cost(intr: &CameraIntrinsics, pose: &Pose, samples: &[Sample]) -> f64 {
    let mut cost = 0.0;
    for s in samples {
        let p_cam = pose.rotation * s.p + pose.translation;
        if p_cam.z < 1e-4 {
            cost += BEHIND_PENALTY;
            continue;
        }
        let [pu, pv] = intr.distort_normalized(p_cam.x / p_cam.z, p_cam.y / p_cam.z);
        let res = (s.m[0] - pu) * s.n[0] + (s.m[1] - pv) * s.n[1];
        cost += 0.5 * res * res;
    }
    cost
}

/// Build the body-frame 6-DoF normal equations `(JᵀWJ, JᵀWr)` and the total cost
/// for the fixed sample set at `pose`, as the per-iteration closure fed to
/// [`nielsen_lm`]. Behind-camera targets contribute [`BEHIND_PENALTY`] to the cost
/// and no Jacobian term.
///
/// Each edge observation is a **1-D residual along the edge normal** `n`. That is
/// *exactly* the 2-D pixel residual `r = measured − projected` weighted by the
/// rank-1 projector `W = n·nᵀ`: for that `W`, [`BodyFrameNormalEquations::add`]
/// accumulates `Σ jⱼᵀ` and `Σ j·(n·r)` with `j = n₀·rowᵤ + n₁·rowᵥ` — i.e. the
/// projection-onto-normal Jacobian and 1-D residual. So the shared, distortion-
/// aware, body-frame-correct accumulator drives the edge fit with no bespoke
/// Jacobian, and the estimator is consistent with `Pose::retract` by construction.
fn edge_normal_equations(
    intr: &CameraIntrinsics,
    pose: &Pose,
    samples: &[Sample],
) -> (nalgebra::Matrix6<f64>, Vector6<f64>, f64) {
    let mut ne = BodyFrameNormalEquations::new(pose);
    let mut cost = 0.0;
    for s in samples {
        let p_cam = pose.rotation * s.p + pose.translation;
        let z = p_cam.z;
        if z < 1e-4 {
            cost += BEHIND_PENALTY; // consistent with `fixed_cost`
            continue;
        }
        let z_inv = 1.0 / z;
        let ([pu, pv], du, dv) =
            projection_and_gradient(intr, p_cam.x * z_inv, p_cam.y * z_inv, z_inv);
        let (res_u, res_v) = (s.m[0] - pu, s.m[1] - pv);
        let (n0, n1) = (s.n[0], s.n[1]);
        // 1-D residual along the edge normal; ½·res² to match the shared Nielsen
        // gain-ratio's cost convention (grad = −jtr).
        let res = n0 * res_u + n1 * res_v;
        cost += 0.5 * res * res;
        // Rank-1 projector onto the edge normal: reduces the 2-D corner accumulator
        // to the 1-D-along-normal edge fit (see the fn doc).
        let w = Matrix2::new(n0 * n0, n0 * n1, n0 * n1, n1 * n1);
        ne.add(&s.p, &du, &dv, res_u, res_v, &w);
    }
    let (jtj, jtr) = ne.finish();
    (jtj, jtr, cost)
}

/// Estimate the projected cell size (px) so the perpendicular scan window stays
/// inside a single cell (no cross-edge contamination).
fn projected_cell_px(corners: &[[f64; 2]; 4], n_cells: usize) -> f64 {
    let mut per = 0.0;
    for i in 0..4 {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        per += ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    }
    (per / 4.0) / (n_cells as f64)
}

/// Refine `init` by aligning the decoded tag's model edges to the image.
/// `dimension` is the data-grid dimension (6 for 36h11); the full grid incl. the
/// 1-cell border is `dimension + 2`. Returns the refined pose, or `None` if the
/// no-worse gate rejects it (caller keeps the corner pose).
#[must_use]
pub(crate) fn refine_pose_model_edges(
    intr: &CameraIntrinsics,
    img: &ImageView,
    dimension: usize,
    tag_size: f64,
    init: &Pose,
    corners: &[[f64; 2]; 4],
) -> Option<Pose> {
    let n = dimension + 2; // full grid incl. border ring
    let h = tag_size * 0.5;
    let cell = 2.0 / (n as f64); // canonical cell width
    let scan_half = (0.45 * projected_cell_px(corners, n)).clamp(1.0, SCAN_HALF_MAX);

    let mut pose = *init;
    // ~35 KB of fixed edge targets on the stack. This is a leaf function on the
    // opt-in Accurate-mode path (2 MB rayon worker stack); boxing would add a
    // per-tag heap allocation on the pose path for no benefit.
    #[allow(clippy::large_stack_arrays)]
    let mut samples = [Sample::default(); MAX_SAMPLES];

    // `measure → fit` outer loop: re-measure the edge targets at the improving
    // pose, then LM-fit the pose to those FIXED targets. Measuring against fixed
    // targets is what lets the cost actually fall as the pose aligns (a moving
    // re-scan would just track the projection and never improve).
    for outer in 0..MEASURE_ITERS {
        let ns = measure_samples(img, intr, &pose, n, cell, h, scan_half, &mut samples);
        if ns < MIN_EDGE_SAMPLES {
            if outer == 0 {
                return None; // never enough evidence — keep the corner pose
            }
            break; // later pass lost evidence — keep the pose earlier passes improved
        }
        let set = &samples[..ns];

        // Inner LM against the fixed targets, via the shared Nielsen trust-region
        // core (body-frame Jacobian + Marquardt damping + grad/func gates). Edges
        // barely constrain depth, so the adaptive damping is essential — Nielsen's
        // rule shrinks λ toward Gauss-Newton as it converges and grows it (with the
        // behind-camera penalty) to reject depth-blowup steps.
        let (refined, _jtj) = nielsen_lm(pose, &EDGE_LM, |p| edge_normal_equations(intr, p, set));
        if !refined.translation.iter().all(|x| x.is_finite())
            || !refined.rotation.iter().all(|x| x.is_finite())
        {
            return None;
        }
        pose = refined;
    }

    finalize(intr, &pose, tag_size, corners, init)
}

/// Stage 2 (translation re-anchor to the corners) + the no-worse gate.
fn finalize(
    intr: &CameraIntrinsics,
    edge_pose: &Pose,
    tag_size: f64,
    corners: &[[f64; 2]; 4],
    init: &Pose,
) -> Option<Pose> {
    let mut pose = *edge_pose;
    // --- Stage 2: translation-only re-anchor against the 4 trusted corners
    // (edges weakly constrain depth/scale; the outer corners constrain it best).
    let obj = crate::pose::centered_tag_corners(tag_size);
    for _ in 0..TRANS_ITERS {
        let mut ata = Matrix3::<f64>::zeros();
        let mut atb = Vector3::<f64>::zeros();
        for i in 0..4 {
            let p_cam = pose.rotation * obj[i] + pose.translation;
            let z = p_cam.z;
            if z < 1e-4 {
                return None;
            }
            let z_inv = 1.0 / z;
            let ([pu, pv], du, dv) =
                projection_and_gradient(intr, p_cam.x * z_inv, p_cam.y * z_inv, z_inv);
            ata += du * du.transpose() + dv * dv.transpose();
            atb += du * (corners[i][0] - pu) + dv * (corners[i][1] - pv);
        }
        let dt = ata.cholesky().map(|c| c.solve(&atb))?;
        if !dt.iter().all(|x| x.is_finite()) {
            return None;
        }
        pose.translation += dt;
        if dt.norm() < 1e-9 {
            break;
        }
    }

    // --- No-worse gate: reject implausibly large jumps from the corner pose.
    let r_rel = init.rotation.transpose() * pose.rotation;
    let cos_ang = ((r_rel.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
    if cos_ang.acos() > MAX_ROT_CHANGE_RAD {
        return None;
    }
    let dist = init.translation.norm().max(1e-6);
    if (pose.translation - init.translation).norm() / dist > MAX_TRANS_CHANGE_FRAC {
        return None;
    }
    Some(pose)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn frontal_pose(z: f64) -> Pose {
        Pose::new(Matrix3::identity(), Vector3::new(0.0, 0.0, z))
    }

    // A contrast-free image exposes no edges → the stage must decline (return
    // None), never emit a garbage pose.
    #[test]
    fn uniform_image_declines() {
        let data = vec![128u8; 200 * 200];
        let img = ImageView::new(&data, 200, 200, 200).unwrap();
        let intr = CameraIntrinsics::new(400.0, 400.0, 100.0, 100.0);
        let pose = frontal_pose(2.0);
        let corners = [[60.0, 60.0], [140.0, 60.0], [140.0, 140.0], [60.0, 140.0]];
        assert!(refine_pose_model_edges(&intr, &img, 6, 0.4, &pose, &corners).is_none());
    }

    // The analytic normal-equations gradient (−Jᵀr, since the cost carries a ½)
    // must match the numeric gradient of `fixed_cost`. This pins the edge fit's
    // body-frame Jacobian (via the shared `BodyFrameNormalEquations` + the rank-1
    // `n·nᵀ` weight) to the `Pose::retract` update + residual convention — a future
    // sign or frame flip in either would break this and only this test.
    #[test]
    fn jacobian_matches_finite_difference() {
        let intr = CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0);
        // A tilted, off-centre pose so all six DOF are genuinely exercised.
        let rot = nalgebra::Rotation3::new(Vector3::new(0.15, -0.10, 0.05))
            .matrix()
            .into_owned();
        let pose = Pose::new(rot, Vector3::new(0.02, -0.01, 1.5));
        let pts = [
            [-0.05, -0.05],
            [0.05, -0.03],
            [0.04, 0.05],
            [-0.03, 0.04],
            [0.0, 0.02],
        ];
        let samples: Vec<Sample> = pts
            .iter()
            .enumerate()
            .map(|(i, [x, y])| {
                let p = Vector3::new(*x, *y, 0.0);
                let pc = pose.rotation * p + pose.translation;
                let [u, v] = intr.distort_normalized(pc.x / pc.z, pc.y / pc.z);
                let ang = 0.3 * i as f64; // vary the normal per sample
                let (nx, ny) = (ang.cos(), ang.sin());
                // Measured edge sits 0.4 px off the projection along the normal.
                Sample {
                    p,
                    m: [u + 0.4 * nx, v + 0.4 * ny],
                    n: [nx, ny],
                }
            })
            .collect();

        let (_jtj, jtr, _cost) = edge_normal_equations(&intr, &pose, &samples);
        let analytic = -jtr; // d/dξ of Σ ½·residual²
        let eps = 1e-6;
        for k in 0..6 {
            let mut dp = Vector6::zeros();
            dp[k] = eps;
            let cp = fixed_cost(&intr, &pose.retract(&dp), &samples);
            dp[k] = -eps;
            let cm = fixed_cost(&intr, &pose.retract(&dp), &samples);
            let fd = (cp - cm) / (2.0 * eps);
            assert!(
                (fd - analytic[k]).abs() < 1e-2 + 1e-4 * analytic[k].abs(),
                "DOF {k}: finite-diff {fd} vs analytic {}",
                analytic[k]
            );
        }
    }

    // 50%-crossing localises a synthetic step edge to sub-pixel accuracy.
    #[test]
    fn crossing_locates_step_edge() {
        let (w, h) = (200usize, 40usize);
        let mut data = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = if (x as f64) >= 100.7 { 255 } else { 0 };
            }
        }
        let img = ImageView::new(&data, w, h, w).unwrap();
        let (off, contrast) = measure_crossing(&img, 100.0, 20.0, 1.0, 0.0, 3.0).unwrap();
        assert!(contrast > 200.0);
        assert!((off - 0.7).abs() < 0.35, "offset {off}");
    }
}
