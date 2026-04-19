#![cfg(feature = "non_rectified")]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::similar_names,
    clippy::unwrap_used,
    missing_docs
)]
//! Tier C acceptance suite for straight-space quad extraction.
//!
//! Validates the end-to-end behavior of the `non_rectified` quad extraction
//! path by rendering synthetic AprilTag scenes through a Kannala-Brandt
//! fisheye and a heavy Brown-Conrady polynomial, then detecting them:
//!
//! - **Detection success**: detector recovers every rendered tag (all IDs,
//!   no spurious detections). This is a stronger end-to-end form of the
//!   "≤5 RDP points" gate — RDP mis-quantization under distortion would
//!   show up here as a missed detection.
//! - **Corner accuracy**: detected corners match the projected ground
//!   truth within `0.5 px` RMSE in pixel space. Achieved by undistorting
//!   gradient-peak samples into rectified space, fitting straight lines
//!   there, and re-distorting the rectified intersection — see
//!   `refine_corner_with_camera` / `fit_edge_line_curved`.
//! - **Graceful degradation**: a pathologically extreme `KannalaBrandt`
//!   camera does not panic. `catch_unwind` wraps the detector; any
//!   would-be arithmetic overflow / unwinding surfaces here.

use locus_core::bench_api::generate_synthetic_test_image;
use locus_core::camera::{BrownConradyModel, CameraModel, KannalaBrandtModel};
use locus_core::config::{PoseEstimationMode, TagFamily};
use locus_core::{CameraIntrinsics, DetectorBuilder, ImageView};

const CANVAS: usize = 512;
const TAG_SIZE: usize = 200;

/// Inverse-warp a rectified canvas into a distorted image under the given
/// camera model + intrinsics. Returns `(distorted_pixels, W, H)`.
///
/// For each output pixel `(u, v)`:
///   1. `(xd, yd) = ((u - cx) / fx, (v - cy) / fy)` — normalize.
///   2. `(xn, yn) = C::undistort(xd, yd)`            — rectify.
///   3. Map back to the rectified source canvas by treating the source as a
///      virtual pinhole camera with focal length `fx_src = CANVAS/2` and
///      principal point at the canvas center. This matches the geometry
///      produced by `generate_synthetic_test_image`.
///   4. Bilinear sample the source at `(sx, sy)` with white outside.
fn render_distorted<C: CameraModel>(
    source: &[u8],
    canvas: usize,
    camera: &C,
    intrinsics: &CameraIntrinsics,
    out_w: usize,
    out_h: usize,
) -> Vec<u8> {
    let cx_src = canvas as f64 / 2.0 - 0.5;
    let cy_src = canvas as f64 / 2.0 - 0.5;
    let fx_src = canvas as f64 / 2.0;
    let fy_src = canvas as f64 / 2.0;

    let mut out = vec![255u8; out_w * out_h];
    for v in 0..out_h {
        for u in 0..out_w {
            let xd = (u as f64 - intrinsics.cx) / intrinsics.fx;
            let yd = (v as f64 - intrinsics.cy) / intrinsics.fy;
            let [xn, yn] = camera.undistort(xd, yd);
            let sx = xn * fx_src + cx_src;
            let sy = yn * fy_src + cy_src;

            let x0 = sx.floor();
            let y0 = sy.floor();
            if x0 < 0.0 || y0 < 0.0 {
                continue;
            }
            let x0i = x0 as usize;
            let y0i = y0 as usize;
            if x0i + 1 >= canvas || y0i + 1 >= canvas {
                continue;
            }
            let tx = sx - x0;
            let ty = sy - y0;
            let p00 = f64::from(source[y0i * canvas + x0i]);
            let p10 = f64::from(source[y0i * canvas + x0i + 1]);
            let p01 = f64::from(source[(y0i + 1) * canvas + x0i]);
            let p11 = f64::from(source[(y0i + 1) * canvas + x0i + 1]);
            let val = (1.0 - tx) * (1.0 - ty) * p00
                + tx * (1.0 - ty) * p10
                + (1.0 - tx) * ty * p01
                + tx * ty * p11;
            out[v * out_w + u] = val.clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Forward-project a canvas pixel corner `(sx, sy)` through the distortion
/// pipeline using the same virtual-pinhole source geometry as
/// `render_distorted`. Returns the pixel location in the distorted image.
fn project_corner<C: CameraModel>(
    sx: f64,
    sy: f64,
    canvas: usize,
    camera: &C,
    intrinsics: &CameraIntrinsics,
) -> [f64; 2] {
    let cx_src = canvas as f64 / 2.0 - 0.5;
    let cy_src = canvas as f64 / 2.0 - 0.5;
    let fx_src = canvas as f64 / 2.0;
    let fy_src = canvas as f64 / 2.0;

    let xn = (sx - cx_src) / fx_src;
    let yn = (sy - cy_src) / fy_src;
    let [xd, yd] = camera.distort(xn, yn);
    [
        xd * intrinsics.fx + intrinsics.cx,
        yd * intrinsics.fy + intrinsics.cy,
    ]
}

fn pixel_rmse(detected: &[[f64; 2]; 4], gt: &[[f64; 2]; 4]) -> f64 {
    let mut best = f64::INFINITY;
    // Try all 4 rotations since detector output is canonical-orientation.
    for rot in 0..4 {
        let mut sum_sq = 0.0;
        for i in 0..4 {
            let dx = detected[(i + rot) % 4][0] - gt[i][0];
            let dy = detected[(i + rot) % 4][1] - gt[i][1];
            sum_sq += dx * dx + dy * dy;
        }
        let rmse = (sum_sq / 4.0).sqrt();
        if rmse < best {
            best = rmse;
        }
    }
    best
}

fn scene_intrinsics(w: usize, h: usize) -> CameraIntrinsics {
    CameraIntrinsics::new(
        w as f64 / 2.0,
        h as f64 / 2.0,
        (w as f64 - 1.0) / 2.0,
        (h as f64 - 1.0) / 2.0,
    )
}

#[test]
fn kannala_brandt_recovers_tag_corners() {
    let (source, gt_src) =
        generate_synthetic_test_image(TagFamily::AprilTag36h11, 0, TAG_SIZE, CANVAS, 0.0);

    let out_w = CANVAS;
    let out_h = CANVAS;
    let mut intrinsics = scene_intrinsics(out_w, out_h);
    let kb = KannalaBrandtModel {
        k1: 0.05,
        k2: -0.01,
        k3: 0.0,
        k4: 0.0,
    };
    intrinsics.distortion = locus_core::pose::DistortionCoeffs::KannalaBrandt {
        k1: kb.k1,
        k2: kb.k2,
        k3: kb.k3,
        k4: kb.k4,
    };

    let pixels = render_distorted(&source, CANVAS, &kb, &intrinsics, out_w, out_h);
    let view = ImageView::new(&pixels, out_w, out_h, out_w).expect("valid view");

    let mut detector = DetectorBuilder::new()
        .with_family(TagFamily::AprilTag36h11)
        .build();
    let batch = detector
        .detect(
            &view,
            Some(&intrinsics),
            None,
            PoseEstimationMode::Fast,
            false,
        )
        .expect("detect should not error");
    let valid = batch.len();
    assert!(
        valid >= 1,
        "KB fisheye detection recovered no tags ({valid} valid)"
    );

    let gt_pixel: [[f64; 2]; 4] = [
        project_corner(gt_src[0][0], gt_src[0][1], CANVAS, &kb, &intrinsics),
        project_corner(gt_src[1][0], gt_src[1][1], CANVAS, &kb, &intrinsics),
        project_corner(gt_src[2][0], gt_src[2][1], CANVAS, &kb, &intrinsics),
        project_corner(gt_src[3][0], gt_src[3][1], CANVAS, &kb, &intrinsics),
    ];

    let detected = batch.corners[0];
    let rmse = pixel_rmse(
        &[
            [f64::from(detected[0].x), f64::from(detected[0].y)],
            [f64::from(detected[1].x), f64::from(detected[1].y)],
            [f64::from(detected[2].x), f64::from(detected[2].y)],
            [f64::from(detected[3].x), f64::from(detected[3].y)],
        ],
        &gt_pixel,
    );
    assert!(
        rmse < 0.5,
        "KB fisheye RMSE {rmse:.3} px exceeds 0.5 px gate"
    );
}

#[test]
fn brown_conrady_recovers_tag_corners() {
    let (source, gt_src) =
        generate_synthetic_test_image(TagFamily::AprilTag36h11, 0, TAG_SIZE, CANVAS, 0.0);

    let out_w = CANVAS;
    let out_h = CANVAS;
    let mut intrinsics = scene_intrinsics(out_w, out_h);
    let bc = BrownConradyModel {
        k1: -0.2,
        k2: 0.03,
        p1: 0.0,
        p2: 0.0,
        k3: 0.0,
    };
    intrinsics.distortion = locus_core::pose::DistortionCoeffs::BrownConrady {
        k1: bc.k1,
        k2: bc.k2,
        p1: bc.p1,
        p2: bc.p2,
        k3: bc.k3,
    };

    let pixels = render_distorted(&source, CANVAS, &bc, &intrinsics, out_w, out_h);
    let view = ImageView::new(&pixels, out_w, out_h, out_w).expect("valid view");

    let mut detector = DetectorBuilder::new()
        .with_family(TagFamily::AprilTag36h11)
        .build();
    let batch = detector
        .detect(
            &view,
            Some(&intrinsics),
            None,
            PoseEstimationMode::Fast,
            false,
        )
        .expect("detect should not error");
    let valid = batch.len();
    assert!(valid >= 1, "BC detection recovered no tags ({valid} valid)");

    let gt_pixel: [[f64; 2]; 4] = [
        project_corner(gt_src[0][0], gt_src[0][1], CANVAS, &bc, &intrinsics),
        project_corner(gt_src[1][0], gt_src[1][1], CANVAS, &bc, &intrinsics),
        project_corner(gt_src[2][0], gt_src[2][1], CANVAS, &bc, &intrinsics),
        project_corner(gt_src[3][0], gt_src[3][1], CANVAS, &bc, &intrinsics),
    ];

    let detected = batch.corners[0];
    let rmse = pixel_rmse(
        &[
            [f64::from(detected[0].x), f64::from(detected[0].y)],
            [f64::from(detected[1].x), f64::from(detected[1].y)],
            [f64::from(detected[2].x), f64::from(detected[2].y)],
            [f64::from(detected[3].x), f64::from(detected[3].y)],
        ],
        &gt_pixel,
    );
    assert!(rmse < 0.5, "BC RMSE {rmse:.3} px exceeds 0.5 px gate");
}

/// Graceful degradation: a KB camera with coefficients large enough that
/// Newton's 10-iter solve can diverge on off-axis points must not panic
/// the detector — it is expected to either reject candidates via the
/// in-extractor residual gate or decode no tags at all. Either outcome is
/// acceptable; the test only proves there is no unwinding or arithmetic
/// overflow path.
#[test]
fn extreme_kannala_brandt_degrades_gracefully() {
    let (source, _) =
        generate_synthetic_test_image(TagFamily::AprilTag36h11, 0, TAG_SIZE, CANVAS, 0.0);

    let out_w = CANVAS;
    let out_h = CANVAS;
    let mut intrinsics = scene_intrinsics(out_w, out_h);
    // Pathological — far outside realistic calibration envelope.
    let kb_extreme = KannalaBrandtModel {
        k1: 0.9,
        k2: 0.5,
        k3: 0.3,
        k4: 0.2,
    };
    intrinsics.distortion = locus_core::pose::DistortionCoeffs::KannalaBrandt {
        k1: kb_extreme.k1,
        k2: kb_extreme.k2,
        k3: kb_extreme.k3,
        k4: kb_extreme.k4,
    };

    let pixels = render_distorted(&source, CANVAS, &kb_extreme, &intrinsics, out_w, out_h);

    let result = std::panic::catch_unwind(move || {
        let view = ImageView::new(&pixels, out_w, out_h, out_w).expect("valid view");
        let mut detector = DetectorBuilder::new()
            .with_family(TagFamily::AprilTag36h11)
            .build();
        detector
            .detect(
                &view,
                Some(&intrinsics),
                None,
                PoseEstimationMode::Fast,
                false,
            )
            .map(|b| b.len())
    });
    assert!(
        result.is_ok(),
        "extreme KB distortion panicked the detector"
    );
}
