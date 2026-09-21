//! Regression: `quad.upscale_factor > 1` must report every output (corners,
//! covariances, pose) in ORIGINAL-image coordinates, using the same
//! centre-aware (+0.5 pixel rule) mapping as decimation.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    dead_code,
    missing_docs
)]
use locus_core::bench_api::{compute_corner_error, generate_synthetic_test_image};
use locus_core::pose::CameraIntrinsics;
use locus_core::{DetectorBuilder, ImageView, TagFamily};

const CANVAS: usize = 640;
const FAMILY: TagFamily = TagFamily::AprilTag36h11;

struct Run {
    corners: [[f64; 2]; 4],
    cov_trace: f32,
    translation: Option<[f32; 3]>,
    n: usize,
}

fn run(data: &[u8], upscale: usize, decimation: usize, pose: bool) -> Run {
    let img = ImageView::new(data, CANVAS, CANVAS, CANVAS).unwrap();
    let mut det = DetectorBuilder::new()
        .with_family(FAMILY)
        .with_decimation(decimation)
        .with_upscale_factor(upscale)
        .build();
    let intr = CameraIntrinsics::new(800.0, 800.0, 320.0, 320.0);
    let out = if pose {
        det.detect(&img, Some(&intr), Some(0.1), false)
    } else {
        det.detect(&img, None, None, false)
    }
    .expect("detection failed");
    assert!(
        !out.is_empty(),
        "no detection (upscale={upscale}, decimation={decimation})"
    );
    let c = out.corners[0];
    let cov = out.corner_covariances[0];
    Run {
        corners: [0, 1, 2, 3].map(|k| [f64::from(c[k].x), f64::from(c[k].y)]),
        cov_trace: cov[0] + cov[3] + cov[4] + cov[7] + cov[8] + cov[11] + cov[12] + cov[15],
        translation: pose.then(|| {
            let p = out.poses[0].data;
            [p[0], p[1], p[2]]
        }),
        n: out.len(),
    }
}

fn max_diff(a: &[[f64; 2]; 4], b: &[[f64; 2]; 4]) -> f64 {
    let mut m = 0.0f64;
    for k in 0..4 {
        m = m
            .max((a[k][0] - b[k][0]).abs())
            .max((a[k][1] - b[k][1]).abs());
    }
    m
}

fn scene(tag_px: usize) -> (Vec<u8>, [[f64; 2]; 4]) {
    generate_synthetic_test_image(FAMILY, 0, tag_px, CANVAS, 0.0)
}

#[test]
fn upscale_corners_are_in_original_coordinates() {
    for tag_px in [100usize, 40] {
        let (data, gt) = scene(tag_px);
        let base = run(&data, 1, 1, false);
        let base_err = compute_corner_error(&base.corners, &gt);
        println!("tag {tag_px}: baseline GT err {base_err}");
        assert!(base_err < 0.05, "baseline err {base_err} (tag {tag_px})");
        // (upscale, decimation): decimation > 1 takes precedence over upscale.
        for (u, d) in [(2, 1), (3, 1), (4, 1), (1, 2), (2, 2)] {
            let r = run(&data, u, d, false);
            assert_eq!(r.n, base.n, "detection count (u={u}, d={d}, tag {tag_px})");
            let diff = max_diff(&r.corners, &base.corners);
            let err = compute_corner_error(&r.corners, &gt);
            println!("tag {tag_px} u={u} d={d}: diff {diff} GT err {err}");
            // u=2 is exact (~1e-4 px). u=3/4 carry a pre-existing ~0.5
            // upscaled-pixel corner-refinement residual on two corners
            // (0.13-0.17 orig px), unrelated to the coordinate mapping; a
            // mapping bug is >= 0.5 orig px (typically hundreds).
            let tol = if u == 2 { 0.02 } else { 0.25 };
            assert!(
                diff < tol && err < tol,
                "tag {tag_px}px u={u} d={d}: corners {:?} vs baseline {:?} (max diff {diff} px, GT err {err} px)",
                r.corners,
                base.corners
            );
        }
    }
}

#[test]
fn upscale_covariance_and_pose_are_in_original_units() {
    let (data, _) = scene(100);
    let base = run(&data, 1, 1, true);
    let up = run(&data, 2, 1, true);
    println!("cov trace base {} up {}", base.cov_trace, up.cov_trace);
    // Covariance is px^2: an un-mapped upscaled value would be ~4x larger.
    if base.cov_trace > 0.0 || up.cov_trace > 0.0 {
        let ratio = f64::from(up.cov_trace) / f64::from(base.cov_trace.max(1e-12));
        assert!(
            ratio < 3.0,
            "covariance trace ratio {ratio} suggests upscaled-frame units \
             (base {}, upscaled {})",
            base.cov_trace,
            up.cov_trace
        );
    }
    let (tb, tu) = (base.translation.unwrap(), up.translation.unwrap());
    for k in 0..3 {
        assert!(
            (tb[k] - tu[k]).abs() < 5e-3,
            "pose translation[{k}] differs: base {tb:?} vs upscaled {tu:?}"
        );
    }
}
