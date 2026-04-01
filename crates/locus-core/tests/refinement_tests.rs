#![allow(missing_docs, clippy::unwrap_used, clippy::cast_sign_loss)]

use locus_core::bench_api::*;
use locus_core::ImageView;

fn render_saddle_point(x0: f64, y0: f64, sigma: f64, width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            let px = x as f64 + 0.5 - x0;
            let py = y as f64 + 0.5 - y0;
            // Saddle point model: I(x,y) = erf(x/sigma) * erf(y/sigma)
            let val = 127.5 + 127.5 * erf_approx(px / sigma) * erf_approx(py / sigma);
            data[y * width + x] = val as u8;
        }
    }
    data
}

#[test]
fn test_refine_saddle_point_synthetic() {
    let width = 11;
    let height = 11;
    let gt_x = 5.3;
    let gt_y = 5.7;
    let sigma = 0.8;
    let data = render_saddle_point(gt_x, gt_y, sigma, width, height);
    let view = ImageView::new(&data, width, height, width).unwrap();

    // Coarse estimate is the center of the window (5.0, 5.0)
    let coarse_x = 5.0;
    let coarse_y = 5.0;
    let radius = 5;

    let refined = refine_saddle_point(&view, coarse_x, coarse_y, radius);
    assert!(refined.is_some(), "Refinement failed to produce a result");
    let (rx, ry) = refined.unwrap();

    let err_x = (rx - gt_x).abs();
    let err_y = (ry - gt_y).abs();
    println!("Refinement error: x={err_x}, y={err_y}");

    // We expect < 0.1 pixel error
    assert!(err_x < 0.1, "Refined X {rx} far from ground truth {gt_x}");
    assert!(err_y < 0.1, "Refined Y {ry} far from ground truth {gt_y}");
}
