#![allow(
    missing_docs,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

mod utils;

use divan::bench;
use locus_core::ImageView;
use locus_core::bench_api::bench_compute_corner_covariance;
use std::hint::black_box;
use utils::BenchDataset;

const ALPHA_MAX: f64 = 0.1;
const SIGMA_N_SQ: f64 = 2.0;
const CENTERS_PER_BENCH: usize = 256;

fn main() {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();

    divan::Divan::from_args().threads([1]).run_benches();
}

fn benchmark_centers(img: &ImageView, radius: i32) -> Vec<[f64; 2]> {
    let margin = (radius + 2) as usize;
    let usable_w = img.width.saturating_sub(2 * margin).max(1);
    let usable_h = img.height.saturating_sub(2 * margin).max(1);
    let grid_w = 16;
    let grid_h = 16;
    let step_x = (usable_w / grid_w).max(1);
    let step_y = (usable_h / grid_h).max(1);

    let mut centers = Vec::with_capacity(CENTERS_PER_BENCH);
    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let x = margin + gx * step_x;
            let y = margin + gy * step_y;
            let sub_x = if (gx + gy) % 2 == 0 { 0.21 } else { 0.73 };
            let sub_y = if gy % 3 == 0 { 0.37 } else { 0.81 };
            centers.push([x as f64 + sub_x, y as f64 + sub_y]);
        }
    }
    centers.truncate(CENTERS_PER_BENCH);
    centers
}

#[bench(args = [2_i32, 4_i32, 8_i32])]
#[allow(clippy::trivially_copy_pass_by_ref)]
fn bench_corner_covariance(bencher: divan::Bencher, &radius: &i32) {
    let dataset = BenchDataset::icra_forward_0();
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .expect("valid benchmark image");
    let centers = benchmark_centers(&img, radius);

    bencher.bench_local(move || {
        let mut acc = 0.0;
        for &center in &centers {
            let cov = bench_compute_corner_covariance(&img, center, ALPHA_MAX, SIGMA_N_SQ, radius);
            acc += cov[(0, 0)] + cov[(0, 1)] + cov[(1, 1)];
        }
        black_box(acc)
    });
}
