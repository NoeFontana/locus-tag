#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]

use bumpalo::Bump;
use divan::{Bencher, black_box};
use locus_core::ImageView;
use locus_core::bench_api::subpixel::{Line, SubpixelEdgeRenderer};
use locus_core::bench_api::{ErfEdgeFitter, RefineConfig, SampleConfig};

fn main() {
    divan::Divan::from_args().threads([1]).run_benches();
}

const NUM_EDGES: usize = 100;
const EDGE_LEN: f64 = 60.0;

/// Generate `NUM_EDGES` synthetic 60-px vertical edges at quarter-pixel offsets.
fn build_edges(sigma: f64) -> Vec<(Vec<u8>, [f64; 2], [f64; 2])> {
    let width = 80;
    let height = (EDGE_LEN as usize) + 20;
    let renderer = SubpixelEdgeRenderer::new(width, height)
        .with_intensities(20.0, 230.0)
        .with_sigma(sigma);

    (0..NUM_EDGES)
        .map(|i| {
            let offset = (i % 4) as f64 * 0.25;
            let x_gt = 40.0 + offset;
            let p1 = [x_gt, 10.0];
            let p2 = [x_gt, 10.0 + EDGE_LEN];
            let line_gt = Line::from_points_cw(p1, p2);
            let data = renderer.render_edge_u8(&line_gt);
            // Seed 0.5 px off the true edge to exercise refinement.
            let seed_x = 40.25;
            (data, [seed_x, 10.0], [seed_x, 10.0 + EDGE_LEN])
        })
        .collect()
}

fn run_bench(bencher: Bencher, sigma: f64, decoder_style: bool) {
    let edges = build_edges(sigma);
    let width = 80;
    let height = (EDGE_LEN as usize) + 20;

    let sample_cfg = if decoder_style {
        SampleConfig::for_decoder()
    } else {
        SampleConfig::for_quad(EDGE_LEN, 1)
    };
    let refine_cfg = if decoder_style {
        RefineConfig::decoder_style(sigma)
    } else {
        RefineConfig::quad_style(sigma)
    };

    bencher.bench_local(move || {
        let arena = Bump::new();
        for (data, p1, p2) in &edges {
            let img = ImageView::new(data, width, height, width).unwrap();
            let mut fitter = ErfEdgeFitter::new(
                &img,
                *p1,
                *p2,
                !decoder_style, // quad uses midpoint init, decoder uses p1
            )
            .unwrap();
            fitter.fit(&arena, &sample_cfg, &refine_cfg);
            black_box(fitter.line_params());
        }
    });
}

#[divan::bench]
fn bench_quad_style_sigma_0_6(bencher: Bencher) {
    run_bench(bencher, 0.6, false);
}

#[divan::bench]
fn bench_quad_style_sigma_0_8(bencher: Bencher) {
    run_bench(bencher, 0.8, false);
}

#[divan::bench]
fn bench_decoder_style_sigma_0_6(bencher: Bencher) {
    run_bench(bencher, 0.6, true);
}

#[divan::bench]
fn bench_decoder_style_sigma_0_8(bencher: Bencher) {
    run_bench(bencher, 0.8, true);
}
