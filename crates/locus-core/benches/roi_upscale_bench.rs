#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::items_after_statements,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unwrap_used,
    dead_code,
    missing_docs
)]
//! Benchmarks for the ROI-rescue upscale primitive.
//!
//! Covers bilinear vs. Lanczos3 on the side lengths that matter in
//! practice (32, 64, 128) and the two supported factors (2, 4). The
//! rescue stage caps at `max_roi_side_px = 128` and `factor ∈ {2, 4}`,
//! so 128×4 = 512 is the worst-case output side.
mod utils;

use divan::bench;
use locus_core::ImageView;
use locus_core::config::RescueInterpolation;
use locus_core::image::upscale_roi_to_buf;
use utils::BenchDataset;

fn main() {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();
    divan::Divan::from_args().threads([1]).run_benches();
}

fn setup(side: usize) -> (BenchDataset, (usize, usize, usize, usize)) {
    let dataset = BenchDataset::icra_forward_0();
    // Centred sub-rectangle of `side × side` — gives the kernel real
    // content to operate on without relying on synthetic patterns.
    let x = (dataset.width - side) / 2;
    let y = (dataset.height - side) / 2;
    (dataset, (x, y, side, side))
}

macro_rules! upscale_bench {
    ($name:ident, $side:expr, $factor:expr, $kernel:expr) => {
        #[bench]
        fn $name(bencher: divan::Bencher) {
            let (dataset, bbox) = setup($side);
            let (_, _, bw, bh) = bbox;
            let factor: u8 = $factor;
            let kernel: RescueInterpolation = $kernel;
            // +3 bytes of tail padding for SIMD gather loads (see .agent/rules/constraints.md §2).
            let out_len = bw * bh * (factor as usize).pow(2) + 3;
            let scratch_len = bw * bh * (factor as usize);
            let mut out = vec![0u8; out_len];
            let mut scratch = vec![0u8; scratch_len];
            bencher.bench_local(move || {
                let img = ImageView::new(
                    &dataset.raw_data,
                    dataset.width,
                    dataset.height,
                    dataset.width,
                )
                .unwrap();
                upscale_roi_to_buf(&img, bbox, &mut out, factor, &mut scratch, kernel)
                    .expect("upscale");
                divan::black_box(&out);
            });
        }
    };
}

upscale_bench!(
    bench_upscale_bilinear_32_2x,
    32,
    2,
    RescueInterpolation::Bilinear
);
upscale_bench!(
    bench_upscale_bilinear_64_2x,
    64,
    2,
    RescueInterpolation::Bilinear
);
upscale_bench!(
    bench_upscale_bilinear_128_2x,
    128,
    2,
    RescueInterpolation::Bilinear
);
upscale_bench!(
    bench_upscale_bilinear_32_4x,
    32,
    4,
    RescueInterpolation::Bilinear
);
upscale_bench!(
    bench_upscale_bilinear_64_4x,
    64,
    4,
    RescueInterpolation::Bilinear
);
upscale_bench!(
    bench_upscale_bilinear_128_4x,
    128,
    4,
    RescueInterpolation::Bilinear
);

upscale_bench!(
    bench_upscale_lanczos_32_2x,
    32,
    2,
    RescueInterpolation::Lanczos3
);
upscale_bench!(
    bench_upscale_lanczos_64_2x,
    64,
    2,
    RescueInterpolation::Lanczos3
);
upscale_bench!(
    bench_upscale_lanczos_128_2x,
    128,
    2,
    RescueInterpolation::Lanczos3
);
upscale_bench!(
    bench_upscale_lanczos_32_4x,
    32,
    4,
    RescueInterpolation::Lanczos3
);
upscale_bench!(
    bench_upscale_lanczos_64_4x,
    64,
    4,
    RescueInterpolation::Lanczos3
);
upscale_bench!(
    bench_upscale_lanczos_128_4x,
    128,
    4,
    RescueInterpolation::Lanczos3
);
