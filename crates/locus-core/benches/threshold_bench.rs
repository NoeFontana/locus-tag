#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]

use divan::bench;
use locus_core::image::ImageView;
use locus_core::test_utils::generate_checkered;
use locus_core::threshold::ThresholdEngine;

fn main() {
    divan::main();
}

#[bench]
fn bench_threshold_1080p_stats_checkered(bencher: divan::Bencher) {
    let width = 1920;
    let height = 1080;
    let data = generate_checkered(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let engine = ThresholdEngine::new();

    bencher.bench_local(move || engine.compute_tile_stats(&img));
}

#[bench]
fn bench_threshold_1080p_stats_checkered_subsampled(bencher: divan::Bencher) {
    let width = 1920;
    let height = 1080;
    let data = generate_checkered(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let ts = 8; // default

    bencher.bench_local(move || {
        let tiles_wide = width / ts;
        let tiles_high = height / ts;
        let mut stats =
            vec![locus_core::threshold::TileStats { min: 255, max: 0 }; tiles_wide * tiles_high];

        for ty in 0..tiles_high {
            let stats_row = &mut stats[ty * tiles_wide..(ty + 1) * tiles_wide];
            // Stride 2 in y
            for dy in (0..ts).step_by(2) {
                let py = ty * ts + dy;
                let src_row = img.get_row(py);

                // Manual stride 2 in x
                let chunks = src_row.chunks_exact(ts);
                for (chunk, stat) in chunks.zip(stats_row.iter_mut()) {
                    let mut rmin = 255u8;
                    let mut rmax = 0u8;
                    for i in (0..ts).step_by(2) {
                        let p = chunk[i];
                        rmin = rmin.min(p);
                        rmax = rmax.max(p);
                    }
                    stat.min = stat.min.min(rmin);
                    stat.max = stat.max.max(rmax);
                }
            }
        }
    });
}

#[bench]
fn bench_threshold_1080p_apply_checkered(bencher: divan::Bencher) {
    let width = 1920;
    let height = 1080;
    let data = generate_checkered(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let engine = ThresholdEngine::new();
    let stats = engine.compute_tile_stats(&img);
    let mut output = vec![0u8; width * height];

    bencher.bench_local(move || {
        engine.apply_threshold(&img, &stats, &mut output);
    });
}
