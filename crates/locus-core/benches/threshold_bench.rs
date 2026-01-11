#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]

use divan::bench;
use locus_core::image::ImageView;
use locus_core::threshold::ThresholdEngine;

fn main() {
    divan::main();
}

#[bench]
fn bench_threshold_1080p(bencher: divan::Bencher) {
    let width = 1920;
    let height = 1080;
    let data = vec![128u8; width * height];
    let img = ImageView::new(&data, width, height, width).unwrap();
    let engine = ThresholdEngine::new();
    let mut output = vec![0u8; width * height];

    bencher.bench_local(move || {
        let stats = engine.compute_tile_stats(&img);
        engine.apply_threshold(&img, &stats, &mut output);
    });
}
