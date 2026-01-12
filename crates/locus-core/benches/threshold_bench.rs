#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]

use divan::bench;
use locus_core::image::ImageView;
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

fn generate_checkered(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![200u8; width * height];
    for y in (0..height).step_by(16) {
        for x in (0..width).step_by(16) {
            if ((x / 16) + (y / 16)) % 2 == 0 {
                for dy in 0..16 {
                    if y + dy < height {
                        let row_off = (y + dy) * width;
                        for dx in 0..16 {
                            if x + dx < width {
                                data[row_off + x + dx] = 50;
                            }
                        }
                    }
                }
            }
        }
    }
    data
}
