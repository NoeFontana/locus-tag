#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]

use bumpalo::Bump;
use divan::bench;
use locus_core::segmentation::label_components_with_stats;

fn main() {
    divan::main();
}

#[bench]
fn bench_segmentation_1080p_empty(bencher: divan::Bencher) {
    let width = 1920;
    let height = 1080;
    let binary = vec![255u8; width * height];
    let mut arena = Bump::new();

    bencher.bench_local(|| {
        arena.reset();
        label_components_with_stats(&arena, &binary, width, height);
    });
}

#[bench]
fn bench_segmentation_1080p_checkered(bencher: divan::Bencher) {
    let width = 1920;
    let height = 1080;
    let mut binary = vec![255u8; width * height];

    // Create a 4x4 checkerboard to generate many runs
    for y in (0..height).step_by(8) {
        for x in (0..width).step_by(8) {
            if ((x / 8) + (y / 8)) % 2 == 0 {
                for dy in 0..4 {
                    if y + dy < height {
                        let row_off = (y + dy) * width;
                        for dx in 0..4 {
                            if x + dx < width {
                                binary[row_off + x + dx] = 0;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut arena = Bump::new();

    bencher.bench_local(|| {
        arena.reset();
        label_components_with_stats(&arena, &binary, width, height);
    });
}
