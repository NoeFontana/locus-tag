use bumpalo::Bump;
use divan::bench;
use locus_core::filter::{bilateral_filter, compute_gradient_map};
use locus_core::image::ImageView;
use locus_core::threshold::{
    adaptive_threshold_gradient_window, adaptive_threshold_integral, compute_integral_image,
};

fn main() {
    divan::main();
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

#[bench]
fn bench_integral_image_1080p(bencher: divan::Bencher) {
    let width = 1920;
    let height = 1080;
    let data = generate_checkered(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let mut integral = vec![0u64; (width + 1) * (height + 1)];

    bencher.bench_local(move || {
        compute_integral_image(&img, &mut integral);
    });
}

#[bench]
fn bench_adaptive_integral_1080p(bencher: divan::Bencher) {
    let width = 1920;
    let height = 1080;
    let data = generate_checkered(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let mut integral = vec![0u64; (width + 1) * (height + 1)];
    compute_integral_image(&img, &mut integral);
    let mut output = vec![0u8; width * height];

    bencher.bench_local(move || {
        adaptive_threshold_integral(&img, &integral, &mut output, 6, 3);
    });
}

#[bench]
fn bench_adaptive_gradient_1080p(bencher: divan::Bencher) {
    let width = 1920;
    let height = 1080;
    let data = generate_checkered(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let mut integral = vec![0u64; (width + 1) * (height + 1)];
    compute_integral_image(&img, &mut integral);
    let mut gradient = vec![0u8; width * height];
    compute_gradient_map(&img, &mut gradient);
    let mut output = vec![0u8; width * height];

    bencher.bench_local(move || {
        adaptive_threshold_gradient_window(&img, &gradient, &integral, &mut output, 2, 7, 40, 3);
    });
}

#[bench]
fn bench_bilateral_r3_1080p(bencher: divan::Bencher) {
    let width = 1920;
    let height = 1080;
    let data = generate_checkered(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let mut output = vec![0u8; width * height];
    let arena = Bump::new();

    bencher.bench_local(move || {
        bilateral_filter(&arena, &img, &mut output, 3, 0.8, 30.0);
    });
}
