use divan::bench;
use locus_core::Detector;
use locus_core::image::ImageView;

fn main() {
    divan::main();
}

/// Create a test image with a tag-like dark square on light background.
fn create_test_image(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![200u8; width * height];

    // Add a dark square with white border (tag-like)
    let cx = width / 2;
    let cy = height / 2;
    let size = 60;
    let border = 10;

    // White quiet zone
    for y in (cy - size - border)..(cy + size + border) {
        for x in (cx - size - border)..(cx + size + border) {
            if y < height && x < width {
                data[y * width + x] = 255;
            }
        }
    }

    // Dark square
    for y in (cy - size)..(cy + size) {
        for x in (cx - size)..(cx + size) {
            if y < height && x < width {
                data[y * width + x] = 30;
            }
        }
    }

    data
}

#[bench]
fn bench_detect_640x480(bencher: divan::Bencher) {
    let width = 640;
    let height = 480;
    let data = create_test_image(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let mut detector = Detector::new();

    bencher.bench_local(move || detector.detect(&img));
}

#[bench]
fn bench_detect_gradient_640x480(bencher: divan::Bencher) {
    let width = 640;
    let height = 480;
    let data = create_test_image(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let mut detector = Detector::new();

    bencher.bench_local(move || detector.detect_gradient(&img));
}

#[bench]
fn bench_detect_1080p(bencher: divan::Bencher) {
    let width = 1920;
    let height = 1080;
    let data = create_test_image(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let mut detector = Detector::new();

    bencher.bench_local(move || detector.detect(&img));
}

#[bench]
fn bench_detect_gradient_1080p(bencher: divan::Bencher) {
    let width = 1920;
    let height = 1080;
    let data = create_test_image(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let mut detector = Detector::new();

    bencher.bench_local(move || detector.detect_gradient(&img));
}
