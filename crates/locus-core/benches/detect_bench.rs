#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::uninlined_format_args)]

use bumpalo::Bump;
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
fn bench_thresholding_640x480(bencher: divan::Bencher) {
    let width = 640;
    let height = 480;
    let data = create_test_image(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let engine = locus_core::threshold::ThresholdEngine::new();
    let mut output = vec![0u8; width * height];

    bencher.bench_local(move || {
        let stats = engine.compute_tile_stats(&img);
        engine.apply_threshold(&img, &stats, &mut output);
    });
}

#[bench]
fn bench_segmentation_640x480(bencher: divan::Bencher) {
    let width = 640;
    let height = 480;
    let data = create_test_image(width, height);
    let mut binarized = vec![0u8; width * height];
    let engine = locus_core::threshold::ThresholdEngine::new();
    let img = ImageView::new(&data, width, height, width).unwrap();
    let stats = engine.compute_tile_stats(&img);
    engine.apply_threshold(&img, &stats, &mut binarized);

    bencher.bench_local(move || {
        // We can't easily reset an arena across bench iterations if it's moved
        // but for segmentation we only care about the time to label.
        // In a real app we reset every frame.
        let local_arena = Bump::new();
        locus_core::segmentation::label_components(&local_arena, &binarized, width, height);
    });
}

#[bench]
fn bench_quad_extraction_640x480(bencher: divan::Bencher) {
    let width = 640;
    let height = 480;
    let data = create_test_image(width, height);
    let arena = Bump::new();
    let mut binarized = vec![0u8; width * height];
    let engine = locus_core::threshold::ThresholdEngine::new();
    let img = ImageView::new(&data, width, height, width).unwrap();
    let stats = engine.compute_tile_stats(&img);
    engine.apply_threshold(&img, &stats, &mut binarized);

    // Pre-label components. Note: labels refer to data in 'arena'.
    let labels = locus_core::segmentation::label_components(&arena, &binarized, width, height);

    bencher.bench_local(move || {
        // Quad extraction needs an arena to store quads.
        // It doesn't modify labels.
        let local_arena = Bump::new();
        locus_core::quad::extract_quads(&local_arena, &img, labels);
    });
}

#[bench]
fn bench_full_detect_640x480(bencher: divan::Bencher) {
    let width = 640;
    let height = 480;
    let data = create_test_image(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let mut detector = Detector::new();

    bencher.bench_local(move || detector.detect(&img));
}

#[bench]
fn bench_full_detect_gradient_640x480(bencher: divan::Bencher) {
    let width = 640;
    let height = 480;
    let data = create_test_image(width, height);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let mut detector = Detector::new();

    bencher.bench_local(move || detector.detect_gradient(&img));
}

#[bench]
fn bench_accuracy_640x480(bencher: divan::Bencher) {
    let width = 640;
    let height = 480;
    let (data, gt_corners) =
        locus_core::test_utils::generate_synthetic_tag(width, height, 0, 100, 100, 120);
    let img = ImageView::new(&data, width, height, width).unwrap();
    let mut detector = Detector::new();

    bencher.bench_local(move || {
        let detections = detector.detect(&img);

        // We don't assert here to avoid failing the benchmark, but we measure
        if !detections.is_empty() {
            let err =
                locus_core::test_utils::compute_corner_error(detections[0].corners, gt_corners);
            // This is a bit of a hack to report accuracy in benchmarks
            // In a better setup we'd use a custom reporter
            divan::black_box(err);
        }
    });
}
