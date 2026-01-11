use divan::bench;
use locus_core::Detector;
use locus_core::image::ImageView;

fn main() {
    divan::main();
}

#[bench]
fn bench_full_detect_1080p(bencher: divan::Bencher) {
    let width = 1920;
    let height = 1080;
    let mut data = vec![200u8; width * height];

    // Add a dark square
    for y in 400..600 {
        for x in 800..1000 {
            data[y * width + x] = 30;
        }
    }

    let img = ImageView::new(&data, width, height, width).unwrap();
    let mut detector = Detector::new();

    bencher.bench_local(move || detector.detect(&img));
}
