#![allow(missing_docs, clippy::unwrap_used)]
use divan::bench;
use locus_core::bench_api::{AprilTag36h11, Homography, TagDecoder};
use locus_core::ImageView;

fn main() {
    divan::main();
}

#[bench]
fn bench_decoding_200_candidates(bencher: divan::Bencher) {
    let canvas_size = 1000usize;
    let tag_size = 100usize;
    let family = locus_core::TagFamily::AprilTag36h11;
    let (data, corners) = locus_core::bench_api::generate_synthetic_test_image(
        family,
        0, // id
        tag_size,
        canvas_size,
        0.0f32, // noise
    );
    let img = ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();
    let decoder = AprilTag36h11;

    // Simulate 200 candidates by jittering the correct corners slightly
    let mut candidates = Vec::with_capacity(200);
    for i in 0..200 {
        let mut jittered = corners;
        let offset = (f64::from(i) * 0.01) % 0.5;
        for p in &mut jittered {
            p[0] += offset;
            p[1] += offset;
        }
        candidates.push(jittered);
    }

    bencher.bench_local(move || {
        let mut arena = bumpalo::Bump::new();
        let mut sum_ids = 0u32;
        for corners in &candidates {
            arena.reset();
            let cand = locus_core::Detection {
                corners: *corners,
                ..Default::default()
            };
            if let Some(bits) =
                locus_core::bench_api::sample_grid(&img, &arena, &cand, &decoder, 20.0)
                && let Some((id, _, _)) = decoder.decode(bits)
            {
                sum_ids += id;
            }
        }
        divan::black_box(sum_ids);
    });
}

#[bench]
fn bench_homography_200_only(bencher: divan::Bencher) {
    let mut candidates = Vec::with_capacity(200);
    for i in 0..200 {
        let x = f64::from(i % 10) * 50.0 + 100.0;
        let y = f64::from(i / 10) * 40.0 + 100.0;
        let corners = [[x, y], [x + 30.0, y], [x + 30.0, y + 30.0], [x, y + 30.0]];
        candidates.push(corners);
    }

    bencher.bench_local(move || {
        for corners in &candidates {
            let h = Homography::square_to_quad(corners);
            divan::black_box(h);
        }
    });
}
