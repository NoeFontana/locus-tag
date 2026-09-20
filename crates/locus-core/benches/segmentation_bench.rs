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
mod utils;

use bumpalo::Bump;
use divan::bench;
use locus_core::ImageView;
use locus_core::bench_api::ThresholdEngine;
use utils::BenchDataset;

fn main() {
    // Force rayon to a single thread for microbenchmarks to avoid cache thrashing.
    // `RAYON_NUM_THREADS`, when set, overrides that: it is how the segmentation
    // thread-scaling benches at the bottom of this file are swept (the pool must
    // be process-global because `bumpalo::Bump` is `!Sync`, so a scoped
    // `ThreadPool::install` around the call is not expressible).
    let threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(1);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();

    divan::Divan::from_args().threads([1]).run_benches();
}

#[bench]
fn bench_segmentation_real_icra_threshold_model(bencher: divan::Bencher) {
    let dataset = BenchDataset::icra_forward_0();
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();
    let setup_arena = Bump::new();
    let config = locus_core::DetectorConfig::default();
    let engine = ThresholdEngine::from_config(&config);

    let tile_stats = engine.compute_tile_stats(&setup_arena, &img);
    let mut threshold_map = vec![0u8; dataset.width * dataset.height];
    let mut binarized = vec![0u8; dataset.width * dataset.height];

    engine.apply_threshold_with_map(
        &setup_arena,
        &img,
        &tile_stats,
        &mut binarized,
        &mut threshold_map,
    );

    bencher.bench_local(move || {
        let arena = Bump::new();
        let _label_result =
            locus_core::bench_api::label_components_lsl(&arena, &img, &threshold_map, true, 16);
    });
}

#[bench]
fn bench_segmentation_real_icra_threshold_model_1080p(bencher: divan::Bencher) {
    let dataset = BenchDataset::load_and_resize_icra_frame("forward", 0, 1920, 1080);
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();
    let setup_arena = Bump::new();
    let config = locus_core::DetectorConfig::default();
    let engine = ThresholdEngine::from_config(&config);

    let tile_stats = engine.compute_tile_stats(&setup_arena, &img);
    let mut threshold_map = vec![0u8; dataset.width * dataset.height];
    let mut binarized = vec![0u8; dataset.width * dataset.height];

    engine.apply_threshold_with_map(
        &setup_arena,
        &img,
        &tile_stats,
        &mut binarized,
        &mut threshold_map,
    );

    bencher.bench_local(move || {
        let arena = Bump::new();
        let _label_result =
            locus_core::bench_api::label_components_lsl(&arena, &img, &threshold_map, true, 16);
    });
}

// ---------------------------------------------------------------------------
// Thread-scaling benches (Liu4K-like textured 4K frames)
//
// The three `bench_segmentation_real_*` benches run on the process-global rayon
// pool, which `main` pins to one thread; they are the "must not regress" gate
// for the single-threaded path. The benches at the bottom of this file install
// an explicit local pool so the 1-thread and N-thread paths can be compared in
// the same run, independently of `RAYON_NUM_THREADS`.
// ---------------------------------------------------------------------------

/// Deterministic stand-in for a Liu4K frame: a low-key, high-frequency textured
/// background (the "water ripple" regime that dominates Liu4K) with a handful of
/// large black-bordered markers on top.
///
/// The texture is what makes Liu4K expensive for CCL: after tile thresholding it
/// produces a speckled foreground with ~10^6 RLE runs per 4K frame, versus ~10^5
/// for the (mostly flat) ICRA frames. No RNG crate is used: the pixel value is a
/// pure function of (x, y), so the frame is byte-reproducible everywhere.
fn liu4k_like_frame(width: usize, height: usize) -> Vec<u8> {
    let mut data = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            // Cheap integer hash -> high-frequency speckle.
            let h = (x as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
            let h = (h ^ (h >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            let noise = ((h >> 33) & 0x3F) as u8; // 0..63

            // Low-frequency lighting ramp keeps the local tile range small and
            // the absolute level low (the Liu4K "low key" regime).
            let ramp = 24 + u8::try_from((x / 97 + y / 61) % 40).unwrap();

            data[y * width + x] = noise / 2 + ramp;
        }
    }

    // Overlay 12 markers with a thick flat-black border and a bright interior.
    let edge = width / 12;
    for k in 0..12usize {
        let ox = (k % 4) * (width / 4) + width / 32;
        let oy = (k / 4) * (height / 3) + height / 24;
        let border = edge / 8;
        for dy in 0..edge {
            for dx in 0..edge {
                let (px, py) = (ox + dx, oy + dy);
                if px >= width || py >= height {
                    continue;
                }
                let inner =
                    dx >= border && dy >= border && dx < edge - border && dy < edge - border;
                data[py * width + px] = if inner { 200 } else { 4 };
            }
        }
    }

    data
}

/// An image plus its precomputed threshold map, so the benches time CCL alone.
struct SegInput {
    raw: Vec<u8>,
    width: usize,
    height: usize,
    threshold_map: Vec<u8>,
}

impl SegInput {
    fn new(raw: Vec<u8>, width: usize, height: usize) -> Self {
        let setup_arena = Bump::new();
        let config = locus_core::DetectorConfig::default();
        let engine = ThresholdEngine::from_config(&config);
        let mut threshold_map = vec![0u8; width * height];
        let mut binarized = vec![0u8; width * height];
        {
            let img = ImageView::new(&raw, width, height, width).unwrap();
            let tile_stats = engine.compute_tile_stats(&setup_arena, &img);
            engine.apply_threshold_with_map(
                &setup_arena,
                &img,
                &tile_stats,
                &mut binarized,
                &mut threshold_map,
            );
        }
        Self {
            raw,
            width,
            height,
            threshold_map,
        }
    }
}

/// Times `label_components_lsl` against a warm (reused) arena — the state the
/// detector hot loop is in. The rayon pool width comes from `RAYON_NUM_THREADS`
/// (see `main`).
fn bench_seg(bencher: divan::Bencher, input: &SegInput) {
    let img = ImageView::new(&input.raw, input.width, input.height, input.width).unwrap();
    let mut arena = Bump::new();
    // Warm the arena so the measurement is not dominated by first-touch page
    // faults on the 33 MB label buffer (the detector reuses one arena per frame).
    {
        let r = locus_core::bench_api::label_components_lsl(
            &arena,
            &img,
            &input.threshold_map,
            true,
            16,
        );
        divan::black_box(r.component_stats.len());
    }
    bencher.bench_local(move || {
        arena.reset();
        let result = locus_core::bench_api::label_components_lsl(
            &arena,
            &img,
            &input.threshold_map,
            true,
            16,
        );
        divan::black_box(result.component_stats.len());
        divan::black_box(result.labels.len());
    });
}

#[bench]
fn bench_segmentation_textured_4k(bencher: divan::Bencher) {
    let input = SegInput::new(liu4k_like_frame(3840, 2160), 3840, 2160);
    bench_seg(bencher, &input);
}

#[bench]
fn bench_segmentation_textured_1080p(bencher: divan::Bencher) {
    let input = SegInput::new(liu4k_like_frame(1920, 1080), 1920, 1080);
    bench_seg(bencher, &input);
}

#[bench]
fn bench_segmentation_icra_4k(bencher: divan::Bencher) {
    let dataset = BenchDataset::load_and_resize_icra_frame("forward", 0, 3840, 2160);
    let input = SegInput::new(dataset.raw_data, dataset.width, dataset.height);
    bench_seg(bencher, &input);
}

#[bench]
fn bench_segmentation_icra_native(bencher: divan::Bencher) {
    let dataset = BenchDataset::icra_forward_0();
    let input = SegInput::new(dataset.raw_data, dataset.width, dataset.height);
    bench_seg(bencher, &input);
}

#[bench]
fn bench_segmentation_real_icra_threshold_model_4k(bencher: divan::Bencher) {
    let dataset = BenchDataset::load_and_resize_icra_frame("forward", 0, 3840, 2160);
    let img = ImageView::new(
        &dataset.raw_data,
        dataset.width,
        dataset.height,
        dataset.width,
    )
    .unwrap();
    let setup_arena = Bump::new();
    let config = locus_core::DetectorConfig::default();
    let engine = ThresholdEngine::from_config(&config);

    let tile_stats = engine.compute_tile_stats(&setup_arena, &img);
    let mut threshold_map = vec![0u8; dataset.width * dataset.height];
    let mut binarized = vec![0u8; dataset.width * dataset.height];

    engine.apply_threshold_with_map(
        &setup_arena,
        &img,
        &tile_stats,
        &mut binarized,
        &mut threshold_map,
    );

    bencher.bench_local(move || {
        let arena = Bump::new();
        let _label_result =
            locus_core::bench_api::label_components_lsl(&arena, &img, &threshold_map, true, 16);
    });
}
