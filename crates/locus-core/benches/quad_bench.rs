#![allow(missing_docs)]

use bumpalo::Bump;
use divan::bench;
use locus_core::config::DetectorConfig;
use locus_core::image::ImageView;
use locus_core::segmentation::ComponentStats;
use locus_core::test_utils::{SceneBuilder, TagPlacement};
use locus_core::config::TagFamily;

fn main() {
    divan::main();
}

#[bench]
fn bench_extract_single_quad(bencher: divan::Bencher) {
    let width = 640;
    let height = 480;
    
    // 1. Create a scene with a single tag
    let mut builder = SceneBuilder::new(width, height);
    builder.add_tag(TagPlacement {
        family: TagFamily::AprilTag36h11,
        id: 0,
        center_x: width as f64 / 2.0,
        center_y: height as f64 / 2.0,
        size: 100.0,
        rotation_rad: 0.0,
    });
    let (data, _) = builder.build();
    let img = ImageView::new(&data, width, height, width).unwrap();
    
    // 2. Threshold and segment to get the labels and stats
    let arena = Bump::new();
    let engine = locus_core::threshold::ThresholdEngine::new();
    let mut binarized = vec![0u8; width * height];
    let tile_stats = engine.compute_tile_stats(&arena, &img);
    engine.apply_threshold(&arena, &img, &tile_stats, &mut binarized);
    
    let label_result = locus_core::segmentation::label_components_with_stats(
        &arena, &binarized, width, height, true,
    );
    
    // Find a valid component stat that has enough pixels (likely the tag border)
    let (label_idx, stat) = label_result.component_stats.iter().enumerate()
        .find(|(_, s)| s.pixel_count > 100 && s.pixel_count < 20000)
        .expect("Should find at least one valid component");
        
    let config = DetectorConfig::default();
    let label = (label_idx + 1) as u32;

    bencher.bench_local(move || {
        let local_arena = Bump::new();
        locus_core::quad::extract_single_quad(
            &local_arena,
            &img,
            label_result.labels,
            label,
            stat,
            &config,
            1,
            &img,
        )
    });
}
