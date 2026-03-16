use locus_core::bench_api::{DetectionBatch, FunnelStatus, Point2f, TileStats, CandidateState};
use locus_core::image::ImageView;

#[test]
fn test_contrast_gate_rejection() {
    let mut batch = DetectionBatch::new();
    // Create a low-contrast quad (e.g., all pixels are around 128)
    batch.corners[0] = [
        Point2f { x: 10.0, y: 10.0 },
        Point2f { x: 20.0, y: 10.0 },
        Point2f { x: 20.0, y: 20.0 },
        Point2f { x: 10.0, y: 20.0 },
    ];
    batch.status_mask[0] = CandidateState::Active;
    
    let width = 32;
    let height = 32;
    // Uniform image (low contrast)
    let data = vec![128u8; width * height];
    let img = ImageView::new(&data, width, height, width).unwrap();
    
    // tile_stats with some range to trigger rejection
    // If range is 50, tau = 10. avg_contrast will be 0.
    let tile_stats = vec![TileStats { min: 100, max: 150 }; 64]; // 4x4 tiles of size 8
    
    locus_core::bench_api::apply_funnel_gate(&mut batch, 1, &img, &tile_stats, 4);
    
    assert_eq!(batch.funnel_status[0], FunnelStatus::RejectedContrast);
    assert_eq!(batch.status_mask[0], CandidateState::FailedDecode);
}

#[test]
fn test_contrast_gate_pass() {
    let mut batch = DetectionBatch::new();
    // Create a high-contrast quad
    batch.corners[0] = [
        Point2f { x: 10.0, y: 10.0 },
        Point2f { x: 20.0, y: 10.0 },
        Point2f { x: 20.0, y: 20.0 },
        Point2f { x: 10.0, y: 20.0 },
    ];
    batch.status_mask[0] = CandidateState::Active;
    
    let width = 32;
    let height = 32;
    let mut data = vec![0u8; width * height];
    // Fill interior of quad (roughly) with 255
    for y in 10..=20 {
        for x in 10..=20 {
            data[y * width + x] = 255;
        }
    }
    let img = ImageView::new(&data, width, height, width).unwrap();
    
    // tile_stats range = 255, tau = 51. avg_contrast should be ~255.
    let tile_stats = vec![TileStats { min: 0, max: 255 }; 64];
    
    locus_core::bench_api::apply_funnel_gate(&mut batch, 1, &img, &tile_stats, 4);
    
    assert_eq!(batch.funnel_status[0], FunnelStatus::PassedContrast);
    assert_eq!(batch.status_mask[0], CandidateState::Active);
}
