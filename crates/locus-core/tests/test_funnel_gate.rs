//! Tests for the fast-path decoding funnel gate.
use locus_core::bench_api::{CandidateState, DetectionBatch, FunnelStatus, Point2f, TileStats};
use locus_core::image::ImageView;

#[test]
fn test_contrast_gate_rejection() {
    let mut batch = DetectionBatch::new();
    // Create a low-contrast quad (e.g., all pixels are around 128)
    batch.corners[0] = [
        Point2f { x: 5.0, y: 5.0 },
        Point2f { x: 25.0, y: 5.0 },
        Point2f { x: 25.0, y: 25.0 },
        Point2f { x: 5.0, y: 25.0 },
    ];
    batch.status_mask[0] = CandidateState::Active;

    let width = 32;
    let height = 32;
    // Uniform image (low contrast)
    let data = vec![128u8; width * height];
    let img = ImageView::new(&data, width, height, width).expect("valid image");

    // tile_stats with some range to trigger rejection
    // If range is 50, tau = 10. avg_contrast will be 0.
    let tile_stats = vec![TileStats { min: 100, max: 150 }; 64]; // 4x4 tiles of size 8

    locus_core::bench_api::apply_funnel_gate(&mut batch, 1, &img, &tile_stats, 4, 20.0, 1.0);

    assert_eq!(batch.funnel_status[0], FunnelStatus::RejectedContrast);
    assert_eq!(batch.status_mask[0], CandidateState::FailedDecode);
}

#[test]
fn test_contrast_gate_pass() {
    let mut batch = DetectionBatch::new();
    // Create a high-contrast quad
    batch.corners[0] = [
        Point2f { x: 5.0, y: 5.0 },
        Point2f { x: 25.0, y: 5.0 },
        Point2f { x: 25.0, y: 25.0 },
        Point2f { x: 5.0, y: 25.0 },
    ];
    batch.status_mask[0] = CandidateState::Active;

    let width = 32;
    let height = 32;
    let mut data = vec![0u8; width * height];
    // Fill interior of quad (roughly) with 255.
    // Quad is [5,5] to [25,25]. Midpoints are (15,5), (25,15), (15,25), (5,15).
    // Sample points are mx +/- 2*nx.
    // e.g. for top edge (15,5), normal is (0,1), sample points are (15,7) [IN] and (15,3) [OUT].
    for y in 7..=23 {
        for x in 7..=23 {
            data[y * width + x] = 255;
        }
    }
    let img = ImageView::new(&data, width, height, width).expect("valid image");

    // tile_stats range = 255, tau = 51. avg_contrast should be ~255.
    let tile_stats = vec![TileStats { min: 0, max: 255 }; 64];

    locus_core::bench_api::apply_funnel_gate(&mut batch, 1, &img, &tile_stats, 4, 20.0, 1.0);

    assert_eq!(batch.funnel_status[0], FunnelStatus::PassedContrast);
    assert_eq!(batch.status_mask[0], CandidateState::Active);
}
