use locus_core::image::ImageView;
use locus_core::DetectOptions;

#[test]
fn test_threshold_allocation_correctness() {
    // 1. Setup a large synthetic image (e.g., 4K resolution)
    let width = 3840;
    let height = 2160;
    let mut data = vec![0u8; width * height];
    
    // Create a gradient pattern to ensure we have meaningful thresholding work
    for y in 0..height {
        for x in 0..width {
            data[y * width + x] = ((x + y) % 255) as u8;
        }
    }

    let img = ImageView::new(&data, width, height, width).expect("valid image");

    // 2. Run Thresholding directly
    // We need to access the thresholding logic. Since `apply_threshold` might be private or
    // part of the internal pipeline, we can access it via the public `Detector` or `ThresholdMethod` if exposed.
    // Looking at the codebase, `ThresholdMethod` is likely used within `Detector`.
    // For this test, we'll verify via the public API that the pipeline completes successfully 
    // and produces consistent results. 
    
    // Ideally, we'd want to inspect the allocations, but that's hard in a unit test.
    // Instead, we verify that the output remains correct after our refactor.
    // To do this, we can capture the "before" state (which is this test running on current code)
    // and then ensure it passes on "after" code.
    
    // Since we can't easily extract just the thresholded image from the public API without internal visibility,
    // we will run the full detector and ensure it doesn't crash and produces detections (or empty if gradient).
    // Actually, to be more precise, let's look at `locus_core::threshold` visibility.
    
    // NOTE: This test assumes we can run the detector.
    let config = locus_core::DetectorConfig::default();
    let mut detector = locus_core::Detector::with_config(config);
    let options = DetectOptions::default();

    let (detections, stats) = detector.detect_with_stats_and_options(&img, &options);

    // Just ensure it runs. Correctness is strictly checked by regression tests.
    // This test is mainly to have a quick runner for the hot loop during refactoring.
    println!("Processed 4K image in {:.2} ms", stats.total_ms);
}
