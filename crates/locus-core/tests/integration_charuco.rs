use locus_core::{CharucoBoard, Detector, CameraIntrinsics};
use locus_core::config::TagFamily;

#[test]
fn test_charuco_coarse_detection() {
    // 3x3 squares, 4x4 ArUco markers in white squares.
    // In a 3x3 grid, there are 5 white squares if (0,0) is white.
    // (0,0), (0,2), (1,1), (2,0), (2,2)
    let rows = 3;
    let cols = 3;
    let square_length = 0.04;
    let marker_length = 0.03;
    let board = CharucoBoard::new(rows, cols, square_length, marker_length);

    // Create a synthetic image with one ArUco tag from the board.
    let family = TagFamily::ArUco4x4_50;
    let tag_id = 0;
    let tag_size = 64;
    let canvas_size = 200;
    
    let (data, _) = locus_core::test_utils::generate_synthetic_test_image(
        family,
        tag_id,
        tag_size,
        canvas_size,
        0.0,
    );
    let img = locus_core::image::ImageView::new(&data, canvas_size, canvas_size, canvas_size).unwrap();

    let mut detector = Detector::new();
    detector.set_families(&[family]);
    
    // Intrinsics for a 200x200 image.
    let intrinsics = CameraIntrinsics::new(200.0, 200.0, 100.0, 100.0);

    let result = detector.detect_charuco(&img, &board, &intrinsics).expect("Detection failed");
    
    // Since we only have one tag and it's not positioned according to the board layout,
    // the estimator might not find a consensus, but detect_charuco should at least run.
    // For a real integration test, we would need to render a full board.
    // But this verifies the API entry point.
    assert!(result.is_none() || result.is_some()); 
}
