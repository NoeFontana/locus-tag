"""
Tune sharpened 1x config to reach 80% recall.
"""
import locus
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_ground_truth(csv_path: Path) -> dict:
    """Load ground truth from tags.csv."""
    gt = defaultdict(set)
    with open(csv_path) as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                img_name = parts[0]
                tag_id = int(parts[1])
                gt[img_name].add(tag_id)
    return gt

def apply_sharpening(img):
    # Standard 3x3 sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def apply_aggressive_sharpening(img):
    # More aggressive sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def test_config(images_dir, gt, name, preprocess_fn=None, **kwargs):
    """Test a specific config and return recall."""
    detector = locus.Detector(**kwargs)
    
    total_detected = 0
    total_gt = 0
    
    img_files = sorted(images_dir.glob("*.png"))
    
    for img_path in img_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        if preprocess_fn:
            img = preprocess_fn(img)
            
        detections = detector.detect(img)
        detected_ids = {d.id for d in detections}
        gt_ids = gt.get(img_path.name, set())
        
        total_detected += len(detected_ids & gt_ids)
        total_gt += len(gt_ids)
    
    recall = (total_detected / total_gt * 100) if total_gt > 0 else 0
    print(f"  {name}: {total_detected}/{total_gt} ({recall:.2f}%)")
    return recall

def main():
    root = Path("tests/data/icra2020/forward/checkerboard_corners_images")
    csv_path = Path("tests/data/icra2020/forward/tags.csv")
    gt = load_ground_truth(csv_path)
    
    print("=== Sharpened 1x Config Tuning ===\n")
    
    # 1x configurations with sharpening
    configs = [
        ("Sharpening Baseline", apply_sharpening, dict(
            segmentation_connectivity=locus.SegmentationConnectivity.Four,
        )),
        ("Aggressive Sharpening", apply_aggressive_sharpening, dict(
            segmentation_connectivity=locus.SegmentationConnectivity.Four,
        )),
        ("Sharpening + Low edge score (0.2)", apply_sharpening, dict(
            segmentation_connectivity=locus.SegmentationConnectivity.Four,
            quad_min_edge_score=0.2,
        )),
        ("Sharpening + Low contrast (10)", apply_sharpening, dict(
            segmentation_connectivity=locus.SegmentationConnectivity.Four,
            decoder_min_contrast=10.0,
        )),
        ("Sharpening + Low contrast (5)", apply_sharpening, dict(
            segmentation_connectivity=locus.SegmentationConnectivity.Four,
            decoder_min_contrast=5.0,
        )),
        ("Sharpening + Small area (4)", apply_sharpening, dict(
            segmentation_connectivity=locus.SegmentationConnectivity.Four,
            quad_min_area=4,
        )),
        ("Sharpening + Aggressive Combo", apply_sharpening, dict(
            segmentation_connectivity=locus.SegmentationConnectivity.Four,
            quad_min_edge_score=0.1,
            quad_min_area=4,
            decoder_min_contrast=5.0,
        )),
    ]
    
    for name, proc, cfg in configs:
        test_config(root, gt, name, preprocess_fn=proc, **cfg)

if __name__ == "__main__":
    main()
