"""
Final push for 80% recall: Combined Upscaling and Sharpening.
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
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
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
    
    print("=== Final Verification (Target: 80% Recall) ===\n")
    
    # Combined config
    cfg = dict(
        segmentation_connectivity=locus.SegmentationConnectivity.Four,
        upscale_factor=2,
        quad_min_edge_score=0.2, # Scaled for upscale
    )
    
    # Test cases
    test_config(root, gt, "Upscale 2x + Sharpening", preprocess_fn=apply_sharpening, **cfg)
    test_config(root, gt, "Upscale 2x only (for comparison)", preprocess_fn=None, **cfg)

if __name__ == "__main__":
    main()
