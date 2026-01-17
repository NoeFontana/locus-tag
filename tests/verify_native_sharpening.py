"""
Verify native Laplacian sharpening recall boost on 1x checkerboard.
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

def test_config(images_dir, gt, name, **kwargs):
    """Test a specific config and return recall."""
    detector = locus.Detector(**kwargs)
    print(f"DEBUG: Python-side detector.enable_sharpening = {detector.enable_sharpening}")
    
    total_detected = 0
    total_gt = 0
    
    img_files = sorted(images_dir.glob("*.png"))
    
    for img_path in img_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
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
    
    print("=== Native Sharpening Verification (1x) ===\n")
    
    # Bilateral filter is enabled by default in locus-py, disable it for clean sharpening test
    base_cfg = dict(
        segmentation_connectivity=locus.SegmentationConnectivity.Four,
        enable_bilateral=False,
    )
    
    # Test cases
    test_config(root, gt, "Baseline (1x, No Sharp)", enable_sharpening=False, **base_cfg)
    test_config(root, gt, "Native Sharpening (1x)", enable_sharpening=True, **base_cfg)
    test_config(root, gt, "Native Sharpening + Low Contrast (10)", enable_sharpening=True, decoder_min_contrast=10.0, **base_cfg)

if __name__ == "__main__":
    main()
