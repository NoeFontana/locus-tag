"""
Tune checkerboard config to achieve 80% recall.
Test various parameter combinations with upscale_factor=2.
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

def test_config(images_dir, gt, config_name, **kwargs):
    """Test a specific config and return recall."""
    detector = locus.Detector(**kwargs)
    
    total_detected = 0
    total_gt = 0
    
    for img_path in sorted(images_dir.glob("*.png"))[:50]:  # Full dataset
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        detections, _ = detector.detect_with_stats(img)
        detected_ids = {d.id for d in detections}
        gt_ids = gt.get(img_path.name, set())
        
        total_detected += len(detected_ids & gt_ids)
        total_gt += len(gt_ids)
    
    recall = (total_detected / total_gt * 100) if total_gt > 0 else 0
    print(f"  {config_name}: {total_detected}/{total_gt} ({recall:.1f}%)")
    return recall

def main():
    root = Path("tests/data/icra2020/forward/checkerboard_corners_images")
    csv_path = Path("tests/data/icra2020/forward/tags.csv")
    gt = load_ground_truth(csv_path)
    
    print("=== Checkerboard Config Tuning ===\n")
    
    # 1x configurations - no upscaling
    configs = [
        ("Baseline 1x 4-way", dict(
            segmentation_connectivity=locus.SegmentationConnectivity.Four,
        )),
        ("Low edge score (0.2)", dict(
            segmentation_connectivity=locus.SegmentationConnectivity.Four,
            quad_min_edge_score=0.2,
        )),
        ("Low edge score (0.1)", dict(
            segmentation_connectivity=locus.SegmentationConnectivity.Four,
            quad_min_edge_score=0.1,
        )),
        ("Small area (4)", dict(
            segmentation_connectivity=locus.SegmentationConnectivity.Four,
            quad_min_area=4,
        )),
        ("Small tile (2)", dict(
            segmentation_connectivity=locus.SegmentationConnectivity.Four,
            threshold_tile_size=2,
        )),
        ("Low min_range (1)", dict(
            segmentation_connectivity=locus.SegmentationConnectivity.Four,
            threshold_min_range=1,
        )),
        ("Low contrast (5)", dict(
            segmentation_connectivity=locus.SegmentationConnectivity.Four,
            decoder_min_contrast=5.0,
        )),
        ("Aggressive combo", dict(
            segmentation_connectivity=locus.SegmentationConnectivity.Four,
            quad_min_edge_score=0.1,
            quad_min_area=4,
            threshold_tile_size=2,
            threshold_min_range=1,
            decoder_min_contrast=5.0,
        )),
    ]
    
    results = []
    for name, cfg in configs:
        recall = test_config(root, gt, name, **cfg)
        results.append((name, recall))
    
    print("\n=== Summary ===")
    best = max(results, key=lambda x: x[1])
    print(f"Best config: {best[0]} ({best[1]:.1f}%)")

if __name__ == "__main__":
    main()
