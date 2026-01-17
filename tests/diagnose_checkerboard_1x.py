"""
Diagnose why checkerboard detection fails at 1x resolution.
Analyze: Segmentation, Quad Extraction, Decoding stages.
"""
import locus
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

def diagnose_image(img_path: Path, ground_truth_ids: set):
    """Diagnose a single checkerboard image."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load {img_path}")
        return
    
    print(f"\n=== {img_path.name} ===")
    print(f"Image size: {img.shape}")
    print(f"Ground truth tags: {len(ground_truth_ids)}")
    
    # Test with various configurations
    configs = [
        ("Default 4-way", dict(segmentation_connectivity=locus.SegmentationConnectivity.Four)),
        ("Low Contrast (10)", dict(segmentation_connectivity=locus.SegmentationConnectivity.Four, decoder_min_contrast=10.0)),
        ("Low Contrast (5)", dict(segmentation_connectivity=locus.SegmentationConnectivity.Four, decoder_min_contrast=5.0)),
        ("Low Contrast (2)", dict(segmentation_connectivity=locus.SegmentationConnectivity.Four, decoder_min_contrast=2.0)),
        ("Combined (Contrast=5, Area=4)", dict(
            segmentation_connectivity=locus.SegmentationConnectivity.Four,
            decoder_min_contrast=5.0,
            quad_min_area=4,
        )),
    ]
    
    for name, cfg in configs:
        detector = locus.Detector(**cfg)
        detections, stats = detector.detect_with_stats(img)
        detected_ids = {d.id for d in detections}
        
        recall = len(detected_ids & ground_truth_ids) / len(ground_truth_ids) * 100
        print(f"  {name}: {len(detections)}/{len(ground_truth_ids)} detected ({recall:.1f}%), {stats.num_candidates} candidates")

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

def main():
    root = Path("tests/data/icra2020/forward/checkerboard_corners_images")
    csv_path = Path("tests/data/icra2020/forward/tags.csv")
    
    if not csv_path.exists():
        print(f"Ground truth not found: {csv_path}")
        return
    
    gt = load_ground_truth(csv_path)
    
    # Diagnose first few images (far distance = hardest)
    for img_name in ["0000.png", "0001.png", "0010.png", "0024.png"]:
        img_path = root / img_name
        if img_path.exists():
            diagnose_image(img_path, gt.get(img_name, set()))

if __name__ == "__main__":
    main()
