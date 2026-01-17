#!/usr/bin/env python3
"""Benchmark detector performance on checkerboard corner images."""

import locus
import cv2
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_ground_truth(root: Path):
    gt = defaultdict(dict)
    with open(root / "tags.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('tag_fully_visible', '1') != '1':
                continue
                
            img = Path(row['image']).name
            tid = int(row['tag_id'])
            x = float(row['ground_truth_x'])
            y = float(row['ground_truth_y'])
            # Just tracking presence for recall
            if tid not in gt[img]:
                gt[img][tid] = True
    return gt

def benchmark_checkerboard():
    root = Path("tests/data/icra2020/forward")
    gt = load_ground_truth(root)
    img_dir = root / "checkerboard_corners_images"
    
    detector = locus.Detector(
        segmentation_connectivity=locus.SegmentationConnectivity.Four,
        quad_min_area=8, # Relax area just in case
        upscale_factor=2,
    )
    
    total_tags = 0
    detected_tags = 0
    
    print(f"Benchmarking {len(gt)} images in checkerboard dataset...")
    
    for img_name in sorted(gt.keys()):
        img_path = img_dir / img_name
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        detections, stats = detector.detect_with_stats(img)
        
        det_ids = {d.id for d in detections}
        img_tags = len(gt[img_name])
        img_detected = 0
        
        for tid in gt[img_name]:
            if tid in det_ids:
                img_detected += 1
                
        total_tags += img_tags
        detected_tags += img_detected
        
        print(f"{img_name}: {img_detected}/{img_tags} ({stats.num_candidates} quads)")

    recall = (detected_tags / total_tags * 100) if total_tags > 0 else 0
    print(f"\n=== Results ===")
    print(f"Total Tags: {total_tags}")
    print(f"Detected: {detected_tags}")
    print(f"Recall: {recall:.2f}%")

if __name__ == "__main__":
    benchmark_checkerboard()
