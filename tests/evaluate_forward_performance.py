#!/usr/bin/env python3
"""Evaluate Locus performance on Forward dataset with different profiles."""

import cv2
import numpy as np
import locus
import os
from pathlib import Path
import csv
import time
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Result:
    recall: float
    precision: float
    rmse: float
    avg_ms: float
    throughput: float
    num_detections: int
    num_gt: int

def load_ground_truth(csv_path: Path):
    """Load ground truth tags."""
    gt = {}  # image -> tag_id -> corners [p0, p1, p2, p3]
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('tag_fully_visible', '1') != '1':
                continue
            
            img = row['image']
            tid = int(row['tag_id'])
            corner = int(row['corner'])
            x = float(row['ground_truth_x'])
            y = float(row['ground_truth_y'])
            
            if img not in gt:
                gt[img] = {}
            if tid not in gt[img]:
                gt[img][tid] = [None, None, None, None]
            
            gt[img][tid][corner] = (x, y)
    
    # Filter out incomplete tags
    for img in gt:
        incomplete = [tid for tid, corners in gt[img].items() if None in corners]
        for tid in incomplete:
            del gt[img][tid]
            
    return gt

def match_detections(detections, ground_truth, dist_thresh=10.0):
    """Match detections to ground truth based on ID and distance."""
    matches = [] # (det, gt_corners)
    tp = 0
    fp = 0
    
    gt_matched = set()
    
    for det in detections:
        tid = det.id
        if tid in ground_truth:
            gt_corners = ground_truth[tid]
            # ID match. Check distance to center or corners
            # Compute detected center
            det_center = np.mean(det.corners, axis=0)
            gt_center = np.mean(gt_corners, axis=0)
            
            dist = np.linalg.norm(det_center - gt_center)
            if dist < dist_thresh:
                matches.append((det, gt_corners))
                gt_matched.add(tid)
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
            
    fn = len(ground_truth) - len(gt_matched)
    return tp, fp, fn, matches

def compute_rmse(matches):
    """Compute RMSE of corner locations."""
    if not matches:
        return 0.0
    
    sq_errors = []
    first = True
    for det, gt_corners in matches:
        det_corners = np.array(det.corners)
        gt_corners = np.array(gt_corners)
        
        # Check all rotations and mirrored rotations
        min_err_sum = float('inf')
        
        variants = [gt_corners]
        variants.append(gt_corners[::-1]) # Reverse
        
        for v in variants:
            for i in range(4):
                rotated = np.roll(v, i, axis=0)
                err = np.sum((det_corners - rotated)**2)
                if err < min_err_sum:
                    min_err_sum = err
        
        if first:
            # print(f"DEBUG: Det center: {np.mean(det_corners, axis=0)}, GT center: {np.mean(gt_corners, axis=0)}")
            # print(f"DEBUG: Min Error Sum: {min_err_sum}")
            first = False
            
        sq_errors.append(min_err_sum / 4.0)
    
    return np.sqrt(np.mean(sq_errors))

def evaluate_profile(name, detector, images_dir, gt_data):
    print(f"\nEvaluating Profile: {name}")
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_ms = []
    all_stats = []
    all_matches = []
    
    image_paths = sorted(list(images_dir.glob("*.png")))
    
    for img_path in image_paths:
        img_name = img_path.name
        if img_name not in gt_data:
            continue
            
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # Timing measurement
        detections, stats = detector.detect_with_stats(img)
        total_ms.append(stats.total_ms)
        all_stats.append(stats)
        
        tp, fp, fn, matches = match_detections(detections, gt_data[img_name])
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_matches.extend(matches)
        
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rmse = compute_rmse(all_matches)
    avg_ms = np.mean(total_ms)
    throughput = 1000.0 / avg_ms
    
    avg_thresh = np.mean([s.threshold_ms for s in all_stats])
    avg_seg = np.mean([s.segmentation_ms for s in all_stats])
    avg_quad = np.mean([s.quad_extraction_ms for s in all_stats])
    avg_dec = np.mean([s.decoding_ms for s in all_stats])

    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  RMSE:      {rmse:.4f} px")
    print(f"  Avg Time:  {avg_ms:.2f} ms (Thresh: {avg_thresh:.2f}, Seg: {avg_seg:.2f}, Quad: {avg_quad:.2f}, Dec: {avg_dec:.2f})")
    print(f"  Throughput:{throughput:.2f} fps")
    
    return Result(recall, precision, rmse, avg_ms, throughput, total_tp, total_tp + total_fn)

def main():
    # Use direct Rust detector to avoid Pydantic pickle issues
    from locus.locus import Detector as RustDetector
    
    root = Path("tests/data/icra2020/forward")
    img_dir = root / "pure_tags_images"
    gt_path = root / "tags.csv"
    
    if not img_dir.exists() or not gt_path.exists():
        print(f"Error: Dataset not found at {root}")
        return

    gt_data = load_ground_truth(gt_path)
    print(f"Loaded ground truth for {len(gt_data)} images.")
    
    # Standard Profile (Optimized for standard tags)
    standard_detector = RustDetector(
        upscale_factor=1,
        enable_sharpening=False,
        enable_bilateral=False,
    )
    
    # Checkerboard Profile (The one we've been tuning)
    # Using the class method we added
    from locus import Detector
    checkerboard_detector = Detector.checkerboard()
    # But wait, we need the Rust internal one if we want to bypass wrappers or just use the wrapper
    # The wrapper's detect_with_stats should work if it's correctly forwarding.
    
    # Let's try using the wrapper for checkerboard if it's clean
    try:
        from locus import Detector as WrapperDetector
        checkerboard_detector = WrapperDetector.checkerboard()
        # We need the internal detector for stats if the wrapper doesn't provide them nicely
        # Actually our wrapper's detect_with_stats returns (detections, stats)
    except:
        # Fallback to manual checkerboard config on Rust detector
        checkerboard_detector = RustDetector(
            enable_sharpening=True,
            enable_bilateral=False,
            quad_min_area=8,
            quad_min_edge_length=2.0,
            decoder_min_contrast=10.0,
        )

    results = {}
    results["Standard"] = evaluate_profile("Standard (1x)", standard_detector, img_dir, gt_data)
    results["Checkerboard"] = evaluate_profile("Checkerboard (1x)", checkerboard_detector, img_dir, gt_data)
    
    # results["Standard (2x)"] = evaluate_profile("Standard (2x)", standard_2x, img_dir, gt_data)

    print("\n" + "="*60)
    print(f"{'Profile':<20} | {'Recall':<8} | {'Precis':<8} | {'RMSE':<8} | {'Latency':<8}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20} | {r.recall:7.2%} | {r.precision:7.2%} | {r.rmse:8.4f} | {r.avg_ms:6.2f} ms")
    print("="*60)

if __name__ == "__main__":
    main()
