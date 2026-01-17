#!/usr/bin/env python3
"""Diagnostic script to analyze forward dataset failures."""

import cv2
import numpy as np
import locus
from pathlib import Path
import csv
import json

def load_ground_truth(csv_path: Path):
    """Load ground truth tags."""
    gt = {}  # image -> tag_id -> corners
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
    return gt

def analyze_forward():
    root = Path("tests/data/icra2020/forward")
    img_dir = root / "pure_tags_images"
    gt = load_ground_truth(root / "tags.csv")
    
    # Use current settings
    detector = locus.Detector(
        quad_min_area=8,
        quad_max_aspect_ratio=20.0,
        quad_min_edge_score=0.3,
        threshold_min_range=2,
        enable_bilateral=False,
        enable_adaptive_window=False,
    )
    
    print(f"Analyzing {len(gt)} images...")
    
    missed_stats = {
        "total_missed": 0,
        "missed_by_size": [], # (avg_edge_len, image_name, tag_id)
        "quad_counts": [],
        "extraction_ms": [],
        "decoding_ms": [],
        "total_ms": []
    }
    
    for img_name in sorted(gt.keys()):
        img_path = img_dir / img_name
        if not img_path.exists():
            continue
            
        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Run detection
        detections, stats = detector.detect_with_stats(img)
        detected_ids = {d.id for d in detections}
        
        missed_tags = []
        for tid, corners in gt[img_name].items():
            if None in corners:
                continue
            
            if tid not in detected_ids:
                # Compute size
                pts = np.array(corners)
                edges = []
                for i in range(4):
                    p1 = pts[i]
                    p2 = pts[(i+1)%4]
                    d = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                    edges.append(d)
                avg_edge = np.mean(edges)
                
                missed_tags.append((tid, avg_edge))
                missed_stats["missed_by_size"].append((avg_edge, img_name, tid))
        
        # Check overlap with candidates
        candidates_poly = []
        for c in detections: # detections are NOT candidates, need logical candidates
            # The detector wrapper returns 'detections' (decoded) and 'stats'.
            # It implies candidates are NOT returned in python bindings usually?
            # actually detector.detect_with_stats returns (detections, stats)
            # but usually doesn't return ALL candidates unless we use a debug method?
            # Wait, stats.num_candidates is just a count.
            pass

        # We need all candidates to check overlap. 
        # The python binding detect_with_stats only returns finalized detections.
        # We need to rely on the fact that if it wasn't detected, it failed decode OR extraction.
        
        missed_stats["total_missed"] += len(missed_tags)
        missed_stats["quad_counts"].append(stats.num_candidates)
        missed_stats["extraction_ms"].append(stats.quad_extraction_ms)
        missed_stats["decoding_ms"].append(stats.decoding_ms)
        missed_stats["total_ms"].append(stats.total_ms)
        
        if len(missed_tags) > 0:
            print(f"{img_name}: {len(missed_tags)} missed tags")
            # Visualize first missed tag
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for tid, sz in missed_tags:
                corners = gt[img_name][tid]
                pts = np.array(corners, dtype=np.int32)
                cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
                cv2.putText(vis, f"{tid}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Save debug image for first failure and a few others
            if img_name == "0000.png" or len(missed_stats["missed_by_size"]) < 500:
                out_path = Path(f"/tmp/forward_missed_{img_name}")
                cv2.imwrite(str(out_path), vis)
                if img_name == "0000.png":
                    print(f"Saved visualization to {out_path}")
    
    # Analysis
    sizes = np.array([x[0] for x in missed_stats["missed_by_size"]])
    print("\n=== Missed Tags Analysis ===")
    print(f"Total Missed: {missed_stats['total_missed']}")
    if len(sizes) > 0:
        print(f"Missed Size Mean: {np.mean(sizes):.2f} px")
        print(f"Missed Size Min: {np.min(sizes):.2f} px")
        print(f"Missed Size Max: {np.max(sizes):.2f} px")
        print(f"Missed < 10px: {np.sum(sizes < 10)} ({np.mean(sizes < 10)*100:.1f}%)")
        print(f"Missed < 20px: {np.sum(sizes < 20)} ({np.mean(sizes < 20)*100:.1f}%)")
    
    print(f"Avg Quads per Image: {np.mean(missed_stats['quad_counts']):.1f}")
    
    print("\n=== Performance Analysis ===")
    print(f"Avg Total Time: {np.mean(missed_stats['total_ms']):.2f} ms")
    print(f"  Avg Quad Extraction: {np.mean(missed_stats['extraction_ms']):.2f} ms")
    print(f"  Avg Decoding: {np.mean(missed_stats['decoding_ms']):.2f} ms")
    print(f"  Avg Throughput: {1000.0 / np.mean(missed_stats['total_ms']):.2f} fps")

if __name__ == "__main__":
    analyze_forward()
