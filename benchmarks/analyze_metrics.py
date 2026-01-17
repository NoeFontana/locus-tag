#!/usr/bin/env python3
"""Analyze detection metrics (area, edge score, etc) for confirmed True Positives."""

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
            img = row['image']
            tid = int(row['tag_id'])
            corner = int(row['corner'])
            x = float(row['ground_truth_x'])
            y = float(row['ground_truth_y'])
            
            if tid not in gt[img]:
                gt[img][tid] = [None]*4
            gt[img][tid][corner] = (x, y)
    return gt

def analyze_dataset(name: str, path: Path):
    print(f"\nAnalyzing {name}...")
    gt_map = load_ground_truth(path)
    img_dir = path / "pure_tags_images"
    
    # Use current default settings
    detector = locus.Detector()
    
    metrics = {
        'area': [],
        'min_edge': [],
        'score': [],
        'contrast': [] # estimate from image
    }
    
    missed_metrics = {
        'min_edge': []
    }

    count = 0
    for img_name in sorted(gt_map.keys()):
        if count >= 50 and name == 'circle': break # Limit circle to save time
        
        img_path = img_dir / img_name
        if not img_path.exists(): continue
        
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        detections, _ = detector.detect_with_stats(img)
        
        det_map = {d.id: d for d in detections}
        
        for tid, corners in gt_map[img_name].items():
            if None in corners: continue
            
            # Compute GT size
            pts = np.array(corners)
            edges = [np.linalg.norm(pts[i]-pts[(i+1)%4]) for i in range(4)]
            min_edge = min(edges)
            
            if tid in det_map:
                # Get detection metrics
                # Note: Python bindings for Detection object might not expose internal score/area directly
                # but we can compute area/edge from corners.
                # Score is harder. Detector doesn't return per-quad score in Detection struct unless exposed.
                # We will just analyze the GEOMETRY of successful detections.
                
                det = det_map[tid]
                dpts = np.array(det.corners)
                
                # Area
                # Shoelace formula
                x = dpts[:,0]
                y = dpts[:,1]
                area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                
                # Min Edge
                dedges = [np.linalg.norm(dpts[i]-dpts[(i+1)%4]) for i in range(4)]
                d_min_edge = min(dedges)
                
                metrics['area'].append(area)
                metrics['min_edge'].append(d_min_edge)
                
            else:
                missed_metrics['min_edge'].append(min_edge)
        
        count += 1
        
    # Stats
    for k in metrics:
        if not metrics[k]: continue
        vals = np.array(metrics[k])
        print(f"  Detected {k}: Mean={vals.mean():.1f}, Min={vals.min():.1f}, P05={np.percentile(vals, 5):.1f}")
        
    if missed_metrics['min_edge']:
        vals = np.array(missed_metrics['min_edge'])
        print(f"  Missed MinEdge: Mean={vals.mean():.1f}, Min={vals.min():.1f}, P05={np.percentile(vals, 5):.1f}")

def main():
    root = Path("tests/data/icra2020")
    if (root / "forward").exists():
        analyze_dataset("forward", root / "forward")
    if (root / "circle").exists():
        analyze_dataset("circle", root / "circle")

if __name__ == "__main__":
    main()
