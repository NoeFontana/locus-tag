#!/usr/bin/env python3
"""Analyze tag sizes in forward and circle datasets."""

import csv
from pathlib import Path
import numpy as np

def analyze_tag_sizes(csv_path: Path, dataset_name: str):
    """Compute tag sizes from corner coordinates."""
    tag_data = {}  # image -> tag_id -> corners
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row['image']
            tid = int(row['tag_id'])
            corner = int(row['corner'])
            x = float(row['ground_truth_x'])
            y = float(row['ground_truth_y'])
            visible = row.get('tag_fully_visible', '1') == '1'
            
            if not visible:
                continue
            
            key = (img, tid)
            if key not in tag_data:
                tag_data[key] = [None, None, None, None]
            tag_data[key][corner] = (x, y)
    
    # Compute sizes
    sizes = []
    for key, corners in tag_data.items():
        if None in corners:
            continue
        
        # Compute edge lengths
        edges = []
        for i in range(4):
            dx = corners[(i+1)%4][0] - corners[i][0]
            dy = corners[(i+1)%4][1] - corners[i][1]
            edges.append(np.sqrt(dx*dx + dy*dy))
        
        avg_edge = np.mean(edges)
        min_edge = np.min(edges)
        sizes.append((avg_edge, min_edge))
    
    if not sizes:
        print(f"{dataset_name}: No valid tags found")
        return
    
    sizes = np.array(sizes)
    print(f"\n=== {dataset_name} ===")
    print(f"  Total valid tags: {len(sizes)}")
    print(f"  Average edge length: {sizes[:,0].mean():.1f}px (std: {sizes[:,0].std():.1f})")
    print(f"  Min edge length: {sizes[:,1].mean():.1f}px (std: {sizes[:,1].std():.1f})")
    print(f"  Tags with min_edge < 10px: {(sizes[:,1] < 10).sum()} ({(sizes[:,1] < 10).mean()*100:.1f}%)")
    print(f"  Tags with min_edge < 5px: {(sizes[:,1] < 5).sum()} ({(sizes[:,1] < 5).mean()*100:.1f}%)")
    print(f"  Tags with min_edge < 3px: {(sizes[:,1] < 3).sum()} ({(sizes[:,1] < 3).mean()*100:.1f}%)")

def main():
    forward_csv = Path("tests/data/icra2020/forward/tags.csv")
    circle_csv = Path("tests/data/icra2020/circle/tags.csv")
    
    if forward_csv.exists():
        analyze_tag_sizes(forward_csv, "Forward")
    
    if circle_csv.exists():
        analyze_tag_sizes(circle_csv, "Circle")

if __name__ == "__main__":
    main()
