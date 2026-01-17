#!/usr/bin/env python3
"""Diagnostic script to visualize quad candidates on extreme tilt images."""

import cv2
import numpy as np
from pathlib import Path

# Load a challenging image from circle dataset
CIRCLE_DIR = Path("tests/data/icra2020/circle/pure_tags_images")

def visualize_quads(img_path: Path):
    """Visualize quad candidates on an image."""
    import locus
    
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    
    # Create color image for visualization
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    detector = locus.Detector(
        quad_min_area=16,
        quad_max_aspect_ratio=20.0,
        quad_min_edge_score=0.5,
        enable_bilateral=False,
        enable_adaptive_window=False,
    )
    
    detections, stats = detector.detect_with_stats(img)
    
    print(f"Image: {img_path.name}")
    print(f"  Quads: {stats.num_candidates}, Detected: {stats.num_detections}")
    
    # Draw detected tags in green
    for det in detections:
        corners = np.array(det.corners, dtype=np.int32)
        cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
        cv2.putText(vis, str(det.id), tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Save visualization
    out_path = Path(f"/tmp/circle_debug_{img_path.stem}.png")
    cv2.imwrite(str(out_path), vis)
    print(f"  Saved: {out_path}")
    
    return stats.num_candidates, stats.num_detections

def main():
    if not CIRCLE_DIR.exists():
        print(f"Circle dataset not found at {CIRCLE_DIR}")
        return
    
    # Test first 5 images (most extreme tilt)
    images = sorted(CIRCLE_DIR.glob("*.png"))[:5]
    
    print("Analyzing extreme tilt images...\n")
    for img_path in images:
        visualize_quads(img_path)
        print()

if __name__ == "__main__":
    main()
