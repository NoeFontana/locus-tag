
import cv2
import numpy as np
import locus
from pathlib import Path
import csv

def load_ground_truth(csv_path: Path, target_img="0000.png"):
    gt = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['image'] != target_img: continue
            if row.get('tag_fully_visible', '1') != '1': continue
            
            tid = int(row['tag_id'])
            corner = int(row['corner'])
            x = float(row['ground_truth_x'])
            y = float(row['ground_truth_y'])
            
            if tid not in gt:
                gt[tid] = [None, None, None, None]
            gt[tid][corner] = (x, y)
    return gt

def main():
    root = Path("tests/data/icra2020/forward")
    img_path = root / "pure_tags_images" / "0000.png"
    
    # Enable all debug features? 
    # To get candidates, we need to inspect internals or rely on detection failure analysis.
    # Locus Python API doesn't expose candidates.
    # BUT, we can use `quad_min_area=1` and minimal filtering to see if we get *detections* if we use a permissive decoder?
    # No, Locus decoder is fixed.
    
    # Workaround: Use 'locus.debug_segmentation' to check connected components.
    
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    # 1. Check Segmentation
    # Getting labeled image
    labels = locus.debug_segmentation(img).astype(np.int32)
    
    gt = load_ground_truth(root / "tags.csv", "0000.png")
    
    print(f"Checking {len(gt)} tags in 0000.png")
    
    segmented_count = 0
    valid_geometry_count = 0
    
    # Visualize
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for tid, corners in gt.items():
        if None in corners: continue
        pts = np.array(corners)
        center = np.mean(pts, axis=0).astype(int)
        
        # Check if center is in a valid component
        lbl = labels[center[1], center[0]]
        
        if lbl > 0:
            segmented_count += 1
            # Check geometry of this component roughly?
            # We can't easily extract quad from label map in python without re-implementing logic.
            # But knowing it's segmented is step 1.
            cv2.circle(vis, tuple(center), 2, (0, 255, 0), -1)
        else:
            cv2.circle(vis, tuple(center), 2, (0, 0, 255), -1)
            # Check if any corner is labeled?
            
    print(f"Tags with center in a component: {segmented_count}/{len(gt)}")
    cv2.imwrite("/tmp/debug_segmentation_centers.png", vis)
    
    # 2. Check Detection with Permissive Settings
    from locus.locus import Detector as RustDetector
    det = RustDetector(
        threshold_tile_size=2,
        quad_min_area=4,
        quad_min_edge_length=2.0,
        quad_min_edge_score=0.1
    )
    detections, stats = det.detect_with_stats(img)
    print(f"Permissive Detection Count: {len(detections)}")
    print(f"Candidates Found: {stats.num_candidates}")

if __name__ == "__main__":
    main()
