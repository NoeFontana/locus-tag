import cv2
import locus
import numpy as np
from pathlib import Path

def main():
    data_dir = Path("tests/data/icra2020/forward")
    img_path = data_dir / "pure_tags_images" / "0001.png"
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    # Try different configs to see impact on candidate count
    configs = [
        ("Default", {}),
        ("Permissive", {
            "quad_min_area": 4,
            "quad_min_edge_score": 0.1,
            "quad_max_aspect_ratio": 10.0,
            "adaptive_threshold_constant": 1
        }),
        ("Strict", {
            "quad_min_area": 25,
            "quad_min_edge_score": 0.5
        })
    ]
    
    for name, cfg_kwargs in configs:
        detector = locus.Detector(**cfg_kwargs)
        detections, stats = detector.detect_with_stats(img)
        print(f"Config: {name}")
        print(f"  Candidates: {stats.num_candidates}")
        print(f"  Detections: {len(detections)}")

if __name__ == "__main__":
    main()
