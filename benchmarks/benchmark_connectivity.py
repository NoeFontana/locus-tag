
import cv2
import locus
import time
import os
import glob
import numpy as np
from pathlib import Path
from tabulate import tabulate

def load_images(img_dir, limit=50):
    print(f"Loading images from {img_dir}...")
    paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    print(f"Found {len(paths)} images.")
    return paths[:limit] if limit else paths

def benchmark_dataset(name, paths, connectivity, expected_tags_per_image=None):
    detector = locus.Detector(
        segmentation_connectivity=connectivity,
        quad_min_area=16,
        threshold_tile_size=4,
        threshold_min_range=2
    )

    total_detections = 0
    total_time = 0
    total_images = len(paths)
    
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        t0 = time.time()
        dets = detector.detect(img)
        dt = (time.time() - t0) * 1000
        
        total_time += dt
        total_detections += len(dets)
        
    avg_time = total_time / total_images
    avg_dets = total_detections / total_images
    
    return {
        "dataset": name,
        "connectivity": str(connectivity).split(".")[-1],
        "total_detections": total_detections,
        "avg_detections": avg_dets,
        "avg_time_ms": avg_time,
        "fps": 1000.0 / avg_time if avg_time > 0 else 0
    }

def main():
    # Datasets
    forward_dir = "tests/data/icra2020/forward/pure_tags_images"
    checker_dir = "tests/data/icra2020/forward/checkerboard_corners_images"
    
    forward_paths = load_images(forward_dir, limit=50)
    checker_paths = load_images(checker_dir, limit=50)
    
    results = []
    
    print("Benchmarking Connectivity Modes...")
    
    # 1. Forward (Isolated Tags)
    print(f"Running Forward Dataset ({len(forward_paths)} images)...")
    results.append(benchmark_dataset("Forward", forward_paths, locus.SegmentationConnectivity.Eight))
    results.append(benchmark_dataset("Forward", forward_paths, locus.SegmentationConnectivity.Four))
    
    # 2. Checkerboard
    print(f"Running Checkerboard Dataset ({len(checker_paths)} images)...")
    results.append(benchmark_dataset("Checkerboard", checker_paths, locus.SegmentationConnectivity.Eight))
    results.append(benchmark_dataset("Checkerboard", checker_paths, locus.SegmentationConnectivity.Four))
    
    print("\nResults:")
    headers = ["Dataset", "Connectivity", "Total Dets", "Avg Dets/Img", "Avg Time (ms)", "FPS"]
    table = [[r["dataset"], r["connectivity"], r["total_detections"], f"{r['avg_detections']:.1f}", f"{r['avg_time_ms']:.2f}", f"{r['fps']:.1f}"] for r in results]
    print(tabulate(table, headers=headers))

if __name__ == "__main__":
    main()
