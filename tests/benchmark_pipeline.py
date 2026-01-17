import cv2
import numpy as np
import locus
import time
import os

def benchmark_config(name, img, **kwargs):
    # Use the internal Rust detector directly to avoid Pydantic/deepcopy issues
    # with SegmentationConnectivity enum during development
    detector = locus.locus.Detector(**kwargs)
    
    # Warmup
    for _ in range(5):
        detector.detect_with_stats(img)
    
    # Measure
    n_iters = 20
    stats_acc = []
    
    for _ in range(n_iters):
        _, stats = detector.detect_with_stats(img)
        stats_acc.append(stats)
        
    print(f"\n=== Benchmark: {name} ===")
    print(f"Image: {img.shape[1]}x{img.shape[0]}")
    print(f"Parameters: {kwargs}")
    
    avg_total = sum(s.total_ms for s in stats_acc) / n_iters
    avg_threshold = sum(s.threshold_ms for s in stats_acc) / n_iters
    avg_seg = sum(s.segmentation_ms for s in stats_acc) / n_iters
    avg_quad = sum(s.quad_extraction_ms for s in stats_acc) / n_iters
    avg_decode = sum(s.decoding_ms for s in stats_acc) / n_iters
    
    max_candidates = max(s.num_candidates for s in stats_acc)
    num_detections = stats_acc[0].num_detections
    
    print(f"  Total:      {avg_total:7.2f} ms")
    print(f"  Threshold:  {avg_threshold:7.2f} ms")
    print(f"  Segment:    {avg_seg:7.2f} ms")
    print(f"  Quad Extr:  {avg_quad:7.2f} ms")
    print(f"  Decoding:   {avg_decode:7.2f} ms")
    print(f"  Candidates: {max_candidates}")
    print(f"  Detections: {num_detections}")

def main():
    # Use a representative image from the dataset
    root = "."
    img_path = os.path.join(root, "crates/locus-core/tests/fixtures/icra2020/0037.png")
    if not os.path.exists(img_path):
        # Fallback to a synthetic or noise if not found
        img = (np.random.randint(0, 2, (480, 640)) * 255).astype(np.uint8)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 1. Baseline
    benchmark_config("Baseline (1x)", img, upscale_factor=1, enable_sharpening=False)
    
    # 2. Sharpening only
    benchmark_config("Sharpening (1x)", img, upscale_factor=1, enable_sharpening=True)
    
    # 3. Upscaling only
    benchmark_config("Upscaling (2x)", img, upscale_factor=2, enable_sharpening=False)
    
    # 4. Sharpening + Upscaling
    benchmark_config("Sharp + Upscale (2x)", img, upscale_factor=2, enable_sharpening=True)
    
    # 5. High Recall Profile (The "Slow" one)
    benchmark_config("Checkerboard Profile (1x)", img, 
                     enable_sharpening=True, 
                     enable_bilateral=False,
                     segmentation_connectivity=locus.locus.SegmentationConnectivity.Four,
                     decoder_min_contrast=10.0,
                     quad_min_area=8,
                     quad_min_edge_length=2.0)

if __name__ == "__main__":
    main()
