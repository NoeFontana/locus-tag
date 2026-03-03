import time
import locus
import numpy as np

def bench_ingestion_penalty():
    """Measure the overhead of passing a 1080p image to the Rust core (Contiguous vs Non-Contiguous)."""
    height, width = 1080, 1920
    
    # 1. Contiguous array
    img_c = np.zeros((height, width), dtype=np.uint8)
    
    # 2. Non-contiguous array (stride_x != 1)
    # We create a 1080p view from a larger array to keep it 1080p but non-contiguous
    img_large = np.zeros((height, width * 2), dtype=np.uint8)
    img_nc = img_large[:, ::2] # 1080x1920 but with stride_x = 2
    
    assert img_c.shape == (1080, 1920)
    assert img_nc.shape == (1080, 1920)
    assert img_c.flags["C_CONTIGUOUS"]
    assert not img_nc.flags["C_CONTIGUOUS"]
    assert img_nc.strides[1] == 2

    detector = locus.Detector()

    # Warmup
    detector.detect_with_stats(img_c)
    detector.detect_with_stats(img_nc)

    iterations = 50
    
    def measure(img, label):
        python_times = []
        rust_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _, stats = detector.detect_with_stats(img)
            end = time.perf_counter()
            python_times.append((end - start) * 1000.0)
            rust_times.append(stats.total_ms)
        
        avg_python = sum(python_times) / iterations
        avg_rust = sum(rust_times) / iterations
        overhead = avg_python - avg_rust
        print(f"\n{label}:")
        print(f"  Avg Python Time: {avg_python:.4f} ms")
        print(f"  Avg Rust Time:   {avg_rust:.4f} ms")
        print(f"  Avg Overhead:    {overhead:.4f} ms")
        return overhead

    overhead_c = measure(img_c, "Contiguous (Zero-Copy)")
    overhead_nc = measure(img_nc, "Non-Contiguous (Auto-Copy)")

    penalty = overhead_nc - overhead_c
    print(f"\nIngestion Penalty (Copy Cost): {penalty:.4f} ms")

if __name__ == "__main__":
    bench_ingestion_penalty()
