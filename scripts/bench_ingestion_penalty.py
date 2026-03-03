import time

import locus
import numpy as np


def bench_ingestion_penalty():
    """Verify that non-contiguous ingestion is strictly blocked and measure baseline."""
    height, width = 1080, 1920

    # 1. Contiguous array
    img_c = np.zeros((height, width), dtype=np.uint8)

    # 2. Non-contiguous array (stride_x != 1)
    img_large = np.zeros((height, width * 2), dtype=np.uint8)
    img_nc = img_large[:, ::2]

    detector = locus.Detector()

    # Contiguous should work fine
    print("\nRunning Contiguous (Zero-Copy) baseline...")
    detector.detect_with_stats(img_c)

    iterations = 50
    python_times = []
    rust_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _, stats = detector.detect_with_stats(img_c)
        end = time.perf_counter()
        python_times.append((end - start) * 1000.0)
        rust_times.append(stats.total_ms)

    avg_python = sum(python_times) / iterations
    avg_rust = sum(rust_times) / iterations
    overhead = avg_python - avg_rust
    print(f"  Avg Python Time: {avg_python:.4f} ms")
    print(f"  Avg Rust Time:   {avg_rust:.4f} ms")
    print(f"  Avg Overhead:    {overhead:.4f} ms")

    # Non-contiguous should raise ValueError
    print("\nVerifying Non-Contiguous is blocked...")
    try:
        detector.detect_with_stats(img_nc)
        print("FAILED: Non-contiguous array was NOT blocked!")
        exit(1)
    except ValueError as e:
        print(f"SUCCESS: Caught expected ValueError: {e}")


if __name__ == "__main__":
    bench_ingestion_penalty()
