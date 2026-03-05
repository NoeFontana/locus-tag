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
    detector.detect(img_c)

    iterations = 50
    python_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        detector.detect(img_c)
        end = time.perf_counter()
        python_times.append((end - start) * 1000.0)

    avg_python = sum(python_times) / iterations
    print(f"  Avg Total Time: {avg_python:.4f} ms")

    # Non-contiguous should raise ValueError
    print("\nVerifying Non-Contiguous is blocked...")
    try:
        detector.detect(img_nc)
        print("FAILED: Non-contiguous array was NOT blocked!")
        exit(1)
    except ValueError as e:
        print(f"SUCCESS: Caught expected ValueError: {e}")


if __name__ == "__main__":
    bench_ingestion_penalty()
