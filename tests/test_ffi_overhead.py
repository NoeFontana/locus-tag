import time

import locus
import numpy as np


def test_ffi_ingestion_overhead():
    """Measure the overhead of passing a 1080p image to the Rust core."""
    # 1080p grayscale image
    height, width = 1080, 1920
    img = np.zeros((height, width), dtype=np.uint8)

    # We use a detector and measure the time of detect_with_stats
    # PipelineStats.total_ms is the time spent INSIDE Rust.
    # The difference between Python-measured time and total_ms is the FFI/Wrapper overhead.

    detector = locus.Detector()

    # Warmup
    detector.detect_with_stats(img)

    iterations = 100
    python_times = []
    rust_times = []

    for _ in range(iterations):
        start = time.perf_counter()
        _, stats = detector.detect_with_stats(img)
        end = time.perf_counter()

        python_times.append((end - start) * 1000.0)  # ms
        rust_times.append(stats.total_ms)

    avg_python = sum(python_times) / iterations
    avg_rust = sum(rust_times) / iterations
    overhead = avg_python - avg_rust

    print("\n1080p Ingestion Overhead:")
    print(f"  Avg Python Time: {avg_python:.4f} ms")
    print(f"  Avg Rust Time:   {avg_rust:.4f} ms")
    print(f"  Avg Overhead:    {overhead:.4f} ms")

    # According to specification, overhead should be < 0.1 ms
    # If it's currently 110ms as reported, this will fail.
    assert overhead < 0.1, f"FFI overhead too high: {overhead:.4f} ms > 0.1 ms"


if __name__ == "__main__":
    try:
        test_ffi_ingestion_overhead()
    except AssertionError as e:
        print(f"FAILED: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
