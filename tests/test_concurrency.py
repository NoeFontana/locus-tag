import threading
import time
import numpy as np
import locus
import pytest

def test_gil_release_concurrency():
    """
    Verify that locus.Detector.detect() releases the GIL by running a
    NumPy heavy task in parallel.
    """
    height, width = 1080, 1920
    img = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    
    detector = locus.Detector()
    
    # 1. Warmup
    detector.detect(img)

    # 2. Measure baseline detection time (synchronous)
    iterations = 5
    start_det = time.perf_counter()
    for _ in range(iterations):
        detector.detect(img)
    end_det = time.perf_counter()
    baseline_det_time = (end_det - start_det) / iterations
    print(f"\nBaseline detection time: {baseline_det_time*1000:.2f} ms")

    # 3. Prepare a NumPy task that takes roughly the same time
    # Matrix multiplication releases the GIL in NumPy
    size = 2000
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    
    def numpy_task():
        np.dot(A, B)

    start_npy = time.perf_counter()
    numpy_task()
    end_npy = time.perf_counter()
    baseline_npy_time = end_npy - start_npy
    print(f"Baseline NumPy CPU task time: {baseline_npy_time*1000:.2f} ms")

    # 4. Run in parallel
    def background_task():
        numpy_task()

    t = threading.Thread(target=background_task)
    
    start_total = time.perf_counter()
    t.start()
    detector.detect(img)
    t.join()
    end_total = time.perf_counter()
    
    total_parallel_time = end_total - start_total
    max_of_baselines = max(baseline_det_time, baseline_npy_time)
    sum_of_baselines = baseline_det_time + baseline_npy_time
    
    print(f"Total parallel time: {total_parallel_time*1000:.2f} ms")
    print(f"Max of baselines: {max_of_baselines*1000:.2f} ms")
    print(f"Sum of baselines: {sum_of_baselines*1000:.2f} ms")
    
    # If they run in parallel, total time should be much less than sum.
    # We use 0.9 as a conservative threshold to ensure we detect sequential execution
    # while allowing for thread management overhead.
    assert total_parallel_time < sum_of_baselines * 0.9, \
        f"GIL was likely not released. Parallel time ({total_parallel_time:.4f}) is too close to sum of baselines ({sum_of_baselines:.4f})"

if __name__ == "__main__":
    test_gil_release_concurrency()
