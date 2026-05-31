import os
import threading
import time

import locus
import numpy as np


def test_gil_release_concurrency():
    """
    Verify that locus.Detector.detect() releases the GIL.

    This test runs a pure-Python CPU-bound task in a background thread.
    If the GIL is correctly released by the Rust core, the Python task
    will execute in parallel (on multi-core systems) or context-switch
    efficiently (on single-core systems).
    """
    # 1. Setup
    height, width = 1080, 1920
    img = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    detector = locus.Detector()

    # Warmup
    detector.detect(img)

    # 2. Calibration: Measure baseline detection time
    # We run multiple times to get a stable average
    det_iterations = 10
    start = time.perf_counter()
    for _ in range(det_iterations):
        detector.detect(img)
    baseline_det_total = time.perf_counter() - start
    avg_det_time = baseline_det_total / det_iterations
    print(f"\nAvg detection time: {avg_det_time * 1000:.2f} ms")

    # 3. Calibration: Prepare a pure-Python task that holds the GIL
    def python_work(n):
        count = 0
        for i in range(n):
            count += i
        return count

    # Find number of iterations that takes roughly the same time as 10 detections
    work_n = 100_000
    start = time.perf_counter()
    python_work(work_n)
    elapsed = time.perf_counter() - start

    target_work_n = int(work_n * (baseline_det_total / elapsed))

    # Measure baseline Python time
    start = time.perf_counter()
    python_work(target_work_n)
    baseline_py_total = time.perf_counter() - start
    print(f"Baseline Python work time: {baseline_py_total * 1000:.2f} ms")

    # 4. Parallel Execution
    # We run the Python work in a background thread and the Rust detections in the main thread.

    def background_thread():
        python_work(target_work_n)

    t = threading.Thread(target=background_thread)

    start_parallel = time.perf_counter()
    t.start()
    for _ in range(det_iterations):
        detector.detect(img)
    t.join()
    parallel_total = time.perf_counter() - start_parallel

    sum_baselines = baseline_det_total + baseline_py_total
    max_baselines = max(baseline_det_total, baseline_py_total)

    print(f"Parallel total time: {parallel_total * 1000:.2f} ms")
    print(f"Sum of baselines: {sum_baselines * 1000:.2f} ms")
    print(f"Max of baselines: {max_baselines * 1000:.2f} ms")

    # 5. Verification
    cpu_count = os.cpu_count() or 1

    if cpu_count >= 2:
        # On multi-core systems, we expect significant overlap.
        # Total time should be much closer to max(baselines) than sum(baselines).
        # We use 0.85 * sum as a conservative bound for "some parallelism happened".
        assert parallel_total < sum_baselines * 0.85, (
            f"GIL not released on {cpu_count}-core system. Parallel time ({parallel_total:.4f}) too close to sum ({sum_baselines:.4f})"
        )
    else:
        # On single-core systems, parallel performance won't improve total time,
        # and might even be slightly slower due to context switching.
        # We just verify it doesn't explode.
        print("Skipping performance assertion on single-core system.")
        assert parallel_total < sum_baselines * 1.2


if __name__ == "__main__":
    test_gil_release_concurrency()
