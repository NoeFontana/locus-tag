import threading
import time

import locus
import numpy as np


def test_gil_release():
    """Verify that the GIL is released during detection."""
    # Create two large images to ensure detection takes some time
    img1 = np.zeros((4000, 4000), dtype=np.uint8)
    img2 = np.zeros((4000, 4000), dtype=np.uint8)

    # Fill with some noise so detection does something
    img1[::2, ::2] = 255
    img2[::3, ::3] = 255

    times = []

    def run_det(img):
        det = locus.Detector()
        start = time.perf_counter()
        det.detect(img)
        end = time.perf_counter()
        times.append(end - start)

    def busy_wait():
        start = time.perf_counter()
        # Compute something in Python to hold the GIL
        s = 0
        while time.perf_counter() - start < 1.0:
            s += 1
        end = time.perf_counter()
        times.append(end - start)

    t1 = threading.Thread(target=busy_wait)

    start_all = time.perf_counter()
    t1.start()
    run_det(img1)
    t1.join()
    end_all = time.perf_counter()

    total_time = end_all - start_all
    sum_individual = sum(times)

    print("\nConcurrency Test (Busy Wait):")
    print(f"  Busy Wait: {times[0] * 1000:.2f} ms")
    print(f"  Detector: {times[1] * 1000:.2f} ms")
    print(f"  Total time: {total_time * 1000:.2f} ms")
    print(f"  Sum of individual: {sum_individual * 1000:.2f} ms")

    overlap_ratio = total_time / sum_individual
    print(f"  Overlap ratio: {overlap_ratio:.2f}")

    assert overlap_ratio < 0.9, "GIL was NOT released (sequential execution detected)"


if __name__ == "__main__":
    try:
        test_gil_release()
    except AssertionError as e:
        print(f"FAILED: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
