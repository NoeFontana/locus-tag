import time

import locus
import numpy as np
import rerun as rr


def main():
    rr.init("locus_visualizer", spawn=True)

    # 1. Generate a synthetic image with a "tag" (white square) and some noise
    img = np.random.randint(0, 50, (1080, 1920), dtype=np.uint8)
    # Add a high-contrast square
    img[400:600, 800:1000] = 200
    # Add some mid-gray background
    img[100:300, 100:300] = 100

    rr.log("input", rr.Image(img))

    # 2. Benchmark the thresholding
    start = time.perf_counter()
    bin_img = locus.debug_threshold(img)
    end = time.perf_counter()

    print(f"Thresholding took {(end - start) * 1000:.2f}ms (including GIL/NumPy overhead)")

    rr.log("thresholded", rr.Image(bin_img))

    # Keep alive for a bit to see the output
    time.sleep(5)


if __name__ == "__main__":
    main()
