import threading
import time

import locus
import numpy as np


def test_simple_gil():
    detector = locus.Detector()
    img = np.zeros((1080, 1920), dtype=np.uint8)

    def background():
        for _i in range(5):
            print(f"Background thread active: {time.time()}")
            time.sleep(0.01)

    t = threading.Thread(target=background)
    print(f"Starting background thread: {time.time()}")
    t.start()
    time.sleep(0.02)  # Let background thread start
    print(f"Starting Rust detection: {time.time()}")
    detector.detect(img)
    print(f"Finished Rust detection: {time.time()}")
    t.join()


if __name__ == "__main__":
    test_simple_gil()
