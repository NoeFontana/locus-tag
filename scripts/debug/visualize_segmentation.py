import time

import locus
import numpy as np
import rerun as rr


def main():
    rr.init("locus_visualizer", spawn=True)

    # 1. Generate a synthetic image with multiple "tags" (dark on light)
    img = np.random.randint(220, 255, (1080, 1920), dtype=np.uint8)

    # Square 1 (dark)
    img[200:400, 200:400] = 30
    # Square 2 (dark)
    img[600:800, 1000:1200] = 50
    # Noise/Background patches
    img[100:150, 1500:1600] = 150

    rr.log("input", rr.Image(img))

    # 2. Segmentation
    labels = locus.debug_segmentation(img)
    rr.log("segmentation", rr.SegmentationImage(labels))

    # 3. Full detection (quad extraction)
    detections = locus.detect_tags(img)
    print(f"Detected {len(detections)} potential quads")

    rr.log("quads", rr.Clear.recursive())  # Clear old quads
    for i, d in enumerate(detections):
        corners = np.array(d.corners)
        # Close the loop for drawing
        corners = np.vstack([corners, corners[0]])
        rr.log(f"quads/q{i}", rr.LineStrips2D(corners, labels=f"Tag {d.id}"))

    # Keep alive to see output
    time.sleep(10)


if __name__ == "__main__":
    main()
