import time

import locus
import numpy as np
import rerun as rr


def main():
    rr.init("locus_decoder_viz", spawn=True)

    # 1. Generate a synthetic image with a known tag pattern
    # 6x6 grid + 2 border = 8x8 units
    # Let's say code 0x123456789 (ID 42)
    # Binary: 0001 0010 0011 0100 0101 0110 0111 1000 1001
    code = 0x123456789

    img = np.ones((1000, 1000), dtype=np.uint8) * 240
    tag_size = 400
    tx, ty = 300, 300

    # Draw border (black)
    img[ty : ty + tag_size, tx : tx + tag_size] = 20

    # Draw content (normalized to 1.0 units)
    # dim = 8 (6 content + 2 border)
    # Unit size = tag_size / 8
    u = tag_size // 8

    for row in range(6):
        for col in range(6):
            if (code >> (row * 6 + col)) & 1:
                # White pixel for '1'
                r0 = ty + (row + 1) * u
                c0 = tx + (col + 1) * u
                img[r0 : r0 + u, c0 : c0 + u] = 230
            else:
                # Black pixel for '0'
                r0 = ty + (row + 1) * u
                c0 = tx + (col + 1) * u
                img[r0 : r0 + u, c0 : c0 + u] = 20

    rr.log("input", rr.Image(img))

    # 2. Run detection
    detections = locus.detect_tags(img)
    print(f"Detected {len(detections)} tags")

    for i, d in enumerate(detections):
        print(f"Tag ID: {d.id}, Hamming: {d.hamming}, Center: {d.center}")
        corners = np.array(d.corners)
        corners = np.vstack([corners, corners[0]])
        rr.log(f"tags/tag{i}", rr.LineStrips2D(corners, labels=f"ID:{d.id} H:{d.hamming}"))

    time.sleep(5)


if __name__ == "__main__":
    main()
