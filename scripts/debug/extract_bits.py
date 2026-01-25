import cv2


def get_apriltag_bits(tag_id, family=cv2.aruco.DICT_APRILTAG_36h11):
    dictionary = cv2.aruco.getPredefinedDictionary(family)
    # Generate a large enough image to see bits clearly
    tag_img = cv2.aruco.generateImageMarker(dictionary, tag_id, 200)

    # AprilTag 36h11 is 8x8 (6x6 data + 1 cell border)
    # Let's sample the 8x8 grid centers
    grid_size = 8
    cell_size = 200 // grid_size
    bits = 0

    # We want the inner 6x6
    for row in range(6):
        for col in range(6):
            # Inner 6x6 starts at index 1,1 in the 8x8 grid
            cy = (row + 1) * cell_size + cell_size // 2
            cx = (col + 1) * cell_size + cell_size // 2
            val = tag_img[cy, cx]
            if val > 128:
                bit_idx = row * 6 + col
                bits |= 1 << bit_idx

    return bits


id0 = get_apriltag_bits(0)
print(f"ID 0: {hex(id0)}")
id1 = get_apriltag_bits(1)
print(f"ID 1: {hex(id1)}")
id2 = get_apriltag_bits(2)
print(f"ID 2: {hex(id2)}")
