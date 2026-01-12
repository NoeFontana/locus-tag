import cv2
import locus
import numpy as np


def create_test_image():
    # Create an ArUco 4x4_50 tag image (known working in Locus)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag_size = 100
    # ID 0
    img = cv2.aruco.generateImageMarker(dictionary, 0, tag_size)

    # Place on canvas
    h, w = 400, 400
    canvas = np.ones((h, w), dtype=np.uint8) * 128
    # White background for tag
    canvas[140:260, 140:260] = 255
    canvas[150:250, 150:250] = img

    # Blur slightly to realistic
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    return canvas


def test_config():
    print("Generating test image...")
    img = create_test_image()

    # 1. Test Default Configuration
    # Note: ArUco 4x4 is not in the default families (default is 36h11),
    # so we need to set it or use detect_with_options.
    print("Testing default configuration with family selection...")
    det = locus.Detector()

    # Check that it doesn't find ArUco by default (it shouldn't)
    res_def = det.detect(img)
    if len(res_def) == 0:
        print("SUCCESS: Default config ignored ArUco tag as expected")
    else:
        print(f"WARNING: Default config found {len(res_def)} tags (maybe false positive?)")

    # Now detect with specific family
    results = det.detect_with_options(img, families=[locus.TagFamily.ArUco4x4_50])

    if len(results) == 1:
        print(f"SUCCESS: detected tag ID {results[0].id}")
    else:
        print(f"FAILURE: found {len(results)} tags")
        exit(1)

    # 2. Test Custom Configuration (Strict Area)
    print("Testing strict area configuration...")
    # Tag is 100x100 = 10000 pixels
    # Let's set min area to 20000
    det_strict = locus.Detector(quad_min_area=20000)
    results_strict = det_strict.detect_with_options(img, families=[locus.TagFamily.ArUco4x4_50])

    if len(results_strict) == 0:
        print("SUCCESS: Strict area config filtered out the tag")
    else:
        print(f"FAILURE: Strict config found {len(results_strict)} tags (should be 0)")
        exit(1)

    # 3. Test Custom Configuration (Strict Edge Score)
    print("Testing strict edge score configuration...")
    # Set edge score requirement insanely high
    det_edge = locus.Detector(quad_min_edge_score=1000.0)
    results_edge = det_edge.detect_with_options(img, families=[locus.TagFamily.ArUco4x4_50])

    if len(results_edge) == 0:
        print("SUCCESS: Strict edge score config filtered out the tag")
    else:
        print("FAILURE: Strict edge score config found tags")
        exit(1)

    # 4. Test Persistence (Arena reuse) calls
    print("Testing multiple calls (persistence)...")
    for _ in range(5):
        assert len(det.detect_with_options(img, families=[locus.TagFamily.ArUco4x4_50])) == 1
    print("SUCCESS: Persistence working")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    test_config()
