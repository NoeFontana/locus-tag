import numpy as np
import locus
import sys

def test_zero_copy():
    print("Testing zero-copy ingestion...")
    
    # 1. Standard C-contiguous array
    img = np.zeros((1080, 1920), dtype=np.uint8)
    img[50, 60] = 255
    detections = locus.detect_tags(img)
    print(f"Standard array: {len(detections)} detections found.")
    assert len(detections) == 1
    assert detections[0].id == 42
    
    # 2. Strided array (padding)
    # Create an array that is larger and then slice it to create non-trivial strides
    full_img = np.zeros((1080, 2000), dtype=np.uint8)
    img_strided = full_img[:, :1920]
    assert img_strided.strides[0] == 2000
    assert img_strided.strides[1] == 1
    
    detections = locus.detect_tags(img_strided)
    print(f"Strided array: {len(detections)} detections found.")
    assert len(detections) == 1

    # 3. F-contiguous array (should fail)
    img_f = np.asfortranarray(img)
    try:
        locus.detect_tags(img_f)
        print("Error: F-contiguous array should have failed but did not.")
        sys.exit(1)
    except ValueError as e:
        print(f"Caught expected error for F-contiguous: {e}")

    print("Verification successful!")

if __name__ == "__main__":
    test_zero_copy()

