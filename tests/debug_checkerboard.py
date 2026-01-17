import cv2
import numpy as np
import locus
from pathlib import Path

def visualize_labels(labels):
    # Map component labels to random colors
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[labels == 0] = 0
    return labeled_img

def main():
    img_path = Path("tests/data/icra2020/forward/checkerboard_corners_images/0000.png")
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return

    print(f"Loading {img_path}...")
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    # 1. Threshold
    print("Running debug_threshold...")
    thresh_img = locus.debug_threshold(img).astype(np.uint8) * 255
    cv2.imwrite("/tmp/checkerboard_thresh.png", thresh_img)
    print("Saved /tmp/checkerboard_thresh.png")
    
    # 2. Segmentation
    print("Running debug_segmentation...")
    labels = locus.debug_segmentation(img).astype(np.uint32)
    print(f"Found {np.max(labels)} components")
    
    vis_labels = visualize_labels(labels)
    cv2.imwrite("/tmp/checkerboard_labels.png", vis_labels)
    print("Saved /tmp/checkerboard_labels.png")
    
    # 3. Detection
    print("Running detection...")
    detector = locus.Detector()
    detections, stats = detector.detect_with_stats(img)
    print(f"Detections: {len(detections)}")
    print(f"Candidates: {stats.num_candidates}")

if __name__ == "__main__":
    main()
