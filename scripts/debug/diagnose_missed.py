import csv
from pathlib import Path

import cv2
import locus
import numpy as np


def parse_gt(csv_path):
    gt_map: dict[str, dict[int, list[list[float] | None]]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row["image"]
            tid = int(row["tag_id"])
            c_idx = int(row["corner"])
            x = float(row["ground_truth_x"])
            y = float(row["ground_truth_y"])
            vis = int(row.get("tag_fully_visible", 1)) == 1
            if not vis:
                continue
            if img not in gt_map:
                gt_map[img] = {}
            if tid not in gt_map[img]:
                gt_map[img][tid] = [None] * 4
            gt_map[img][tid][c_idx] = [x, y]
    final_gt = {}
    for img, tags in gt_map.items():
        valid_tags = []
        for tid, corners in tags.items():
            if all(c is not None for c in corners):
                valid_tags.append({"id": tid, "corners": np.array(corners)})
        if valid_tags:
            final_gt[img] = valid_tags
    return final_gt


def main():
    data_dir = Path("tests/data/icra2020/forward")
    img_dir = data_dir / "pure_tags_images"
    gt_map = parse_gt(data_dir / "tags.csv")

    # Use baseline config
    detector = locus.Detector()

    for img_name in ["0001.png", "0012.png"]:
        img_path = img_dir / img_name
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        detections = detector.detect(img)
        det_ids = {d.id for d in detections}

        bin_img = locus.debug_threshold(img)
        # Also try global Otsu for comparison
        _, bin_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        gt_tags = gt_map[img_name]
        for gt in gt_tags:
            if gt["id"] not in det_ids:
                # Missed tag. Crop around it.
                corners = gt["corners"]
                center = np.mean(corners, axis=0).astype(int)
                x, y = center
                r = 30
                x0, x1 = max(0, x - r), min(img.shape[1], x + r)
                y0, y1 = max(0, y - r), min(img.shape[0], y + r)

                crop_raw = img[y0:y1, x0:x1]
                crop_bin = bin_img[y0:y1, x0:x1]
                crop_otsu = bin_otsu[y0:y1, x0:x1]

                # Combine raw, bin, otsu side-by-side
                combined = np.hstack([crop_raw, crop_bin, crop_otsu])
                out_path = f"debug_missed_{img_name}_tag_{gt['id']}.png"
                cv2.imwrite(out_path, combined)
                print(f"Saved debug crop: {out_path}")

                # Only save a few per image
                if gt["id"] > 50:
                    break


if __name__ == "__main__":
    main()
