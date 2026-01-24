import argparse
import csv
from pathlib import Path

import cv2
import locus
import numpy as np
import rerun as rr
from tqdm import tqdm


def parse_gt(csv_path):
    gt_map = {}
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
    parser = argparse.ArgumentParser(description="Locus Recall/RMSE Debugger")
    parser.add_argument("--limit", type=int, default=50)
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "locus_debug")

    data_dir = Path("tests/data/icra2020/forward")
    img_dir = data_dir / "pure_tags_images"
    csv_path = data_dir / "tags.csv"

    gt_map = parse_gt(csv_path)

    detector = locus.Detector()

    images = sorted(list(img_dir.glob("*.png")))[: args.limit]

    all_recalls = []
    all_rmses = []

    for img_path in tqdm(images):
        img_name = img_path.name
        if img_name not in gt_map:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        detections, stats = detector.detect_with_stats(img)

        step = int(img_name.split(".")[0])
        rr.set_time(timeline="image_idx", sequence=step)
        rr.log("input", rr.Image(img))

        gt_tags = gt_map[img_name]

        # Log GT
        for gt in gt_tags:
            corners = np.vstack([gt["corners"], gt["corners"][0]])
            rr.log(f"gt/tag_{gt['id']}", rr.LineStrips2D(corners, colors=[0, 255, 0], radii=0.5))

        # Log Detections
        det_ids = {d.id for d in detections}
        for d in detections:
            corners = np.array(d.corners)
            corners = np.vstack([corners, corners[0]])
            rr.log(f"det/tag_{d.id}", rr.LineStrips2D(corners, colors=[255, 255, 0], radii=0.5))

        # Log missed tags explicitly in red
        for gt in gt_tags:
            if gt["id"] not in det_ids:
                corners = np.vstack([gt["corners"], gt["corners"][0]])
                rr.log(
                    f"missed/tag_{gt['id']}",
                    rr.LineStrips2D(corners, colors=[255, 0, 0], radii=1.0),
                )

        # Log Thresholding/Segmentation if possible
        bin_img = locus.debug_threshold(img)
        rr.log("thresholded", rr.Image(bin_img))

        # Match for metrics
        correct = 0
        img_rmses = []
        matched_gt = set()

        for d in detections:
            best_dist = 1e9
            best_gt_idx = -1

            d_center = np.array(d.center)
            for idx, gt in enumerate(gt_tags):
                gt_center = np.mean(gt["corners"], axis=0)
                dist = np.linalg.norm(d_center - gt_center)
                if dist < best_dist:
                    best_dist = dist
                    best_gt_idx = idx

            if (
                best_dist < 20.0
                and best_gt_idx not in matched_gt
                and gt_tags[best_gt_idx]["id"] == d.id
            ):
                matched_gt.add(best_gt_idx)
                correct += 1

                # RMSE calculation
                gt_corners = gt_tags[best_gt_idx]["corners"]
                det_corners = np.array(d.corners)

                # Find best corner rotation and winding
                min_rmse = 1e9
                for ordering in [det_corners, det_corners[::-1]]:
                    for rot in range(4):
                        rotated = np.roll(ordering, rot, axis=0)
                        rmse = np.sqrt(np.mean(np.sum((gt_corners - rotated) ** 2, axis=1)))
                        if rmse < min_rmse:
                            min_rmse = rmse
                img_rmses.append(min_rmse)

        recall = correct / len(gt_tags) if gt_tags else 0
        all_recalls.append(recall)
        avg_rmse = np.mean(img_rmses) if img_rmses else 0
        if img_rmses:
            all_rmses.append(avg_rmse)

        print(f"Image {img_name}: Recall={recall:.4f}, RMSE={avg_rmse:.4f}")

    print(f"\nMean Recall: {np.mean(all_recalls):.4f}")
    if all_rmses:
        print(f"Mean RMSE: {np.mean(all_rmses):.4f}")

    rr.script_teardown(args)


if __name__ == "__main__":
    main()
