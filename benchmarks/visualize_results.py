import argparse
from pathlib import Path

import cv2
import icra2020
import locus
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def run_visualization(scenario, types=("tags",), limit=None):
    print(f"Running visualization for scenario: {scenario}")
    dataset_handler = icra2020.Icra2020Dataset()
    if not dataset_handler.prepare_scenario(scenario):
        print(f"Failed to prepare scenario {scenario}")
        return

    datasets = dataset_handler.find_datasets(scenario, list(types))
    if not datasets:
        print(f"No datasets found for {scenario}")
        return

    # Use the first dataset for plotting
    ds_name, img_dir, gt_map = datasets[0]

    # Sort images by name to maintain sequence
    img_names = sorted(gt_map.keys())
    if limit:
        img_names = img_names[:limit]

    detector = locus.Detector(
        quad_min_area=16,
        quad_max_aspect_ratio=20.0,
        quad_min_edge_score=0.5,
        enable_bilateral=False,
        enable_adaptive_window=False,
    )

    results = []

    for img_name in tqdm(img_names, desc=f"Processing {scenario}"):
        img_path = img_dir / img_name
        gt_tags = gt_map[img_name]

        # We reuse the process_image logic from icra2020 or re-implement it simply
        # For visualization script, let's keep it simple here.
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        detections, stats = detector.detect_with_stats(img)

        correct = 0
        error_sum = 0.0
        error_count = 0

        matched_gt_indices = set()
        for det in detections:
            min_dist = float("inf")
            best_gt_idx = -1

            for idx, gt in enumerate(gt_tags):
                gt_center = np.mean(gt.corners, axis=0)
                dist = np.linalg.norm(np.array(det.center) - gt_center)
                if dist < min_dist:
                    min_dist = dist
                    best_gt_idx = idx

            if (
                min_dist < 20.0
                and best_gt_idx != -1
                and gt_tags[best_gt_idx].tag_id == det.id
                and best_gt_idx not in matched_gt_indices
            ):
                matched_gt_indices.add(best_gt_idx)
                correct += 1

                # Corner error
                det_corners = np.array(det.corners, dtype=np.float32)
                best_err = float("inf")
                for ordering in [det_corners, det_corners[::-1]]:
                    for rot in range(4):
                        rotated = np.roll(ordering, rot, axis=0)
                        err = np.sqrt(
                            np.mean(np.sum((gt_tags[best_gt_idx].corners - rotated) ** 2, axis=1))
                        )
                        best_err = min(best_err, err)

                error_sum += best_err
                error_count += 1

        recall = (correct / len(gt_tags)) if gt_tags else 0
        avg_error = (error_sum / error_count) if error_count > 0 else np.nan

        results.append({"name": img_name, "recall": recall, "error": avg_error})

    # Prepare plotting data
    recalls = [r["recall"] * 100 for r in results]
    errors = [r["error"] for r in results]
    indices = np.arange(len(results))

    if scenario == "forward":
        # Index 0 is far (18m), last index is near (0.5m)
        # Sequence typically goes from far to near.
        # To match the paper plot (increasing distance), we keep index 0 at 18.0
        x_vals = np.linspace(18.0, 0.5, len(results))
        x_label = "Distance to board in m"
    elif scenario == "circle":
        # -90 to 90 degrees as per paper
        x_vals = np.linspace(-90, 90, len(results))
        x_label = "Angle between camera and board in °"
    else:
        x_vals = indices
        x_label = "Image Index"

    # Create plots
    plt.figure(figsize=(10, 8), dpi=150)

    # Plot 1: Recall
    plt.subplot(2, 1, 1)
    # Re-calculate to 0..1 to match paper exactly
    plt.plot(x_vals, [r / 100.0 for r in recalls], linestyle="-", color="#1f77b4", linewidth=1.5)
    plt.title(f"Scenario: {scenario.capitalize()}")
    plt.ylabel("Detection rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: RMS Error
    plt.subplot(2, 1, 2)
    valid_mask = ~np.isnan(errors)
    if any(valid_mask):
        plt.plot(
            np.array(x_vals)[valid_mask],
            np.array(errors)[valid_mask],
            linestyle="-",
            color="#d62728",
            linewidth=1.5,
        )
    plt.xlabel(x_label)
    plt.ylabel("RMS Error in px")
    plt.ylim(0, 1.25)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    output_dir = Path("benchmarks/plots")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{scenario}_performance.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=["forward", "circle"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    for s in args.scenarios:
        run_visualization(s, limit=args.limit)
