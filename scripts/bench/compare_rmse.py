import argparse

import cv2
import numpy as np
from tqdm import tqdm

from scripts.bench.utils import (
    AprilTagWrapper,
    DatasetLoader,
    LocusWrapper,
    Metrics,
)


def compute_errors(detections, gt_tags):
    """
    Returns a dict mapping gt_index -> rmse for successfully matched tags.
    """
    matched_errors = {}

    for det in detections:
        det_center = np.array(det["center"])
        best_gt_idx = -1
        min_dist = float("inf")

        for idx, gt in enumerate(gt_tags):
            if idx in matched_errors or gt.tag_id != det["id"]:
                continue
            gt_center = np.mean(gt.corners, axis=0)
            dist = np.linalg.norm(det_center - gt_center)
            if dist < min_dist:
                min_dist = dist
                best_gt_idx = idx

        if best_gt_idx != -1 and min_dist < 20.0:
            # Found a match, compute RMSE using standardized Metric
            det_corners = np.array(det["corners"])
            gt_corners = gt_tags[best_gt_idx].corners

            error = Metrics.compute_corner_error(det_corners, gt_corners)
            matched_errors[best_gt_idx] = error

    return matched_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scenario", default="forward")
    args = parser.parse_args()

    loader = DatasetLoader()
    if not loader.prepare_scenario(args.scenario):
        print("Failed to prepare dataset")
        return

    datasets = loader.find_datasets(args.scenario, ["tags"])

    locus_detector = LocusWrapper()
    apriltag_detector = AprilTagWrapper(nthreads=8)

    # Storage for all errors
    # list of (locus_err, april_err) for common tags
    common_errors = []
    # list of locus_err for tags only locus found
    locus_only_errors = []
    # list of april_err for tags only april found
    april_only_errors = []

    for ds_name, img_dir, gt_map in datasets:
        print(f"Processing {ds_name}...")
        img_names = sorted(gt_map.keys())
        if args.limit:
            img_names = img_names[: args.limit]

        for img_name in tqdm(img_names):
            img_path = img_dir / img_name
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            gt = gt_map[img_name]

            # Run Detectors
            locus_dets, _ = locus_detector.detect(img)
            april_dets, _ = apriltag_detector.detect(img)

            # Compute errors for each
            locus_matches = compute_errors(locus_dets, gt)  # gt_idx -> rmse
            april_matches = compute_errors(april_dets, gt)  # gt_idx -> rmse

            # Compare
            all_gt_indices = set(locus_matches.keys()) | set(april_matches.keys())

            for idx in all_gt_indices:
                in_locus = idx in locus_matches
                in_april = idx in april_matches

                if in_locus and in_april:
                    common_errors.append((locus_matches[idx], april_matches[idx]))
                elif in_locus:
                    locus_only_errors.append(locus_matches[idx])
                elif in_april:
                    april_only_errors.append(april_matches[idx])

    # Statistics
    print("\n" + "=" * 60)
    print(f"RMSE ANALYSIS (Scenario: {args.scenario})")
    print("=" * 60)

    # Intersection
    if common_errors:
        l_errs, a_errs = zip(*common_errors)
        rmse_l_common = np.mean(l_errs)
        rmse_a_common = np.mean(a_errs)
        print(f"INTERSECTION (Count: {len(common_errors)})")
        print(f"  Locus RMSE:    {rmse_l_common:.4f} px")
        print(f"  AprilTag RMSE: {rmse_a_common:.4f} px")
        print(f"  Delta:         {rmse_l_common - rmse_a_common:+.4f} px")
    else:
        print("No intersection found.")

    print("-" * 60)

    # Locus Only
    if locus_only_errors:
        rmse_l_only = np.mean(locus_only_errors)
        print(f"LOCUS ONLY (Count: {len(locus_only_errors)})")
        print(f"  RMSE:          {rmse_l_only:.4f} px")
        print(f"  (These are tags AprilTag missed)")
    else:
        print("Locus detected no unique tags.")

    print("-" * 60)

    # AprilTag Only
    if april_only_errors:
        rmse_a_only = np.mean(april_only_errors)
        print(f"APRILTAG ONLY (Count: {len(april_only_errors)})")
        print(f"  RMSE:          {rmse_a_only:.4f} px")
    else:
        print("AprilTag detected no unique tags.")

    print("=" * 60)


if __name__ == "__main__":
    main()
