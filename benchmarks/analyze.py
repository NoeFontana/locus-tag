import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from utils import DatasetLoader, LocusWrapper


def analyze_tag_sizes(args):
    loader = DatasetLoader()
    for scenario in args.scenarios:
        if not loader.prepare_scenario(scenario):
            continue
        datasets = loader.find_datasets(scenario, ["tags"])
        for ds_name, _, gt_map in datasets:
            print(f"\nAnalyzing Tag Sizes in {ds_name}...")
            sizes = []
            for _, tags in gt_map.items():
                for gt in tags:
                    edges = [
                        np.linalg.norm(gt.corners[i] - gt.corners[(i + 1) % 4]) for i in range(4)
                    ]
                    sizes.append(np.mean(edges))

            if not sizes:
                continue
            sizes = np.array(sizes)
            print(f"  Total tags: {len(sizes)}")
            print(f"  Mean size: {np.mean(sizes):.1f} px")
            print(f"  Min size: {np.min(sizes):.1f} px")
            print(f"  Small (<10px): {np.sum(sizes < 10)} ({np.mean(sizes < 10) * 100:.1f}%)")


def diagnose_failures(args):
    loader = DatasetLoader()
    wrapper = LocusWrapper()

    for scenario in args.scenarios:
        if not loader.prepare_scenario(scenario):
            continue
        datasets = loader.find_datasets(scenario, ["tags"])

        for ds_name, img_dir, gt_map in datasets:
            print(f"\nDiagnosing failures in {ds_name}...")
            missed_by_size = []

            img_names = sorted(gt_map.keys())
            if args.limit:
                img_names = img_names[: args.limit]

            for img_name in tqdm(img_names):
                img_path = img_dir / img_name
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                gt_tags = gt_map[img_name]
                detections, _ = wrapper.detect(img)
                det_ids = {d["id"] for d in detections}

                for gt in gt_tags:
                    if gt.tag_id not in det_ids:
                        edges = [
                            np.linalg.norm(gt.corners[i] - gt.corners[(i + 1) % 4])
                            for i in range(4)
                        ]
                        missed_by_size.append(np.mean(edges))

            if missed_by_size:
                m = np.array(missed_by_size)
                print(f"  Missed {len(m)} tags.")
                print(f"  Mean missed size: {np.mean(m):.1f} px")
                print(f"  Min missed size: {np.min(m):.1f} px")


def profile_bottlenecks(args):
    from utils import generate_synthetic_image

    wrapper = LocusWrapper()

    img, _ = generate_synthetic_image(args.targets, (1280, 720), noise_sigma=args.noise)
    print(f"\nProfiling {args.targets} tags (noise={args.noise})...")

    stats_list = []
    for _ in range(args.iterations):
        _, stats = wrapper.detect(img)
        stats_list.append(stats)

    avg_thresh = np.mean([s.threshold_ms for s in stats_list])
    avg_seg = np.mean([s.segmentation_ms for s in stats_list])
    avg_quad = np.mean([s.quad_extraction_ms for s in stats_list])
    avg_dec = np.mean([s.decoding_ms for s in stats_list])
    avg_total = np.mean([s.total_ms for s in stats_list])

    print(f"  Thresholding: {avg_thresh:.2f} ms")
    print(f"  Segmentation: {avg_seg:.2f} ms")
    print(f"  Quad Extraction: {avg_quad:.2f} ms")
    print(f"  Decoding: {avg_dec:.2f} ms")
    print(f"  Total: {avg_total:.2f} ms")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    size = subparsers.add_parser("sizes")
    size.add_argument("--scenarios", nargs="+", default=["forward", "circle"])

    diag = subparsers.add_parser("diagnose")
    diag.add_argument("--scenarios", nargs="+", default=["forward"])
    diag.add_argument("--limit", type=int)

    prof = subparsers.add_parser("profile")
    prof.add_argument("--targets", type=int, default=50)
    prof.add_argument("--noise", type=float, default=0.0)
    prof.add_argument("--iterations", type=int, default=50)

    args = parser.parse_args()
    if args.cmd == "sizes":
        analyze_tag_sizes(args)
    elif args.cmd == "diagnose":
        diagnose_failures(args)
    elif args.cmd == "profile":
        profile_bottlenecks(args)


if __name__ == "__main__":
    main()
