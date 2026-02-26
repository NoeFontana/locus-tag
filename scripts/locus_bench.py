import argparse
import time
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from scripts.bench.utils import (
    AprilTagWrapper,
    DatasetLoader,
    HubBenchmarkLoader,
    LibraryWrapper,
    LocusWrapper,
    Metrics,
    OpenCVWrapper,
    generate_synthetic_image,
)

try:
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False


def run_hosted_benchmark(args):
    """Evaluates the detector against datasets hosted on the Hugging Face Hub.

    Args:
        args: Parsed command-line arguments containing configs, limit, and skip.
    """
    loader = HubBenchmarkLoader()
    configs = args.configs

    wrappers: list[LibraryWrapper] = []
    wrappers.append(LocusWrapper(decimation=args.decimation))
    if args.compare:
        wrappers.append(OpenCVWrapper())
        wrappers.append(AprilTagWrapper(nthreads=8))

    for config in configs:
        print(f"\nEvaluating {config} (from Hugging Face Hub)...")

        # Initialize stats for each wrapper
        wrapper_stats = {
            w.name: {"gt": 0, "det": 0, "err_sum": 0.0, "latency": []} for w in wrappers
        }

        subset_stream = loader.stream_subset(config)
        pbar = tqdm(desc="Processing Images")

        for idx, (_name, img, gt_tags) in enumerate(subset_stream):
            if args.skip and idx < args.skip:
                continue
            if args.limit and (idx - args.skip) >= args.limit:
                break

            for wrapper in wrappers:
                start = time.perf_counter()
                detections, _ = wrapper.detect(img)
                latency = (time.perf_counter() - start) * 1000

                correct, err_sum, _ = Metrics.match_detections(detections, gt_tags)

                stats = wrapper_stats[wrapper.name]
                stats["latency"].append(latency)
                stats["gt"] += len(gt_tags)
                stats["det"] += correct
                stats["err_sum"] += err_sum

            pbar.update(1)

        pbar.close()

        # Output results for all wrappers
        for wrapper in wrappers:
            stats = wrapper_stats[wrapper.name]
            recall = (stats["det"] / stats["gt"] * 100) if stats["gt"] > 0 else 0
            avg_err = (stats["err_sum"] / stats["det"]) if stats["det"] > 0 else 0
            avg_lat = np.mean(stats["latency"])

            print(
                f"  {wrapper.name:<10} | Recall: {recall:>6.2f}% | RMSE: {avg_err:>6.4f} px | Latency: {avg_lat:>6.2f} ms"
            )


def run_real_benchmark(args):
    """Evaluates the detector against local real-world datasets (ICRA).

    Args:
        args: Parsed command-line arguments containing scenarios, types, and limit.
    """
    loader = DatasetLoader()
    scenarios = args.scenarios
    types = args.types

    wrappers: list[LibraryWrapper] = []
    wrappers.append(LocusWrapper(decimation=args.decimation))
    if args.compare:
        wrappers.append(OpenCVWrapper())
        wrappers.append(AprilTagWrapper(nthreads=8))

    for scenario in scenarios:
        if not loader.prepare_icra(scenario):
            continue

        datasets = loader.find_datasets(scenario, types)
        for ds_name, img_dir, gt_map in datasets:
            print(f"\nEvaluating {ds_name}...")

            img_names = sorted(gt_map.keys())
            if args.skip:
                img_names = img_names[args.skip :]
            if args.limit:
                img_names = img_names[: args.limit]

            for wrapper in wrappers:
                stats: dict[str, Any] = {"gt": 0, "det": 0, "err_sum": 0.0, "latency": []}

                for img_name in tqdm(img_names, desc=f"{wrapper.name:<10}"):
                    img_path = img_dir / img_name
                    if not img_path.exists():
                        continue

                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    gt_tags = gt_map[img_name]

                    start = time.perf_counter()
                    detections, _ = wrapper.detect(img)
                    stats["latency"].append((time.perf_counter() - start) * 1000)

                    correct, err_sum, _ = Metrics.match_detections(detections, gt_tags)
                    stats["gt"] += len(gt_tags)
                    stats["det"] += correct
                    stats["err_sum"] += err_sum

                recall = (stats["det"] / stats["gt"] * 100) if stats["gt"] > 0 else 0
                avg_err = (stats["err_sum"] / stats["det"]) if stats["det"] > 0 else 0
                avg_lat = np.mean(stats["latency"])

                print(
                    f"  {wrapper.name:<10} | Recall: {recall:>6.2f}% | RMSE: {avg_err:>6.4f} px | Latency: {avg_lat:>6.2f} ms"
                )


def run_synthetic_benchmark(args):
    """Evaluates the detector against procedurally generated synthetic images.

    Args:
        args: Parsed command-line arguments containing targets and noise.
    """
    wrappers: list[LibraryWrapper] = []
    wrappers.append(LocusWrapper(decimation=args.decimation))
    if args.compare:
        wrappers.append(OpenCVWrapper())
        wrappers.append(AprilTagWrapper())

    counts = [int(x) for x in args.targets.split(",")]
    res = (1280, 720)

    print(
        f"{'Targets':<8} | {'Lib':<10} | {'Latency (ms)':<12} | {'Recall':<8} | {'CornerErr':<10}"
    )
    print("-" * 60)

    for count in counts:
        img, gt_tags = generate_synthetic_image(count, res, noise_sigma=args.noise)

        for wrapper in wrappers:
            latencies = []
            detections = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                detections, _ = wrapper.detect(img)
                latencies.append((time.perf_counter() - start) * 1000)

            correct, err_sum, _ = Metrics.match_detections(detections, gt_tags)
            recall = (correct / len(gt_tags) * 100) if gt_tags else 0
            avg_err = (err_sum / correct) if correct > 0 else 0

            print(
                f"{count:<8} | {wrapper.name:<10} | {np.mean(latencies):<12.3f} | {recall:<7.1f}% | {avg_err:<10.3f}"
            )


def analyze_tag_sizes(args):
    """Analyzes the distribution of tag sizes within a dataset.

    Args:
        args: Parsed command-line arguments containing scenarios.
    """
    loader = DatasetLoader()
    for scenario in args.scenarios:
        if not loader.prepare_icra(scenario):
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
            sizes_arr = np.array(sizes)
            print(f"  Total tags: {len(sizes_arr)}")
            print(f"  Mean size: {np.mean(sizes_arr):.1f} px")
            print(f"  Min size: {np.min(sizes_arr):.1f} px")
            print(
                f"  Small (<10px): {np.sum(sizes_arr < 10)} ({np.mean(sizes_arr < 10) * 100:.1f}%)"
            )


def profile_bottlenecks(args):
    """Profiles individual stages of the Locus pipeline to identify bottlenecks.

    Args:
        args: Parsed command-line arguments containing targets and iterations.
    """
    wrapper = LocusWrapper()
    res = (1280, 720)
    img, _ = generate_synthetic_image(args.targets, res, noise_sigma=args.noise)
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


def prepare_datasets(args):
    """Downloads and prepares all required benchmarking datasets.

    Args:
        args: Parsed command-line arguments.
    """
    loader = DatasetLoader()
    print("Preparing datasets...")
    loader.prepare_all()
    print("Done.")


def main():
    """Main entry point for the Locus Unified Benchmarking CLI."""
    parser = argparse.ArgumentParser(description="Locus Unified Benchmarking CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Run Command
    run = subparsers.add_parser("run", help="Run benchmarks")
    run_sub = run.add_subparsers(dest="mode", required=True)

    real = run_sub.add_parser("real", help="Run on real datasets")
    real.add_argument("--scenarios", nargs="+", default=["forward"])
    real.add_argument("--types", nargs="+", choices=["tags", "checkerboard"], default=["tags"])
    real.add_argument("--limit", type=int)
    real.add_argument("--skip", type=int, default=0)
    real.add_argument("--compare", action="store_true")
    real.add_argument("--decimation", type=int, default=1)

    synth = run_sub.add_parser("synthetic", help="Run on synthetic images")
    synth.add_argument("--targets", type=str, default="1,10,50,100")
    synth.add_argument("--noise", type=float, default=0.0)
    synth.add_argument("--iterations", type=int, default=10)
    synth.add_argument("--compare", action="store_true")
    synth.add_argument("--decimation", type=int, default=1)

    hosted = run_sub.add_parser("hosted", help="Run on hosted datasets (Hub)")
    hosted.add_argument("--configs", nargs="+", required=True)
    hosted.add_argument("--limit", type=int)
    hosted.add_argument("--skip", type=int, default=0)
    hosted.add_argument("--compare", action="store_true")
    hosted.add_argument("--decimation", type=int, default=1)

    # Analyze Command
    analyze = subparsers.add_parser("analyze", help="Analyze datasets")
    analyze.add_argument("--scenarios", nargs="+", default=["forward", "circle"])

    # Profile Command
    profile = subparsers.add_parser("profile", help="Profile pipeline bottlenecks")
    profile.add_argument("--targets", type=int, default=50)
    profile.add_argument("--noise", type=float, default=0.0)
    profile.add_argument("--iterations", type=int, default=50)

    # Prepare Command
    subparsers.add_parser("prepare", help="Download and prepare all datasets")

    args = parser.parse_args()

    if args.cmd == "run":
        if args.mode == "real":
            run_real_benchmark(args)
        elif args.mode == "hosted":
            run_hosted_benchmark(args)
        else:
            run_synthetic_benchmark(args)
    elif args.cmd == "analyze":
        analyze_tag_sizes(args)
    elif args.cmd == "profile":
        profile_bottlenecks(args)
    elif args.cmd == "prepare":
        prepare_datasets(args)


if __name__ == "__main__":
    main()
