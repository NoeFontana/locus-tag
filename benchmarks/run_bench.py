import argparse
import time
from pathlib import Path

import cv2
import locus
import numpy as np

try:
    import rerun as rr

    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
from tqdm import tqdm
from utils import (
    AprilTagWrapper,
    DatasetLoader,
    EvalResult,
    LocusWrapper,
    Metrics,
    OpenCVWrapper,
    TagGroundTruth,
)


def run_real_benchmark(args):
    loader = DatasetLoader()
    scenarios = args.scenarios
    types = args.types

    results = []

    locus_config = locus.DetectorConfig(
        threshold_tile_size=4,
        quad_min_area=4,
        enable_bilateral=False,
    )
    wrappers = [LocusWrapper(config=locus_config, decimation=1)]
    if args.compare:
        wrappers.append(OpenCVWrapper())
        wrappers.append(AprilTagWrapper(nthreads=8))

    for scenario in scenarios:
        if not loader.prepare_scenario(scenario):
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
                stats = {"gt": 0, "det": 0, "err_sum": 0.0, "latency": []}

                for img_name in tqdm(img_names, desc=f"{wrapper.name:<10}"):
                    img_path = img_dir / img_name
                    if not img_path.exists():
                        continue

                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    gt_tags = gt_map[img_name]

                    # Warmup and timed run
                    start = time.perf_counter()
                    detections, _ = wrapper.detect(img)
                    stats["latency"].append((time.perf_counter() - start) * 1000)

                    correct, err_sum, matched_cnt = Metrics.match_detections(detections, gt_tags)
                    stats["gt"] += len(gt_tags)
                    stats["det"] += correct
                    stats["err_sum"] += err_sum

                    if args.rerun and RERUN_AVAILABLE and wrapper.name == "Locus":
                        log_to_rerun(ds_name, img_name, img, detections, gt_tags)

                recall = (stats["det"] / stats["gt"] * 100) if stats["gt"] > 0 else 0
                avg_err = (stats["err_sum"] / stats["det"]) if stats["det"] > 0 else 0
                avg_lat = np.mean(stats["latency"])

                print(
                    f"  {wrapper.name:<10} | Recall: {recall:>6.2f}% | RMSE: {avg_err:>6.4f} px | Latency: {avg_lat:>6.2f} ms"
                )


def run_synthetic_benchmark(args):
    locus_config = locus.DetectorConfig(
        threshold_tile_size=4,
        quad_min_area=4,
        enable_bilateral=False,
    )
    wrappers = [LocusWrapper(config=locus_config, decimation=1)]
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


def generate_synthetic_image(num_tags, res, noise_sigma=0.0):
    img = np.zeros((res[1], res[0]), dtype=np.uint8) + 128
    cols = int(np.ceil(np.sqrt(num_tags)))
    rows = int(np.ceil(num_tags / cols))
    cell_w, cell_h = res[0] // cols, res[1] // rows
    tag_size = int(min(cell_w, cell_h) * 0.6)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    gt_data = []

    for i in range(num_tags):
        r, c = i // cols, i % cols
        x = c * cell_w + (cell_w - tag_size) // 2
        y = r * cell_h + (cell_h - tag_size) // 2

        tag_id = i % 587
        tag_img = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_size)

        padding = tag_size // 5
        img[y - padding : y + tag_size + padding, x - padding : x + tag_size + padding] = 255
        img[y : y + tag_size, x : x + tag_size] = tag_img

        ts = tag_size - 1
        corners = np.array([[x, y], [x + ts, y], [x + ts, y + ts], [x, y + ts]], dtype=np.float32)
        gt_data.append(TagGroundTruth(tag_id=tag_id, corners=corners))

    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img, gt_data


def log_to_rerun(ds_name, img_name, img, detections, gt_tags):
    base = f"benchmarks/{ds_name}/{Path(img_name).stem}"
    rr.set_time_sequence("frame_idx", 0)
    rr.log(f"{base}/image", rr.Image(img))

    for gt in gt_tags:
        c = np.vstack([gt.corners, gt.corners[0]])
        rr.log(f"{base}/gt/tag_{gt.tag_id}", rr.LineStrips2D(c, colors=[0, 255, 0]))

    for det in detections:
        c = np.array(det["corners"])
        c = np.vstack([c, c[0]])
        rr.log(f"{base}/det/tag_{det['id']}", rr.LineStrips2D(c, colors=[255, 0, 0]))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    real = subparsers.add_parser("real")
    real.add_argument("--scenarios", nargs="+", default=["forward"])
    real.add_argument("--types", nargs="+", choices=["tags", "checkerboard"], default=["tags"])
    real.add_argument("--limit", type=int)
    real.add_argument("--skip", type=int, default=0)
    real.add_argument("--compare", action="store_true")
    real.add_argument("--rerun", action="store_true")

    synth = subparsers.add_parser("synthetic")
    synth.add_argument("--targets", type=str, default="1,10,50,100")
    synth.add_argument("--noise", type=float, default=0.0)
    synth.add_argument("--iterations", type=int, default=10)
    synth.add_argument("--compare", action="store_true")

    args = parser.parse_args()

    if args.mode == "real":
        if args.rerun and RERUN_AVAILABLE:
            rr.init("Locus Real Benchmark", spawn=True)
        run_real_benchmark(args)
    else:
        run_synthetic_benchmark(args)


if __name__ == "__main__":
    main()
