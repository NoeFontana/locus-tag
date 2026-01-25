import argparse
import time

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

                for i, img_name in enumerate(tqdm(img_names, desc=f"{wrapper.name:<10}")):
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
                        log_to_rerun(ds_name, img_name, img, detections, gt_tags, frame_idx=i)

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
            # Try to get stats from the last run
            _, stats = wrapper.detect(img)
            if stats is not None:
                print(
                    f"       [Stats] Thresh: {stats.threshold_ms:.2f}ms, Seg: {stats.segmentation_ms:.2f}ms, Quad: {stats.quad_extraction_ms:.2f}ms, Decode: {stats.decoding_ms:.2f}ms"
                )
                print(
                    f"       [Stats] Candidates: {stats.num_candidates}, Detections: {stats.num_detections}"
                )
                print(
                    f"       [Stats] Rejected: Contrast: {stats.num_rejected_by_contrast}, Hamming: {stats.num_rejected_by_hamming}"
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


def log_to_rerun(ds_name, img_name, img, detections, gt_tags, frame_idx=0):
    rr.set_time(timeline="frame_idx", sequence=frame_idx)
    rr.log("benchmark/image", rr.Image(img))

    # 1. Log Ground Truth (Green) - Nested under image path
    if gt_tags:
        gt_strips = []
        gt_ids = []
        gt_centers = []
        for gt in gt_tags:
            c = np.vstack([gt.corners, gt.corners[0]])
            gt_strips.append(c)
            gt_ids.append(f"GT:{gt.tag_id}")
            gt_centers.append(np.mean(gt.corners, axis=0))

        rr.log(
            "benchmark/image/gt",
            rr.LineStrips2D(gt_strips, colors=[0, 255, 0], radii=1.0),
        )
        rr.log(
            "benchmark/image/gt/labels",
            rr.Points2D(gt_centers, labels=gt_ids, colors=[0, 255, 0], radii=2.0),
        )

    # 2. Log Detections (Red) - Nested under image path
    if detections:
        det_strips = []
        det_ids = []
        det_centers = []
        for det in detections:
            c = np.array(det["corners"])
            c = np.vstack([c, c[0]])
            det_strips.append(c)
            det_ids.append(f"ID:{det['id']}")
            det_centers.append(np.array(det["center"]))

        rr.log(
            "benchmark/image/det",
            rr.LineStrips2D(det_strips, colors=[0, 0, 255], radii=0.5),
        )
        rr.log(
            "benchmark/image/det/labels",
            rr.Points2D(det_centers, labels=det_ids, colors=[0, 0, 255], radii=1.0),
        )


def main():
    parser = argparse.ArgumentParser()
    # Add to main parser for global flags (e.g. run_bench.py --serve real)
    rr.script_add_args(parser)

    subparsers = parser.add_subparsers(dest="mode", required=True)

    real = subparsers.add_parser("real")
    real.add_argument("--scenarios", nargs="+", default=["forward"])
    real.add_argument("--types", nargs="+", choices=["tags", "checkerboard"], default=["tags"])
    real.add_argument("--limit", type=int)
    real.add_argument("--skip", type=int, default=0)
    real.add_argument("--compare", action="store_true")
    real.add_argument("--rerun", action="store_true")
    # Add to subparser for local flags (e.g. run_bench.py real --serve)
    rr.script_add_args(real)

    synth = subparsers.add_parser("synthetic")
    synth.add_argument("--targets", type=str, default="1,10,50,100")
    synth.add_argument("--noise", type=float, default=0.0)
    synth.add_argument("--iterations", type=int, default=10)
    synth.add_argument("--compare", action="store_true")
    # Add to subparser for local flags (e.g. run_bench.py synthetic --serve)
    rr.script_add_args(synth)

    args = parser.parse_args()

    if args.mode == "real":
        if args.rerun and RERUN_AVAILABLE:
            rr.script_setup(args, "locus_real_benchmark")
        run_real_benchmark(args)
        if args.rerun and RERUN_AVAILABLE:
            rr.script_teardown(args)
    else:
        run_synthetic_benchmark(args)


if __name__ == "__main__":
    main()
