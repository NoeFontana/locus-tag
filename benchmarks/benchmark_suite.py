import argparse
import time
from pathlib import Path

import cv2
import locus
import numpy as np
from pupil_apriltags import Detector as AprilTagDetector


class BenchmarkSuite:
    def __init__(self, resolution=(1280, 720)):
        self.resolution = resolution
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.at_detector = AprilTagDetector(families="tag36h11", nthreads=1)
        self.ocv_detector = cv2.aruco.ArucoDetector(self.dictionary, cv2.aruco.DetectorParameters())

    def generate_image(self, num_tags, noise_sigma=0.0, blur_k=0):
        img = np.zeros((self.resolution[1], self.resolution[0]), dtype=np.uint8) + 128
        h, w = self.resolution[1], self.resolution[0]

        # Grid layout for tags
        cols = int(np.ceil(np.sqrt(num_tags)))
        rows = int(np.ceil(num_tags / cols))

        cell_w = w // cols
        cell_h = h // rows
        tag_size = int(min(cell_w, cell_h) * 0.6)

        gt_data = []

        for i in range(num_tags):
            r = i // cols
            c = i % cols

            # Center of the cell
            cx = c * cell_w + cell_w // 2
            cy = r * cell_h + cell_h // 2

            # Random offset within cell
            ox = np.random.randint(-cell_w // 10, cell_w // 10)
            oy = np.random.randint(-cell_h // 10, cell_h // 10)

            x = cx + ox - tag_size // 2
            y = cy + oy - tag_size // 2

            tag_id = i % 3
            tag_img = cv2.aruco.generateImageMarker(self.dictionary, tag_id, tag_size)

            # White quiet zone
            padding = tag_size // 5
            img[y - padding : y + tag_size + padding, x - padding : x + tag_size + padding] = 255
            # Tag
            img[y : y + tag_size, x : x + tag_size] = tag_img

            ts = tag_size - 1
            gt_corners = np.array(
                [[x, y], [x + ts, y], [x + ts, y + ts], [x, y + ts]], dtype=np.float32
            )
            gt_data.append({"id": tag_id, "corners": gt_corners})

        if blur_k > 0:
            img = cv2.GaussianBlur(img, (blur_k * 2 + 1, blur_k * 2 + 1), 0)

        if noise_sigma > 0:
            noise = np.random.normal(0, noise_sigma, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img, gt_data

    @staticmethod
    def compute_corner_error(detected, gt):
        if detected is None or len(detected) != 4:
            return float("inf")

        det = np.array(detected).reshape(4, 2)
        min_err = float("inf")

        for rot in range(4):
            rotated = np.roll(det, rot, axis=0)
            err = np.mean(np.linalg.norm(rotated - gt, axis=1))
            min_err = min(min_err, err)

        det_flipped = det[::-1]
        for rot in range(4):
            rotated = np.roll(det_flipped, rot, axis=0)
            err = np.mean(np.linalg.norm(rotated - gt, axis=1))
            min_err = min(min_err, err)

        return min_err

    @staticmethod
    def match_and_compute_errors(detections, gt_list, compute_err_fn, id_attr="id"):
        available_detections = list(detections)
        found_ids = 0
        errors = []
        for gt in gt_list:
            matches = [d for d in available_detections if getattr(d, id_attr) == gt["id"]]
            if matches:
                best_match = min(
                    matches,
                    key=lambda d: np.linalg.norm(
                        np.array(d.center) - np.mean(gt["corners"], axis=0)
                    ),
                )
                errors.append(compute_err_fn(best_match.corners, gt["corners"]))
                available_detections.remove(best_match)
                found_ids += 1
            else:
                errors.append(float("inf"))
        return found_ids, errors

    def run_benchmark_locus(self, img, gt_list, iterations):
        # Warmup
        _ = locus.detect_tags(img)
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            detections = locus.detect_tags(img)
            latencies.append(time.perf_counter() - start)

        # Accuracy check (once)
        f, e = self.match_and_compute_errors(
            locus.detect_tags(img), gt_list, self.compute_corner_error, "id"
        )
        return np.mean(latencies), f / len(gt_list), np.median(e) if e else float("inf")

    def run_benchmark_locus_gradient(self, img, gt_list, iterations):
        _ = locus.detect_tags_gradient(img)
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            detections = locus.detect_tags_gradient(img)
            latencies.append(time.perf_counter() - start)

        f, e = self.match_and_compute_errors(
            locus.detect_tags_gradient(img), gt_list, self.compute_corner_error, "id"
        )
        return np.mean(latencies), f / len(gt_list), np.median(e) if e else float("inf")

    def run_benchmark_apriltag(self, img, gt_list, iterations):
        _ = self.at_detector.detect(img)
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            detections = self.at_detector.detect(img)
            latencies.append(time.perf_counter() - start)

        f, e = self.match_and_compute_errors(
            self.at_detector.detect(img), gt_list, self.compute_corner_error, "tag_id"
        )
        return np.mean(latencies), f / len(gt_list), np.median(e) if e else float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=str, default="1,5,10,20")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--noise", type=float, default=0.0)
    args = parser.parse_args()

    target_counts = [int(x) for x in args.targets.split(",")]
    suite = BenchmarkSuite()

    print(
        f"{'Targets':<8} | {'Lib':<10} | {'Latency (ms)':<12} | {'Recall':<8} | {'CornerErr':<10}"
    )
    print("-" * 60)

    benchmarks = [
        ("Locus", suite.run_benchmark_locus),
        ("LocusGrad", suite.run_benchmark_locus_gradient),
        ("AprilTag", suite.run_benchmark_apriltag),
    ]

    for count in target_counts:
        img, gt = suite.generate_image(count, noise_sigma=args.noise)
        for name, bench_fn in benchmarks:
            lat, recall, err = bench_fn(img, gt, args.iterations)
            err_str = f"{err:.3f}" if err != float("inf") else "N/A"
            print(
                f"{count:<8} | {name:<10} | {lat * 1000:<12.3f} | {recall * 100:<7.1f}% | {err_str:<10}"
            )


if __name__ == "__main__":
    main()
