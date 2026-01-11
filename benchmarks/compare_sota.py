import argparse
import time
from pathlib import Path

import cv2
import locus
import numpy as np
from pupil_apriltags import Detector as AprilTagDetector


def generate_synthetic_data(num_images=5, size=(640, 480)):
    """Generate synthetic images with known ground truth tags."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    images = []
    ground_truth = []

    for i in range(num_images):
        tag_id = i % 3
        tag_size = 120
        tag_img = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_size)

        img = np.zeros(size, dtype=np.uint8) + 128
        x, y = 100 + i * 40, 100 + i * 20

        padding = 20
        img[y - padding : y + tag_size + padding, x - padding : x + tag_size + padding] = 255
        img[y : y + tag_size, x : x + tag_size] = tag_img

        images.append(img)

        # Ground truth corners (top-left, top-right, bottom-right, bottom-left)
        # Tag occupies pixels [x, x+tag_size-1] x [y, y+tag_size-1]
        # Corners are at the edges of the tag, so the outside corner is at x-0.5, y-0.5
        # But we measure at pixel centers, so the corner pixel is at x, x+tag_size-1
        ts = tag_size - 1  # rightmost/bottommost pixel of tag
        gt_corners = np.array(
            [[x, y], [x + ts, y], [x + ts, y + ts], [x, y + ts]],
            dtype=np.float32,
        )
        ground_truth.append({"id": tag_id, "corners": gt_corners})

    return images, ground_truth


def compute_corner_error(detected_corners, gt_corners):
    """Compute mean Euclidean distance between detected and ground truth corners."""
    if detected_corners is None or len(detected_corners) != 4:
        return float("inf")
    detected = np.array(detected_corners).reshape(4, 2)
    gt = np.array(gt_corners).reshape(4, 2)

    # Find best matching rotation and flip (corners may be in different order/winding)
    min_error = float("inf")
    for rotation in range(4):
        rotated = np.roll(detected, rotation, axis=0)
        error = np.mean(np.linalg.norm(rotated - gt, axis=1))
        min_error = min(min_error, error)
    # Also try reversed winding
    detected_flipped = detected[::-1]
    for rotation in range(4):
        rotated = np.roll(detected_flipped, rotation, axis=0)
        error = np.mean(np.linalg.norm(rotated - gt, axis=1))
        min_error = min(min_error, error)
    return min_error


def benchmark_locus(images, ground_truth, iterations):
    latencies = []
    id_correct = 0
    corner_errors = []

    _ = locus.detect_tags(images[0])  # Warm up

    for img, gt in zip(images, ground_truth):
        for _ in range(iterations):
            start = time.perf_counter()
            detections = locus.detect_tags(img)
            latencies.append(time.perf_counter() - start)

            if detections:
                det = detections[0]
                if det.id == gt["id"]:
                    id_correct += 1
                corner_errors.append(compute_corner_error(det.corners, gt["corners"]))
            else:
                corner_errors.append(float("inf"))

    return latencies, id_correct / (len(images) * iterations), np.median(corner_errors)


def benchmark_opencv(images, ground_truth, iterations):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    _, _, _ = detector.detectMarkers(images[0])  # Warm up

    latencies = []
    id_correct = 0
    corner_errors = []

    for img, gt in zip(images, ground_truth):
        for _ in range(iterations):
            start = time.perf_counter()
            corners, ids, _ = detector.detectMarkers(img)
            latencies.append(time.perf_counter() - start)

            if ids is not None and len(ids) > 0:
                detected_id = ids[0][0]
                if detected_id == gt["id"]:
                    id_correct += 1
                # OpenCV corners are in shape (1, 4, 2)
                corner_errors.append(compute_corner_error(corners[0].reshape(4, 2), gt["corners"]))
            else:
                corner_errors.append(float("inf"))

    return latencies, id_correct / (len(images) * iterations), np.median(corner_errors)


def benchmark_apriltag(images, ground_truth, iterations):
    at_detector = AprilTagDetector(families="tag36h11", nthreads=1)

    _ = at_detector.detect(images[0])  # Warm up

    latencies = []
    id_correct = 0
    corner_errors = []

    for img, gt in zip(images, ground_truth):
        for _ in range(iterations):
            start = time.perf_counter()
            detections = at_detector.detect(img)
            latencies.append(time.perf_counter() - start)

            if detections:
                det = detections[0]
                if det.tag_id == gt["id"]:
                    id_correct += 1
                # apriltag corners are in a different order, try all rotations
                detected_corners = det.corners.reshape(4, 2)
                corner_errors.append(compute_corner_error(detected_corners, gt["corners"]))
            else:
                corner_errors.append(float("inf"))

    return latencies, id_correct / (len(images) * iterations), np.median(corner_errors)


def main():
    parser = argparse.ArgumentParser(description="Fair Locus SOTA Benchmark")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations per image")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    args = parser.parse_args()

    if args.synthetic:
        print("Generating synthetic data with ground truth...")
        images, ground_truth = generate_synthetic_data()
    else:
        print("Real data mode not yet supported with ground truth. Using synthetic.")
        images, ground_truth = generate_synthetic_data()

    print(f"Benchmarking with {len(images)} images, {args.iterations} iterations each.")
    print("All libraries run on same resolution (640x480), no downsampling.\n")

    libraries = [
        ("locus", benchmark_locus),
        ("opencv", benchmark_opencv),
        ("apriltag", benchmark_apriltag),
    ]

    results = {}
    for name, func in libraries:
        print(f"Running {name}...")
        lat, id_acc, corner_err = func(images, ground_truth, args.iterations)
        results[name] = {"latencies": lat, "id_accuracy": id_acc, "corner_error": corner_err}

    # Print summary
    print("\n=== Fair Benchmark Results ===")
    print(f"{'Library':<12} | {'Median (ms)':<12} | {'ID Accuracy':<12} | {'Corner Err (px)':<15}")
    print("-" * 60)
    for name, data in results.items():
        lat_ms = np.array(data["latencies"]) * 1000
        corner_err = data["corner_error"] if data["corner_error"] != float("inf") else 999.9
        print(
            f"{name:<12} | {np.median(lat_ms):<12.3f} | {data['id_accuracy'] * 100:<11.1f}% | {corner_err:<15.2f}"
        )


if __name__ == "__main__":
    main()
