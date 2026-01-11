import argparse
import time
from pathlib import Path

import cv2
import locus
import numpy as np
from pupil_apriltags import Detector as AprilTagDetector


def generate_synthetic_data(num_images=5, size=(640, 480)):
    """Generate synthetic images with tags for benchmarking."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    images = []
    for i in range(num_images):
        # Generate a standard AprilTag (ID 0, 1, or 2)
        tag_id = i % 3
        tag_size = 120
        tag_img = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_size)

        # Place it in a larger image
        img = np.zeros(size, dtype=np.uint8) + 128
        x, y = 100 + i * 40, 100 + i * 20

        # Add a white "quiet zone" around the tag to help detectors
        padding = 20
        img[y - padding : y + tag_size + padding, x - padding : x + tag_size + padding] = 255
        img[y : y + tag_size, x : x + tag_size] = tag_img

        images.append(img)
    return images


def benchmark_locus(images, iterations):
    latencies = []
    detections_count = 0
    # Warm up
    _ = locus.detect_tags(images[0])

    for img in images:
        for _ in range(iterations):
            start = time.perf_counter()
            detections = locus.detect_tags(img)
            latencies.append(time.perf_counter() - start)
            detections_count += len(detections)
    return latencies, detections_count


def benchmark_locus_gradient(images, iterations):
    latencies = []
    detections_count = 0
    # Warm up
    _ = locus.detect_tags_gradient(images[0])

    for img in images:
        for _ in range(iterations):
            start = time.perf_counter()
            detections = locus.detect_tags_gradient(img)
            latencies.append(time.perf_counter() - start)
            detections_count += len(detections)
    return latencies, detections_count


def benchmark_opencv(images, iterations):
    # ArUco AprilTag 36h11 dictionary for comparison
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Warm up
    _, _, _ = detector.detectMarkers(images[0])

    latencies = []
    detections_count = 0
    for img in images:
        for _ in range(iterations):
            start = time.perf_counter()
            corners, ids, _ = detector.detectMarkers(img)
            latencies.append(time.perf_counter() - start)
            if ids is not None:
                detections_count += len(ids)
    return latencies, detections_count


def benchmark_apriltag(images, iterations):
    at_detector = AprilTagDetector(families="tag36h11", nthreads=1)

    # Warm up
    _ = at_detector.detect(images[0])

    latencies = []
    detections_count = 0
    for img in images:
        for _ in range(iterations):
            start = time.perf_counter()
            detections = at_detector.detect(img)
            latencies.append(time.perf_counter() - start)
            detections_count += len(detections)
    return latencies, detections_count


def main():
    parser = argparse.ArgumentParser(description="Locus SOTA Benchmark")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations per image")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    args = parser.parse_args()

    # Data loading
    if args.synthetic:
        print("Generating synthetic data...")
        images = generate_synthetic_data()
    else:
        # Try to load from tests/data/umich if it exists
        data_dir = Path("tests/data/umich")
        # Check if dir exists and has images
        image_paths = []
        if data_dir.exists():
            image_paths = list(data_dir.rglob("*.png")) + list(data_dir.rglob("*.jpg"))

        if not image_paths:
            print("No real data found in tests/data/umich, falling back to synthetic.")
            images = generate_synthetic_data()
        else:
            print(f"Loading {len(image_paths)} images from {data_dir}...")
            # Use a subset if many images
            if len(image_paths) > 20:
                print("Limiting to 20 images for benchmarking.")
                image_paths = image_paths[:20]
            images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in image_paths]

    print(f"Benchmarking with {len(images)} images, {args.iterations} iterations each.")

    results = {}
    metrics = {}

    libraries = [
        ("locus", benchmark_locus),
        ("locus_grad", benchmark_locus_gradient),
        ("opencv", benchmark_opencv),
        ("apriltag", benchmark_apriltag),
    ]

    for name, func in libraries:
        print(f"Running {name}...")
        lat, count = func(images, args.iterations)
        results[name] = lat
        metrics[name] = count

    # Print summary
    print("\nBenchmark Results:")
    print(
        f"{'Library':<15} | {'Mean (ms)':<10} | {'Median (ms)':<11} | {'99th (ms)':<10} | {'Detections/Img':<15}"
    )
    print("-" * 80)
    for name in results.keys():
        latencies = results[name]
        lat_ms = np.array(latencies) * 1000
        detections = metrics[name] / (len(images) * args.iterations)
        print(
            f"{name:<15} | {np.mean(lat_ms):<10.3f} | {np.median(lat_ms):<11.3f} | {np.percentile(lat_ms, 99):<10.3f} | {detections:<15.2f}"
        )


if __name__ == "__main__":
    main()
