#!/usr/bin/env python3
"""Diagnostic script to analyze where quads fail in the circle dataset pipeline."""

from pathlib import Path

import cv2
import locus

# Load a few challenging images from circle dataset
CIRCLE_DIR = Path("tests/data/icra2020/circle/pure_tags_images")


def analyze_image(img_path: Path):
    """Analyze a single image to see quad extraction vs decoding."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    detector = locus.Detector(
        quad_min_area=16,
        quad_max_aspect_ratio=20.0,
        quad_min_edge_score=0.5,
        enable_bilateral=False,
        enable_adaptive_window=False,
    )

    detections, stats = detector.detect_with_stats(img)

    return {
        "image": img_path.name,
        "quads_extracted": stats.num_candidates,
        "tags_decoded": stats.num_detections,
        "decode_rate": stats.num_detections / max(1, stats.num_candidates) * 100,
        "threshold_ms": stats.threshold_ms,
        "quad_ms": stats.quad_extraction_ms,
        "decode_ms": stats.decoding_ms,
    }


def main():
    if not CIRCLE_DIR.exists():
        print(f"Circle dataset not found at {CIRCLE_DIR}")
        return

    images = sorted(CIRCLE_DIR.glob("*.png"))[:50]  # First 50 images

    print(f"Analyzing {len(images)} images from circle dataset...\n")
    print(
        f"{'Image':<20} {'Quads':<8} {'Decoded':<8} {'Rate%':<8} {'Thresh(ms)':<10} {'Quad(ms)':<10} {'Decode(ms)':<10}"
    )
    print("-" * 90)

    total_quads = 0
    total_decoded = 0

    for img_path in images:
        result = analyze_image(img_path)
        if result:
            total_quads += result["quads_extracted"]
            total_decoded += result["tags_decoded"]
            print(
                f"{result['image']:<20} {result['quads_extracted']:<8} {result['tags_decoded']:<8} {result['decode_rate']:<8.1f} {result['threshold_ms']:<10.2f} {result['quad_ms']:<10.2f} {result['decode_ms']:<10.2f}"
            )

    print("-" * 90)
    overall_rate = total_decoded / max(1, total_quads) * 100
    print(f"{'TOTAL':<20} {total_quads:<8} {total_decoded:<8} {overall_rate:<8.1f}")

    print(f"\n=== DIAGNOSIS ===")
    print(f"Total quads extracted: {total_quads}")
    print(f"Total tags decoded: {total_decoded}")
    print(f"Conversion rate: {overall_rate:.1f}%")

    if overall_rate < 50:
        print("\n⚠️  LOW CONVERSION RATE: Most quads are failing at decoding stage.")
        print("   Likely causes:")
        print("   - Homography sampling errors (bit sampling misalignment)")
        print("   - Bit thresholding too strict")
        print("   - Code lookup failures (hamming distance too high)")
    else:
        print("\n⚠️  MODERATE CONVERSION: Issue may be in quad extraction.")
        print("   Likely causes:")
        print("   - Geometric constraints too strict")
        print("   - Line grouping failing for extreme tilts")
        print("   - Segmentation missing low-contrast regions")


if __name__ == "__main__":
    main()
