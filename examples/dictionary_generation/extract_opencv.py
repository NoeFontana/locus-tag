#!/usr/bin/env python3
"""
Locus: OpenCV ArUco Dictionary Extractor
----------------------------------------
Extracts standard OpenCV ArUco dictionaries into the Locus IR format.
Ensures consistent spatial mapping and bit-order by sampling generated images.

Usage:
    uv run examples/dictionary_generation/extract_opencv.py --all
    uv run examples/dictionary_generation/extract_opencv.py --dict DICT_4X4_50
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_VERSION = "2.1.0"

# Standard OpenCV families to extract by default
# tuple: (opencv_name, grid_size, payload_length, min_hamming)
# Note: min_hamming is often reported as an estimate in ArUco
# AprilTag values are based on the family definitions (e.g. 36h11 -> payload 36, dist 11)
STANDARD_FAMILIES = [
    ("DICT_4X4_50", 4, 16, 4),
    ("DICT_4X4_100", 4, 16, 3),
    ("DICT_5X5_50", 5, 25, 4),
    ("DICT_5X5_100", 5, 25, 4),
    ("DICT_6X6_50", 6, 36, 4),
    ("DICT_6X6_100", 6, 36, 4),
    ("DICT_7X7_50", 7, 49, 4),
    ("DICT_7X7_100", 7, 49, 4),
    ("DICT_APRILTAG_16h5", 4, 16, 5),
    ("DICT_APRILTAG_36h11", 6, 36, 11),
]


class OpenCVExtractor:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_canonical_points(self, grid_size: int) -> list[list[float]]:
        """
        Computes sampling centers in a [-1.0, 1.0] continuous space.
        Uses a dense row-major grid corresponding to OpenCV's internal layout.
        Assumes the canonical square [-1, 1] covers the FULL tag (including 1-bit border).
        """
        full_dim = grid_size + 2
        points = []
        for y in range(grid_size):
            for x in range(grid_size):
                # Data bits are in indices [1, grid_size].
                # Map [0, full_dim-1] to centers in [-1.0, 1.0]
                # Center of cell g is (g + 0.5) * 2 / full_dim - 1
                cx = (float(x) + 1.5) * 2.0 / full_dim - 1.0
                cy = (float(y) + 1.5) * 2.0 / full_dim - 1.0
                # Precision limited to 4 decimals for clean IR
                points.append([round(float(cx), 4), round(float(cy), 4)])
        return points

    def get_aruco_dict(self, dict_id: int) -> Any:
        """
        Safely fetch dictionary object.
        """
        try:
            # Modern OpenCV 4.x
            return cv2.aruco.getPredefinedDictionary(dict_id)
        except AttributeError:
            # Older OpenCV 3.x
            try:
                return cv2.aruco.Dictionary_get(dict_id)  # type: ignore
            except AttributeError:
                return None

    def extract(
        self,
        name: str,
        grid_size: int,
        payload_length: int,
        min_hamming: int,
    ) -> Path | None:
        """
        Extracts a single dictionary and writes it to the output directory.
        """
        if not hasattr(cv2.aruco, name):
            logger.warning(f"Dictionary '{name}' not found in cv2.aruco. Skipping.")
            return None

        dict_id = getattr(cv2.aruco, name)
        aruco_dict = self.get_aruco_dict(dict_id)

        if not aruco_dict:
            logger.error(f"Failed to fetch dictionary object for '{name}'.")
            return None

        logger.info(
            f"Extracting {name} ({len(aruco_dict.bytesList)} codes, {grid_size}x{grid_size})..."
        )

        base_codes = []
        for i in range(len(aruco_dict.bytesList)):
            # Generate a large enough image to see bits clearly without aliasing
            test_size = (grid_size + 2) * 10
            tag_img = cv2.aruco.generateImageMarker(aruco_dict, i, test_size)

            bits = 0
            # We sample the grid_size x grid_size data bits.
            cell_size = test_size // (grid_size + 2)

            for row in range(grid_size):
                for col in range(grid_size):
                    # Data grid starts at index 1,1
                    cy = (row + 1) * cell_size + cell_size // 2
                    cx = (col + 1) * cell_size + cell_size // 2

                    if tag_img[cy, cx] > 127:
                        bit_idx = row * grid_size + col
                        bits |= 1 << bit_idx

            base_codes.append(f"{bits:08X}")

        dictionary_ir = {
            "payload_length": payload_length,
            "minimum_hamming_distance": min_hamming,
            "dictionary_size": len(base_codes),
            "canonical_sampling_points": self.compute_canonical_points(grid_size),
            "base_codes": base_codes,
            "_provenance": {
                "source_uri": f"cv2.aruco.{name}",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "script_version": SCRIPT_VERSION,
            },
        }

        out_path = self.output_dir / f"{name.lower()}.json"
        with open(out_path, "w") as f:
            json.dump(dictionary_ir, f, indent=2)

        return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract OpenCV dictionaries to Locus IR")
    parser.add_argument("--output", type=str, default="data/dictionaries", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Extract all standard families")
    parser.add_argument("--dict", type=str, help="Specific dictionary name (e.g. DICT_4X4_50)")

    args = parser.parse_args()
    output_dir = Path(args.output)
    extractor = OpenCVExtractor(output_dir)

    if args.all:
        for name, grid_size, payload, hamming in STANDARD_FAMILIES:
            extractor.extract(name, grid_size, payload, hamming)
    elif args.dict:
        # Find metadata
        entry = next((e for e in STANDARD_FAMILIES if e[0] == args.dict), None)
        if entry:
            extractor.extract(*entry)
        else:
            logger.error(f"Metadata for '{args.dict}' not found in standard registry.")
            logger.info("Please use --all or check supported families.")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
