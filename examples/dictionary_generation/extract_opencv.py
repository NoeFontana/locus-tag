#!/usr/bin/env python3
"""
Locus: OpenCV ArUco Dictionary Extractor
----------------------------------------
Extracts standard OpenCV ArUco dictionaries into the Locus IR format.
Ensures consistent spatial mapping and bit-ordering for zero-copy high-performance decoding.

This script fetches pre-defined dictionaries from OpenCV, compute canonical sampling
points in the [-1.0, 1.0] continuous space, and exports a unified JSON representation.

Usage:
    uv run scripts/data/extract_opencv.py --all
    uv run scripts/data/extract_opencv.py --dict DICT_4X4_50
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Configure logging for professional output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("locus.extractor")

try:
    import cv2
except ImportError:
    cv2 = None

SCRIPT_VERSION = "2.0.1"

# Standard OpenCV families to extract by default
# tuple: (opencv_name, grid_size, payload_length, min_hamming)
# Note: min_hamming is often reported as an estimate in ArUco
STANDARD_FAMILIES = [
    ("DICT_4X4_50", 4, 16, 4),
    ("DICT_4X4_100", 4, 16, 3),
]


class OpenCVExtractor:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if cv2 is None:
            logger.error("OpenCV (cv2) not installed. Please install 'opencv-contrib-python'.")
            sys.exit(1)

    def compute_canonical_points(self, grid_size: int) -> list[list[float]]:
        """
        Computes sampling centers in a [-1.0, 1.0] continuous space.
        Uses a dense row-major grid corresponding to OpenCV's internal layout.
        """
        points = []
        for y in range(grid_size):
            for x in range(grid_size):
                # Map [0, grid_size-1] to centers in [-1.0, 1.0]
                cx = -1.0 + (2.0 * x + 1.0) / grid_size
                cy = -1.0 + (2.0 * y + 1.0) / grid_size
                # Precision limited to 4 decimals for clean IR
                points.append([round(float(cx), 4), round(float(cy), 4)])
        return points

    def get_aruco_dict(self, dict_id: int) -> Any:
        try:
            # Modern OpenCV 4.7+
            return cv2.aruco.getPredefinedDictionary(dict_id)
        except AttributeError:
            try:
                # Older OpenCV 4.x
                return cv2.aruco.Dictionary_get(dict_id)
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
        num_bytes = (payload_length + 7) // 8

        # OpenCV bytesList shape: (n_markers, 4_rotations, bytes_per_marker)
        for i in range(len(aruco_dict.bytesList)):
            marker_bytes = aruco_dict.bytesList[i][0]  # Get canonical rotation
            # Construct integer from bytes (Big Endian as used in ArUco)
            # Masking isn't strictly necessary as ArUco handles padding,
            # but we extract exactly what we need for the IR.
            val = int.from_bytes(marker_bytes[:num_bytes], byteorder="big")

            # Convert to hex string, matching the IR convention (Uppercase, no 0x)
            hex_str = hex(val)[2:].upper()
            base_codes.append(hex_str)

        dictionary_ir = {
            "payload_length": payload_length,
            "minimum_hamming_distance": min_hamming,
            "dictionary_size": len(base_codes),
            "canonical_sampling_points": self.compute_canonical_points(grid_size),
            "base_codes": base_codes,
            "_provenance": {
                "source_uri": f"cv2.aruco.{name}",
                "cv2_version": cv2.__version__ if cv2 else "Unknown",
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "script_version": SCRIPT_VERSION,
            },
        }

        out_path = self.output_dir / f"{name.lower()}.json"
        with open(out_path, "w") as f:
            json.dump(dictionary_ir, f, indent=2)

        logger.info(f"Successfully wrote {out_path}")
        return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract OpenCV ArUco dictionaries.")
    parser.add_argument("--all", action="store_true", help="Extract all standard ArUco families.")
    parser.add_argument("--dict", type=str, help="Specific dictionary name (e.g., DICT_4X4_50).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[2] / "data" / "dictionaries",
        help="Output directory for JSON files.",
    )

    args = parser.parse_args()
    extractor = OpenCVExtractor(args.output)

    if args.all:
        for name, grid, payload, hd in STANDARD_FAMILIES:
            extractor.extract(name, grid, payload, hd)
    elif args.dict:
        # Search for metadata in standard families if possible
        metadata = next((f for f in STANDARD_FAMILIES if f[0] == args.dict), None)
        if metadata:
            extractor.extract(*metadata)
        else:
            logger.error(f"Metadata for '{args.dict}' not found in standard registry.")
            logger.info("Please use --all or check supported families.")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
