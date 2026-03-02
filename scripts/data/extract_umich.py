#!/usr/bin/env python3
"""
Locus: AprilTag (UMich) Dictionary Extractor
------------------------------------------
Extracts official AprilTag (UMich) dictionaries directly from the C source code.
Translates spiral bit ordering to the canonical Locus continuous space [-1.0, 1.0].

Usage:
    uv run scripts/data/extract_umich.py --all
    uv run scripts/data/extract_umich.py --family tag36h11
"""

import argparse
import json
import logging
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("locus.umich_extractor")

# Configuration for upstream repository
UPSTREAM_REPO = "https://raw.githubusercontent.com/AprilRobotics/apriltag"
DEFAULT_COMMIT = "master"
SCRIPT_VERSION = "2.0.1"

# Import local fallbacks for high-latency/offline environments
SCRIPT_DIR = Path(__file__).parent
try:
    sys.path.insert(0, str(SCRIPT_DIR))
    from apriltag_41h12 import APRILTAG_41H12_CODES_SPIRAL, UMICH_41H12_BIT_ORDER
except ImportError:
    logger.warning("Local fallback for tag41h12 not found. Online fetching required.")
    APRILTAG_41H12_CODES_SPIRAL = []
    UMICH_41H12_BIT_ORDER = []

# Registry of supported AprilTag families
FAMILIES: dict[str, Any] = {
    "tag36h11": {
        "url_suffix": "tag36h11.c",
        "payload_length": 36,
        "minimum_hamming_distance": 11,
        "dictionary_size": 587,
        "spiral_order": [
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (2, 2),
            (3, 2),
            (4, 2),
            (3, 3),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (5, 2),
            (5, 3),
            (5, 4),
            (4, 3),
            (6, 6),
            (5, 6),
            (4, 6),
            (3, 6),
            (2, 6),
            (5, 5),
            (4, 5),
            (3, 5),
            (4, 4),
            (1, 6),
            (1, 5),
            (1, 4),
            (1, 3),
            (1, 2),
            (2, 5),
            (2, 4),
            (2, 3),
            (3, 4),
        ],
    },
    "tagStandard41h12": {
        "url_suffix": "tagStandard41h12.c",
        "payload_length": 41,
        "minimum_hamming_distance": 12,
        "dictionary_size": 2115,
        "spiral_order": UMICH_41H12_BIT_ORDER,
        "local_fallback": [hex(c)[2:].upper() for c in APRILTAG_41H12_CODES_SPIRAL],
    },
}


class AprilTagExtractor:
    def __init__(self, output_dir: Path, commit: str = DEFAULT_COMMIT):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.commit = commit

    def umich_coords_to_canonical(self, coords: list[tuple[int, int]]) -> list[list[float]]:
        """
        Maps UMich discrete spiral coordinates to Locus canonical space [-1.0, 1.0].
        """
        if not coords:
            return []

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # AprilTags are square, so we use max dimension for grid scale
        grid_size = max(max_x - min_x, max_y - min_y) + 1

        canonical_points = []
        for x, y in coords:
            # Map discrete coordinates to centers in [-1.0, 1.0]
            cx = -1.0 + (2.0 * (x - min_x) + 1.0) / grid_size
            cy = -1.0 + (2.0 * (y - min_y) + 1.0) / grid_size
            canonical_points.append([round(float(cx), 4), round(float(cy), 4)])

        return canonical_points

    def fetch_codes(self, url: str) -> list[str]:
        """Fetches C source and extracts hex codes using regex."""
        logger.info(f"Fetching source from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                content = response.read().decode("utf-8")
                # Match hex patterns like 0x123456789ABCDEFLL or 0x1234
                matches = re.findall(r"0x([0-9a-fA-F]+)", content)
                # Clean and normalize
                return [m.upper().rstrip("L") for m in matches]
        except Exception as e:
            logger.error(f"Network error: {e}")
            return []

    def extract(self, family_name: str) -> Path | None:
        if family_name not in FAMILIES:
            logger.error(f"Unknown family: {family_name}")
            return None

        cfg = FAMILIES[family_name]
        url = f"{UPSTREAM_REPO}/{self.commit}/{cfg['url_suffix']}"

        raw_codes = self.fetch_codes(url)

        if not raw_codes:
            if "local_fallback" in cfg and cfg["local_fallback"]:
                logger.warning(f"Using local fallback for {family_name}.")
                raw_codes = cfg["local_fallback"]
            else:
                logger.error(f"No codes available for {family_name}. Skipping.")
                return None

        # Standard UMich files often contain metadata or bit-orders at the start
        # We slice based on the expected dictionary size from the registry
        if len(raw_codes) > cfg["dictionary_size"]:
            # Logic: UMich files usually have codes at the end or in a specific array.
            # For tag36h11, they are the first few hundred.
            # We filter out very small values that might be bit-orders.
            codes = [c for c in raw_codes if len(c) > 4]
            codes = codes[: cfg["dictionary_size"]]
        else:
            codes = raw_codes

        # Clean codes: remove leading zeros to match IR convention, but keep '0' if value is 0
        final_codes = [c.lstrip("0") or "0" for c in codes]

        if not cfg["spiral_order"]:
            logger.error(
                f"Spiral order metadata missing for {family_name}. Cannot map to canonical space."
            )
            return None

        dictionary_ir = {
            "payload_length": cfg["payload_length"],
            "minimum_hamming_distance": cfg["minimum_hamming_distance"],
            "dictionary_size": len(final_codes),
            "canonical_sampling_points": self.umich_coords_to_canonical(cfg["spiral_order"]),
            "base_codes": final_codes,
            "_provenance": {
                "source_uri": url,
                "commit_hash": self.commit,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "script_version": SCRIPT_VERSION,
            },
        }

        out_path = self.output_dir / f"{family_name.lower()}.json"
        with open(out_path, "w") as f:
            json.dump(dictionary_ir, f, indent=2)

        logger.info(f"Successfully wrote {out_path} ({len(final_codes)} codes)")
        return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract UMich AprilTag dictionaries.")
    parser.add_argument("--all", action="store_true", help="Extract all standard families.")
    parser.add_argument("--family", type=str, help="Specific family (e.g., tag36h11).")
    parser.add_argument(
        "--commit", type=str, default=DEFAULT_COMMIT, help="Target git commit/branch."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[2] / "data" / "dictionaries",
        help="Output directory.",
    )

    args = parser.parse_args()
    extractor = AprilTagExtractor(args.output, commit=args.commit)

    if args.all:
        for family in FAMILIES:
            extractor.extract(family)
    elif args.family:
        extractor.extract(args.family)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
