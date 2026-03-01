#!/usr/bin/env python3
"""
Extracts UMich AprilTag dictionaries directly from the official C source code.
Translates spiral bit ordering to the canonical continuous space [-1.0, 1.0].
"""

import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

try:
    from apriltag_41h12 import APRILTAG_41H12_CODES_SPIRAL, UMICH_41H12_BIT_ORDER
except ImportError:
    APRILTAG_41H12_CODES_SPIRAL = []
    UMICH_41H12_BIT_ORDER = []

SCRIPT_VERSION = "1.0.1"
UPSTREAM_REPO = "https://raw.githubusercontent.com/AprilRobotics/apriltag"
COMMIT_HASH = "master"

FAMILIES = {
    "tag36h11": {
        "url": f"{UPSTREAM_REPO}/{COMMIT_HASH}/tag36h11.c",
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
    "tag16h5": {
        "url": f"{UPSTREAM_REPO}/{COMMIT_HASH}/tag16h5.c",
        "payload_length": 16,
        "minimum_hamming_distance": 5,
        "dictionary_size": 30,
        "spiral_order": [
            (1, 1),
            (2, 1),
            (3, 1),
            (2, 2),
            (4, 1),
            (4, 2),
            (4, 3),
            (3, 2),
            (4, 4),
            (3, 4),
            (2, 4),
            (3, 3),
            (1, 4),
            (1, 3),
            (1, 2),
            (2, 3),
        ],
    },
    "tag41h12": {
        "url": f"{UPSTREAM_REPO}/{COMMIT_HASH}/tagStandard41h12.c",
        "payload_length": 41,
        "minimum_hamming_distance": 12,
        "dictionary_size": 2115,
        "spiral_order": UMICH_41H12_BIT_ORDER,
        "local_fallback_codes": [hex(c)[2:].upper().zfill(16) for c in APRILTAG_41H12_CODES_SPIRAL],
    },
}


def umich_coords_to_canonical(coords):
    canonical_points = []
    min_x = min(x for x, y in coords)
    max_x = max(x for x, y in coords)
    grid_size = max_x - min_x + 1

    for x, y in coords:
        cx = -1.0 + (2.0 * (x - min_x) + 1.0) / grid_size
        cy = -1.0 + (2.0 * (y - min_x) + 1.0) / grid_size  # assuming square
        canonical_points.append([float(f"{cx:.4f}"), float(f"{cy:.4f}")])
    return canonical_points


def extract_umich(family_name, config):
    url = config["url"]
    print(f"Fetching {family_name} from {url}...")
    base_codes = []
    try:
        req = urllib.request.urlopen(url)
        c_code = req.read().decode("utf-8")
        matches = re.findall(r"0x[0-9a-fA-F]+", c_code)
        for m in matches:
            base_codes.append(m[2:].upper().zfill(16))
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        if "local_fallback_codes" in config:
            print("Using local fallback codes.")
            base_codes = config["local_fallback_codes"]

    canonical_sampling_points = umich_coords_to_canonical(config["spiral_order"])

    base_codes = [c.lstrip("0") or "0" for c in base_codes]
    if len(base_codes) > config["dictionary_size"]:
        base_codes = base_codes[: config["dictionary_size"]]

    dictionary_ir = {
        "payload_length": config["payload_length"],
        "minimum_hamming_distance": config["minimum_hamming_distance"],
        "dictionary_size": len(base_codes),
        "canonical_sampling_points": canonical_sampling_points,
        "base_codes": base_codes,
        "_provenance": {
            "source_uri": url,
            "commit_hash": COMMIT_HASH,
            "cv2_version": "N/A",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "script_version": SCRIPT_VERSION,
        },
    }

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "dictionaries")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{family_name}.json")
    with open(out_path, "w") as f:
        json.dump(dictionary_ir, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    for family, config in FAMILIES.items():
        extract_umich(family, config)
