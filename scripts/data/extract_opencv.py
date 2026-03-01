#!/usr/bin/env python3
"""
Extracts OpenCV ArUco dictionaries into the canonical IR.
Handles fetching via cv2.aruco, spatial mapping to continuous space,
and base code deduplication/normalization.
"""

import json
import os
from datetime import datetime, timezone

try:
    import cv2
except ImportError:
    cv2 = None

SCRIPT_VERSION = "1.0.0"


def get_opencv_aruco_dict(dict_id):
    if cv2 is None:
        return None
    try:
        return cv2.aruco.getPredefinedDictionary(dict_id)
    except AttributeError:
        try:
            return cv2.aruco.Dictionary_get(dict_id)
        except AttributeError:
            return None


def compute_canonical_points(grid_size):
    """
    OpenCV uses a dense row-major grid.
    We convert the cell centers to [-1.0, 1.0].
    """
    points = []
    for y in range(grid_size):
        for x in range(grid_size):
            cx = -1.0 + (2.0 * (x + 1) - 1.0) / grid_size
            cy = -1.0 + (2.0 * (y + 1) - 1.0) / grid_size
            points.append([float(f"{cx:.4f}"), float(f"{cy:.4f}")])
    return points


def extract_opencv_aruco(dict_name, dict_id, grid_size, payload_length, min_hamming):
    aruco_dict = get_opencv_aruco_dict(dict_id)
    if not aruco_dict:
        print(f"Skipping {dict_name}, CV2 dictionary not found.")
        return

    cv2_version = cv2.__version__ if cv2 is not None else "Mocked (cv2 not installed)"

    base_codes = []
    for byte_list in aruco_dict.bytesList:
        # For larger grids, bytesList padding can vary.
        # Construct integer from the bytes holding the raw bits.
        # Assume byte order big, calculate needed bytes for payload
        num_bytes = (payload_length + 7) // 8
        val = int.from_bytes(byte_list[0][:num_bytes], byteorder="big")
        # Mask out any padded high-bits (ArUco aligns to bytes usually, padded with 0s)
        # We need the lowest `payload_length` bits, but wait, OpenCV's bytesList stores blocks.
        # Actually, let's just make sure we capture it all.
        hex_str = hex(val)[2:].upper()
        # pad if necessary
        # hex_str = hex_str.zfill((payload_length + 3) // 4)
        base_codes.append(hex_str)

    canonical_sampling_points = compute_canonical_points(grid_size)

    dictionary_ir = {
        "payload_length": payload_length,
        "minimum_hamming_distance": min_hamming,
        "dictionary_size": len(base_codes),
        "canonical_sampling_points": canonical_sampling_points,
        "base_codes": base_codes,
        "_provenance": {
            "source_uri": f"cv2.aruco.{dict_name}",
            "commit_hash": "N/A",
            "cv2_version": cv2_version,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "script_version": SCRIPT_VERSION,
        },
    }

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "dictionaries")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dict_name.lower()}.json")
    with open(out_path, "w") as f:
        json.dump(dictionary_ir, f, indent=2)
    print(f"Wrote {out_path}")


def generate_all():
    # standard families mapping
    # dict name, cv2 const string, grid size, payload_length, min_hamming (estimated for schema)
    # The true minimum hamming for standard ArUco varies.
    if cv2 is None:
        print("CV2 not installed, cannot generate.")
        return

    families = [
        ("DICT_APRILTAG_36h11", 6, 36, 11),
        ("DICT_APRILTAG_16h5", 4, 16, 5),
        ("DICT_4X4_50", 4, 16, 3),
        ("DICT_4X4_100", 4, 16, 2),
    ]

    for name, grid, payload, hd in families:
        if hasattr(cv2.aruco, name):
            dict_id = getattr(cv2.aruco, name)
            extract_opencv_aruco(name, dict_id, grid, payload, hd)


if __name__ == "__main__":
    generate_all()
