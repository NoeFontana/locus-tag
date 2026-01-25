#!/usr/bin/env python3
"""Extract ArUco dictionary codes from OpenCV for use in Locus.

This script extracts all ArUco marker codes from OpenCV's predefined
dictionaries and outputs them in a format suitable for the Locus generator.

Usage:
    python scripts/extract_aruco.py
"""

import cv2


def extract_aruco_codes(dict_id: int, dict_name: str, marker_size: int) -> list[int]:
    """Extract all codes from an ArUco dictionary.

    Args:
        dict_id: OpenCV dictionary ID (e.g., cv2.aruco.DICT_4X4_50)
        dict_name: Human-readable name
        marker_size: Grid size (e.g., 4 for 4x4)

    Returns:
        List of u64 codes in row-major bit ordering
    """
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    n_markers = dictionary.bytesList.shape[0]

    print(f"{dict_name}: {n_markers} markers, {marker_size}x{marker_size}")

    codes = []
    for marker_id in range(n_markers):
        # Generate a marker image large enough to sample clearly
        img_size = marker_size * 10  # 10 pixels per cell
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, img_size)

        # The total grid includes 1-cell border on each side
        total_size = marker_size + 2
        cell_size = img_size // total_size

        # Extract bits from inner grid (skip border)
        bits = 0
        for row in range(marker_size):
            for col in range(marker_size):
                # Sample center of each cell (skip 1-cell border)
                cy = (row + 1) * cell_size + cell_size // 2
                cx = (col + 1) * cell_size + cell_size // 2
                val = marker_img[cy, cx]

                # ArUco: white = 1, black = 0
                if val > 128:
                    bit_idx = row * marker_size + col
                    bits |= 1 << bit_idx

        codes.append(bits)

    return codes


def format_as_python(name: str, codes: list[int]) -> str:
    """Format codes as Python list for inclusion in generator."""
    lines = [f"# {name} ({len(codes)} markers)"]
    lines.append(f"{name} = [")
    for i in range(0, len(codes), 8):
        chunk = codes[i : i + 8]
        hex_strs = [f"0x{c:04x}" for c in chunk]
        lines.append(f"    {', '.join(hex_strs)},")
    lines.append("]")
    return "\n".join(lines)


def main():
    print("Extracting ArUco dictionaries from OpenCV...")
    print()

    # DICT_4X4 family
    dict_4x4_50 = extract_aruco_codes(cv2.aruco.DICT_4X4_50, "DICT_4X4_50", 4)
    dict_4x4_100 = extract_aruco_codes(cv2.aruco.DICT_4X4_100, "DICT_4X4_100", 4)

    print()
    print("=" * 60)
    print("CODES FOR GENERATOR (copy to generate_dictionaries.py)")
    print("=" * 60)
    print()

    print(format_as_python("ARUCO_4X4_50_CODES", dict_4x4_50))
    print()
    print(format_as_python("ARUCO_4X4_100_CODES", dict_4x4_100))
    print()

    # Verify a few codes manually
    print()
    print("Verification (first 5 codes of each):")
    print(f"  4X4_50:  {[hex(c) for c in dict_4x4_50[:5]]}")
    print(f"  4X4_100: {[hex(c) for c in dict_4x4_100[:5]]}")


if __name__ == "__main__":
    main()
