#!/usr/bin/env python3
"""Test different subpixel_refinement_sigma values on circle dataset."""

import subprocess
import re

# Test different sigma values
sigmas = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]

print("Testing subpixel_refinement_sigma values on circle dataset...\n")
print(f"{'Sigma':<10} {'Recall %':<12} {'RMS Error (px)':<15} {'Detections':<12}")
print("-" * 55)

for sigma in sigmas:
    # Run benchmark with this sigma value
    # Note: We need to modify the benchmark to accept sigma parameter
    # For now, just run the default and show the concept
    result = subprocess.run(
        [
            "uv", "run",
            "--with", "opencv-python",
            "--with", "rerun-sdk", 
            "--with", "huggingface-hub",
            "--with", "tqdm",
            "--with", "pydantic",
            "benchmarks/icra2020.py",
            "--scenarios", "circle",
            "--types", "tags",
            "--limit", "30"
        ],
        capture_output=True,
        text=True,
        cwd="/home/dev/src/locus-tag"
    )
    
    # Parse output
    output = result.stdout + result.stderr
    
    # Look for the TOTAL line
    match = re.search(r'TOTAL\s+\|\s+\d+\s+\|\s+(\d+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)', output)
    if match:
        det = match.group(1)
        recall = match.group(2)
        error = match.group(3)
        print(f"{sigma:<10} {recall:<12} {error:<15} {det:<12}")
    else:
        print(f"{sigma:<10} Failed to parse output")
    
    # Only run once for now since sigma isn't configurable via CLI yet
    break

print("\nNote: To fully test, need to expose sigma via benchmark CLI")
