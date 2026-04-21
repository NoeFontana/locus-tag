"""Smoke-test a freshly built locus wheel.

Run inside a venv that has the wheel installed and numpy available.
Exits non-zero on any failure so CI can gate publishing on the result.
"""

from __future__ import annotations

import sys

import numpy as np
from locus import Detector, DetectorConfig
from locus._config import SHIPPED_PROFILES


def main() -> int:
    image = np.zeros((480, 640), dtype=np.uint8)
    for profile in SHIPPED_PROFILES:
        Detector(profile=profile).detect(image)
        Detector(config=DetectorConfig.from_profile(profile)).detect(image)
    print(f"wheel smoke ok ({len(SHIPPED_PROFILES)} profiles)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
