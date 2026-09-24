#!/usr/bin/env bash
# Download the EuRoC MAV cam_april calibration dataset.
#
# Usage:
#   bash scripts/fetch_euroc_calibration.sh [target_dir]
#
# The default target is tests/data/euroc. The script is idempotent:
# re-running it with an already-populated directory is a no-op.
set -euo pipefail

DEST="${1:-tests/data/euroc}"
CAM_APRIL_URL="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/calibration_datasets/cam_april.zip"

mkdir -p "$DEST"

if [ -d "$DEST/cam_april/mav0" ]; then
    echo "EuRoC cam_april already present at $DEST/cam_april/ — skipping download."
    exit 0
fi

echo "Downloading EuRoC cam_april calibration dataset (~200 MB)..."
curl -fSL "$CAM_APRIL_URL" -o "$DEST/cam_april.zip"
echo "Extracting..."
unzip -qo "$DEST/cam_april.zip" -d "$DEST/"
rm -f "$DEST/cam_april.zip"
echo "Done — EuRoC cam_april extracted to $DEST/cam_april/"
