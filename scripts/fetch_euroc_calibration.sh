#!/usr/bin/env bash
# Download the EuRoC MAV cam_april calibration dataset.
#
# Usage:
#   bash scripts/fetch_euroc_calibration.sh [target_dir]
#
# The default target is tests/data/euroc. The script is idempotent:
# re-running it with an already-populated directory is a no-op.
#
# NOTE: ETH decommissioned the original robotics.ethz.ch static file host
# (confirmed dead as of 2026-09 — connection times out, not a 404). The
# dataset now lives in ETH's Research Collection (DSpace) under DOI
# 10.3929/ethz-b-000690084, bundled as a single ~4.2 GB zip covering all
# three calibration sequences (imu_april, cam_april, cam_checkerboard). We
# only need `calibration_datasets/cam_april/cam_april.zip` from inside it,
# so we download the bundle to a temp file, extract just that one nested
# zip member (cheap — unzip doesn't decompress sibling entries), extract
# that in turn into place, then discard both temp zips.
#
# Unlike the old host's cam_april.zip, this one's internal root is `mav0/`
# directly (no `cam_april/` prefix inside), so we extract into an explicit
# `$DEST/cam_april/` subdirectory to match what `EurocProvider` expects.
set -euo pipefail

DEST="${1:-tests/data/euroc}"
BUNDLE_URL="https://www.research-collection.ethz.ch/server/api/core/bitstreams/5732e864-10f1-49e7-befb-669ee29ff770/content"
# DSpace's edge occasionally 429s the default curl UA under light rate
# limiting; a browser UA avoids it.
BROWSER_UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"

mkdir -p "$DEST"

if [ -d "$DEST/cam_april/mav0" ]; then
    echo "EuRoC cam_april already present at $DEST/cam_april/ — skipping download."
    exit 0
fi

TMP_BUNDLE="$(mktemp -t euroc_calibration_datasets.XXXXXX.zip)"
trap 'rm -f "$TMP_BUNDLE" "$DEST/cam_april.zip"' EXIT

echo "Downloading EuRoC calibration bundle (~4.2 GB; only cam_april is kept)..."
curl -fSL -A "$BROWSER_UA" "$BUNDLE_URL" -o "$TMP_BUNDLE"

echo "Extracting cam_april.zip from the bundle..."
unzip -jo "$TMP_BUNDLE" "calibration_datasets/cam_april/cam_april.zip" -d "$DEST/"

echo "Extracting cam_april.zip contents..."
mkdir -p "$DEST/cam_april"
unzip -qo "$DEST/cam_april.zip" -d "$DEST/cam_april/"
rm -rf "$DEST/cam_april/__MACOSX"

echo "Done — EuRoC cam_april extracted to $DEST/cam_april/"
