#!/usr/bin/env bash
# Download the EuRoC MAV cam_april calibration dataset.
#
# Usage:
#   bash scripts/fetch_euroc_calibration.sh [target_dir]
#
# The default target is tests/data/euroc. The script is idempotent:
# re-running it with an already-populated directory is a no-op.
#
# Primary source: a private Hugging Face Hub mirror
# (NoeFontana/euroc-mav-cam-april-mirror, see its dataset card for license
# and citation details). ETH decommissioned the original robotics.ethz.ch
# static file host (confirmed dead as of 2026-09 — connection times out, not
# a 404). Its replacement, ETH's Research Collection (DSpace) under DOI
# 10.3929/ethz-b-000690084, works, but is frequently blocked by the
# outbound-host allowlists sandboxed CI/agent environments use — the Hugging
# Face Hub is broadly allowlisted in those same environments, so mirroring
# there fixes fetch reliability without changing what data is used.
#
# For provenance (NOT used by default — unreliable from restricted-network
# environments): the Research Collection bundles all three calibration
# sequences (imu_april, cam_april, cam_checkerboard) as a single ~4.2 GB zip;
# only `calibration_datasets/cam_april/cam_april.zip` is needed, and that
# nested zip's own internal root is `mav0/` directly (no `cam_april/` prefix
# inside), hence the explicit `$DEST/cam_april/` subdirectory below.
#   https://www.research-collection.ethz.ch/handle/20.500.11850/690084
set -euo pipefail

DEST="${1:-tests/data/euroc}"
HF_REPO_ID="${LOCUS_EUROC_HF_REPO:-NoeFontana/euroc-mav-cam-april-mirror}"

mkdir -p "$DEST"

if [ -d "$DEST/cam_april/mav0" ]; then
    echo "EuRoC cam_april already present at $DEST/cam_april/ — skipping download."
    exit 0
fi

if ! command -v hf >/dev/null 2>&1; then
    echo "error: the 'hf' CLI (huggingface_hub) is required but not found on PATH." >&2
    echo "Install it with: curl -LsSf https://hf.co/cli/install.sh | bash -s" >&2
    exit 1
fi

DEST_ABS="$(cd "$DEST" && pwd)"

echo "Downloading EuRoC cam_april calibration dataset from HF mirror ($HF_REPO_ID)..."
hf download "$HF_REPO_ID" \
    --repo-type dataset \
    --include "cam_april.zip" \
    --local-dir "$DEST_ABS"

echo "Extracting..."
unzip -qo "$DEST_ABS/cam_april.zip" -d "$DEST_ABS/"
rm -f "$DEST_ABS/cam_april.zip"
rm -rf "$DEST_ABS/.cache"

echo "Done — EuRoC cam_april extracted to $DEST/cam_april/"
