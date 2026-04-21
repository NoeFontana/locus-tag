#!/usr/bin/env bash
# Install a freshly built locus wheel into a throwaway venv and run smoke_wheel.py.
# Usage: install_and_smoke_wheel.sh [<wheel-glob>]
#   wheel-glob defaults to dist/locus_tag-*.whl
set -euo pipefail

WHEEL_GLOB="${1:-dist/locus_tag-*.whl}"
# shellcheck disable=SC2086  # glob expansion is intentional
WHEEL=$(ls $WHEEL_GLOB 2>/dev/null | head -n 1)
test -n "$WHEEL" || { echo "no wheel matched: $WHEEL_GLOB" >&2; exit 1; }

VENV=$(mktemp -d)/venv
python -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet numpy "$WHEEL"

python "$(dirname "$0")/smoke_wheel.py"
