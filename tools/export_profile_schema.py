"""Emit the profile JSON Schema to ``schemas/profile.schema.json``.

This script is the canonical generator for the Rust↔Python referee
document. CI also re-runs it and diffs against the checked-in file; any
deviation fails the build so the schema and the Pydantic model cannot
silently drift apart.

Run with ``uv run python tools/export_profile_schema.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from locus._config import DetectorConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "profile.schema.json"


def build_schema() -> dict[str, object]:
    # ``mode="serialization"`` describes the form profiles take on disk:
    # enums are emitted as their Rust variant names (strings).
    schema = DetectorConfig.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def main(argv: list[str]) -> int:
    schema = build_schema()
    rendered = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    if "--check" in argv:
        if not SCHEMA_PATH.exists():
            print(f"missing: {SCHEMA_PATH}", file=sys.stderr)
            return 1
        current = SCHEMA_PATH.read_text()
        if current != rendered:
            print("schema drift detected; re-run tools/export_profile_schema.py", file=sys.stderr)
            return 1
        print("schema parity: ok")
        return 0
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(rendered)
    print(f"wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
