#!/usr/bin/env python3
"""Extract the CHANGELOG section for a release tag.

Used by `.github/workflows/release.yml` (`create-github-release` job) to
feed `gh release create --notes-file`. Failing here forces the operator
to land a CHANGELOG entry before re-tagging — better than shipping a
Release page with empty body.

Usage:
    python3 scripts/ci/extract_changelog_section.py <version> <path>

`<version>` is the bare semver from the tag (no leading `v`), optionally
with an `-rc.N` suffix. RC versions extract their parent's section
(e.g., `0.5.0-rc.1` → `## [0.5.0]`); if the parent doesn't exist yet,
fall back to `## Unreleased` with a one-line preamble so the dry-run
still produces a meaningful Release page.

Recognised heading forms (locus-tag CHANGELOG conventions):
    ## Unreleased
    ## [Unreleased]
    ## [X.Y.Z]
    ## [X.Y.Z] - YYYY-MM-DD
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Matches any `## …` line and captures whichever identifier the line
# carries. Supports both bracketed (`## [0.5.0]`) and bare
# (`## Unreleased`) forms; trailing text on the same line (date suffix
# etc.) is ignored.
HEADING_RE = re.compile(
    r"^##\s+(?:\[(?P<bracketed>[^\]]+)\]|(?P<bare>\S+))",
    re.MULTILINE,
)


def _heading_id(match: re.Match[str]) -> str:
    return match.group("bracketed") or match.group("bare")


def extract(version: str, changelog: str) -> str:
    """Return the body of the `## <version>` block, exclusive of the
    heading line, stopping at the next `## …` heading or EOF.

    Falls back to `## Unreleased` for RC versions whose parent block
    has not been cut yet.
    """
    base = version.split("-rc.", 1)[0]
    is_rc = base != version

    candidates = [base]
    if is_rc:
        candidates.append("Unreleased")

    for target in candidates:
        for match in HEADING_RE.finditer(changelog):
            if _heading_id(match) != target:
                continue
            # Advance to the start of the line *after* the heading so
            # the body never includes the heading's date suffix.
            after_heading = changelog.find("\n", match.end())
            start = len(changelog) if after_heading == -1 else after_heading + 1
            next_heading = HEADING_RE.search(changelog, start)
            end = next_heading.start() if next_heading else len(changelog)
            body = changelog[start:end].strip()
            if not body:
                raise SystemExit(
                    f"error: CHANGELOG section ## {target} is empty",
                )
            preamble = (
                f"_Release candidate {version} — notes from `## {target}`._\n\n" if is_rc else ""
            )
            return preamble + body + "\n"

    raise SystemExit(
        f"error: no CHANGELOG section matches version {version!r}"
        f" (tried {', '.join(repr(c) for c in candidates)})",
    )


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            f"usage: {argv[0]} <version> <changelog-path>",
            file=sys.stderr,
        )
        return 2
    version = argv[1].lstrip("v")
    changelog_path = Path(argv[2])
    sys.stdout.write(extract(version, changelog_path.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
