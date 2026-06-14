#!/usr/bin/env python3
"""Extract the CHANGELOG section for a release tag.

Used by `.github/workflows/release.yml` (`create-github-release` job) to
feed `gh release create --notes-file`. Failing here forces the operator
to land a CHANGELOG entry before re-tagging — better than shipping a
Release page with empty body.

Usage:
    python3 scripts/ci/extract_changelog_section.py <version> <path>

`<version>` is the bare semver from the tag (no leading `v`), optionally
with any prerelease suffix after a `-` (`0.5.0-rc.1`, `0.5.0-beta.1`,
…). Prerelease versions extract their parent's section
(e.g., `0.5.0-rc.1` → `## [0.5.0]`); if the parent doesn't exist yet,
fall back to `## Unreleased` with a one-line preamble so the dry-run
still produces a meaningful Release page. The preamble fires ONLY on
the fallback path — a prerelease whose parent block already exists
emits the parent body verbatim.

Recognised heading forms (locus-tag CHANGELOG conventions):
    ## Unreleased
    ## [Unreleased]
    ## [X.Y.Z]
    ## [X.Y.Z] - YYYY-MM-DD

`##`-prefixed lines inside fenced code blocks are NOT treated as
headings — they're stripped before regex search so an example like
`\\`\\`\\`md\\n## [1.0.0]\\n\\`\\`\\`` inside `## Unreleased` doesn't
truncate the body.
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


def _blank_code_fences(text: str) -> str:
    """Return `text` with fenced code-block contents blanked out.

    Preserves line count and character offsets so positions found in
    the returned string map 1:1 to positions in `text`. Both the
    fence markers and the lines between them are replaced by
    equal-length whitespace runs.
    """
    lines = text.split("\n")
    in_fence = False
    out: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        is_fence = stripped.startswith("```") or stripped.startswith("~~~")
        if is_fence:
            in_fence = not in_fence
            out.append(" " * len(line))
            continue
        if in_fence:
            out.append(" " * len(line))
        else:
            out.append(line)
    return "\n".join(out)


def extract(version: str, changelog: str) -> str:
    """Return the body of the `## <version>` block, exclusive of the
    heading line, stopping at the next `## …` heading or EOF.

    Prerelease versions fall back to `## Unreleased` if the parent
    block has not been cut yet; the preamble fires only on the
    fallback path.
    """
    base = version.split("-", 1)[0]
    is_prerelease = base != version

    candidates = [base]
    if is_prerelease:
        candidates.append("Unreleased")

    search_view = _blank_code_fences(changelog)

    for target in candidates:
        for match in HEADING_RE.finditer(search_view):
            if _heading_id(match) != target:
                continue
            # Advance to the start of the line *after* the heading so
            # the body never includes the heading's date suffix.
            after_heading = search_view.find("\n", match.end())
            start = len(changelog) if after_heading == -1 else after_heading + 1
            next_heading = HEADING_RE.search(search_view, start)
            end = next_heading.start() if next_heading else len(changelog)
            body = changelog[start:end].strip()
            if not body:
                raise SystemExit(
                    f"error: CHANGELOG section ## {target} is empty",
                )
            is_fallback = is_prerelease and target != base
            preamble = (
                f"_Release candidate {version} — notes inherited from `## {target}`._\n\n"
                if is_fallback
                else ""
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
    version = argv[1].removeprefix("v")
    changelog_path = Path(argv[2])
    sys.stdout.write(extract(version, changelog_path.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
