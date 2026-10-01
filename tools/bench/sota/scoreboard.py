"""Cross-benchmark win table: one Locus run against the best reference, everywhere it was scored.

Usage::

    python -m tools.bench.sota.scoreboard <runs_root> [champion]

Reads every ``<runs_root>/<benchmark>/score.json`` (+ ``meta.json``) whose directory name is
a ``[sota.*]`` benchmark and writes ``<runs_root>/../scoreboard.md``. Latency cells are only
judged for runs whose timing is valid (detectors run serially).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.bench.sota.report import (
    DEFAULT_CHAMPION,
    REFERENCES,
    Verdict,
    fmt,
    verdict_cell,
    win_table,
)
from tools.bench.sota.spec import load_specs


def collect(runs_root: Path, champion: str) -> dict[str, list[Verdict]]:
    specs = load_specs()
    out: dict[str, list[Verdict]] = {}
    for name, spec in specs.items():
        score_path = runs_root / name / "score.json"
        meta_path = runs_root / name / "meta.json"
        if not (score_path.exists() and meta_path.exists()):
            continue
        score = json.loads(score_path.read_text())
        rows = {k: v for k, v in score.items() if not k.startswith("_")}
        if champion not in rows:
            continue
        timing_valid = bool(json.loads(meta_path.read_text())["timing_valid"])
        out[name] = win_table(rows, spec.scorer, champion, timing_valid)
    return out


def render(board: dict[str, list[Verdict]], champion: str) -> str:
    wins = sum(v.win is True for vs in board.values() for v in vs)
    judged = sum(v.win is not None for vs in board.values() for v in vs)
    out = [
        "# SOTA scoreboard",
        "",
        f"`{champion}` against the best published reference operating point "
        f"({', '.join(REFERENCES)}) per metric; a tie is a win. "
        f"**{wins} / {judged}** judged cells won.",
        "",
        "| Benchmark | Metric | Locus | Best reference | Value | Verdict |",
        "| :-- | :-- | --: | :-- | --: | :-: |",
    ]
    for name, verdicts in board.items():
        for v in verdicts:
            nd = v.metric.digits
            out.append(
                f"| {name} | {v.metric.label} | {fmt(v.champion, nd)} | {v.reference or '—'} | "
                f"{fmt(v.best, nd)} | {verdict_cell(v)} |"
            )
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> None:
    runs_root = Path(argv[0])
    champion = argv[1] if len(argv) > 1 else DEFAULT_CHAMPION
    board = collect(runs_root, champion)
    if not board:
        raise SystemExit(f"no scored benchmark with a `{champion}` run under {runs_root}")
    md = render(board, champion)
    (runs_root.parent / "scoreboard.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1:])
