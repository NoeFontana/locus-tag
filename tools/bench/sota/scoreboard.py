"""Cross-benchmark win table: one Locus run against the best reference, everywhere it was scored.

Usage::

    python -m tools.bench.sota.scoreboard <runs_root> [champion]

Reads every ``<runs_root>/<benchmark>/score.json`` (+ ``meta.json``) whose directory name is
a ``[sota.*]`` benchmark and writes ``<runs_root>/../scoreboard.md``. Latency cells are only
judged from runs whose timing is valid (detectors run serially): the benchmark's own run, or
else a ``<name>@<tag>`` timing run next to it.
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


def _load(run_dir: Path) -> tuple[dict[str, dict], bool] | None:
    score_path, meta_path = run_dir / "score.json", run_dir / "meta.json"
    if not (score_path.exists() and meta_path.exists()):
        return None
    score = json.loads(score_path.read_text())
    rows = {k: v for k, v in score.items() if not k.startswith("_")}
    return rows, bool(json.loads(meta_path.read_text())["timing_valid"])


def collect(runs_root: Path, champion: str) -> dict[str, list[Verdict]]:
    """Accuracy cells from ``<runs_root>/<name>``; latency cells from it when its timing is
    valid, else from the first timing-valid ``<name>@<tag>`` run (e.g. a strided serial run)."""
    specs = load_specs()
    out: dict[str, list[Verdict]] = {}
    for name, spec in specs.items():
        loaded = _load(runs_root / name)
        if loaded is None or champion not in loaded[0]:
            continue
        rows, timing_valid = loaded
        verdicts = win_table(rows, spec.scorer, champion, timing_valid)
        if not timing_valid:
            for tagged in sorted(runs_root.glob(f"{name}@*")):
                timed = _load(tagged)
                if timed is None or not timed[1] or champion not in timed[0]:
                    continue
                latency = {
                    v.metric.key: v
                    for v in win_table(timed[0], spec.scorer, champion, True)
                    if v.metric.timing
                }
                verdicts = [latency.get(v.metric.key, v) for v in verdicts]
                break
        out[name] = verdicts
    return out


def collect_bias(runs_root: Path, champion: str) -> dict[str, dict[str, float]]:
    """Mean radial corner offset (px, + = outward) of the champion and each reference on the
    common tags, per benchmark: the photometric part of corner error the debiased cells set
    aside, reported so a systematic offset stays visible."""
    out: dict[str, dict[str, float]] = {}
    for name in load_specs():
        loaded = _load(runs_root / name)
        if loaded is None or champion not in loaded[0]:
            continue
        rows = loaded[0]
        biases = {
            d: float(rows[d]["corner_bias_common"])
            for d in (champion, *REFERENCES)
            if rows.get(d, {}).get("corner_bias_common") is not None
        }
        if champion in biases:
            out[name] = biases
    return out


def render(
    board: dict[str, list[Verdict]],
    champion: str,
    bias: dict[str, dict[str, float]] | None = None,
) -> str:
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
    if bias:
        out += [
            "",
            "Corner cells are judged on the RMSE left after removing each detector's mean radial",
            "offset on the benchmark; the offsets themselves (px, + = outward) are not judged:",
            "",
            f"| Benchmark | {champion} | " + " | ".join(REFERENCES) + " |",
            "| :-- " + "| --: " * (1 + len(REFERENCES)) + "|",
        ]
        for name, b in bias.items():
            cells = [b.get(d) for d in (champion, *REFERENCES)]
            out.append(
                f"| {name} | " + " | ".join("—" if c is None else f"{c:+.3f}" for c in cells) + " |"
            )
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> None:
    runs_root = Path(argv[0])
    champion = argv[1] if len(argv) > 1 else DEFAULT_CHAMPION
    board = collect(runs_root, champion)
    if not board:
        raise SystemExit(f"no scored benchmark with a `{champion}` run under {runs_root}")
    md = render(board, champion, collect_bias(runs_root, champion))
    (runs_root.parent / "scoreboard.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1:])
