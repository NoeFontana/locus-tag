"""Render ``score.json`` + ``meta.json`` of a SOTA run into ``report.md``.

Usage::

    python -m tools.bench.sota.report <benchmark> <runs_dir> [name=unsupported-reason ...]

Besides the per-detector tables, the report carries a **win table**: the champion Locus run
(``locus_standard``) against the *best* published reference operating point on every metric
(:data:`REFERENCES`). It is the exit criterion of the SOTA programme; a tie counts as a win.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.bench.sota.score import LOCUS_PREFIX, failed_runs, run_labels
from tools.bench.sota.spec import load_specs

# Published operating points the win table compares against (`refrun` modes). AprilTag 3
# is scored and reported but is not part of the win criterion.
REFERENCES = ("aruco_nano", "opencv", "opencv_subpix", "opencv_apriltag")
DEFAULT_CHAMPION = "locus_standard"


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    higher_is_better: bool
    digits: int = 2
    timing: bool = False


_LATENCY = Metric("ms_mean", "ms/img", higher_is_better=False, digits=1, timing=True)
METRICS: dict[str, list[Metric]] = {
    "liu4k": [
        Metric("recall", "Recall %", True),
        Metric("precision", "Precision %", True),
        Metric("f1", "F1", True),
        Metric(
            "corner_debiased_common_median", "Corner px, debiased, common tags (median)", False, 3
        ),
        _LATENCY,
    ],
    "euroc": [
        Metric("recall", "Recall %", True),
        Metric("precision", "Precision %", True),
        Metric("loo_common_median_px", "LOO px, common tags (median)", False, 3),
        Metric("loo_common_p90_px", "LOO px, common tags (p90)", False, 3),
        _LATENCY,
    ],
    "gt": [
        Metric("recall", "Recall %", True),
        Metric("precision", "Precision %", True),
        Metric("f1", "F1", True),
        Metric(
            "corner_debiased_common_mean", "Corner RMSE px, debiased, common tags (mean)", False, 3
        ),
        Metric(
            "corner_debiased_common_p90", "Corner RMSE px, debiased, common tags (p90)", False, 3
        ),
        _LATENCY,
    ],
}


def metrics_for(scorer: str) -> list[Metric]:
    return METRICS["gt" if scorer.startswith("gt-") else scorer]


@dataclass(frozen=True)
class Verdict:
    metric: Metric
    champion: float | None
    reference: str | None
    best: float | None
    # True = win or tie, False = loss, None = not comparable (missing value / invalid timing).
    win: bool | None

    @property
    def margin(self) -> float | None:
        if self.champion is None or self.best is None:
            return None
        return self.champion - self.best


def win_table(
    rows: dict[str, dict[str, Any]], scorer: str, champion: str, timing_valid: bool
) -> list[Verdict]:
    """Champion vs the best reference operating point present, per metric."""
    refs = [r for r in REFERENCES if r in rows]
    out = []
    for m in metrics_for(scorer):
        mine = rows.get(champion, {}).get(m.key)
        vals: list[tuple[str, float]] = [
            (r, float(rows[r][m.key])) for r in refs if rows[r].get(m.key) is not None
        ]
        if not vals:
            out.append(Verdict(m, mine, None, None, None))
            continue
        pick = max if m.higher_is_better else min
        ref, best = pick(vals, key=lambda rv: rv[1])
        win = None
        if mine is not None and (timing_valid or not m.timing):
            win = mine >= best if m.higher_is_better else mine <= best
        out.append(Verdict(m, mine, ref, best, win))
    return out


def fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def verdict_cell(v: Verdict) -> str:
    if v.win is None:
        return "n/a" if v.metric.timing and v.champion is not None else "—"
    return "✅" if v.win else "❌"


def render_win_table(verdicts: list[Verdict], champion: str) -> list[str]:
    out = [
        f"### Win table — `{champion}` vs best reference ({', '.join(REFERENCES)})",
        "",
        "| Metric | Locus | Best reference | Reference value | Margin | Verdict |",
        "| :-- | --: | :-- | --: | --: | :-: |",
    ]
    for v in verdicts:
        nd = v.metric.digits
        margin = v.margin
        out.append(
            f"| {v.metric.label} | {fmt(v.champion, nd)} | {v.reference or '—'} | "
            f"{fmt(v.best, nd)} | {'—' if margin is None else f'{margin:+.{nd}f}'} | "
            f"{verdict_cell(v)} |"
        )
    return out


def _side_cols(rows: dict[str, dict[str, Any]]) -> list[str]:
    first = next(iter(rows.values()))
    return list(first.get("recall_by_side", {}))


def _detector_table(scorer: str, rows: dict[str, dict[str, Any]], order: list[str]) -> list[str]:
    if scorer == "euroc":
        out = [
            "| Detector | Recall % | Precision % | FP | ms/img | LOO median px (common) "
            "| LOO p90 px (common) | LOO median px (own) |",
            "| :-- | --: | --: | --: | --: | --: | --: | --: |",
        ]
        for n in order:
            r = rows[n]
            out.append(
                f"| {n} | {fmt(r['recall'])} | {fmt(r['precision'])} | {r['fp']} | "
                f"{fmt(r['ms_mean'], 1)} | {fmt(r['loo_common_median_px'], 3)} | "
                f"{fmt(r['loo_common_p90_px'], 3)} | {fmt(r['loo_own_median_px'], 3)} |"
            )
        return out
    bins = _side_cols(rows)
    out = [
        "| Detector | Recall % | Precision % | F1 | TP | FP | ms/img | Corner px mean / median "
        "/ p90 / p99 | Common mean / p90 | Common bias | Common debiased mean / p90 | "
        + " | ".join(f"R% side {b}" for b in bins)
        + " |",
        "| :-- " + "| --: " * (10 + len(bins)) + "|",
    ]
    for n in order:
        r = rows[n]
        own = " / ".join(fmt(r.get(f"corner_{k}"), 3) for k in ("mean", "median", "p90", "p99"))
        com = " / ".join(fmt(r.get(f"corner_common_{k}"), 3) for k in ("mean", "p90"))
        bias = r.get("corner_bias_common")
        deb = " / ".join(fmt(r.get(f"corner_debiased_common_{k}"), 3) for k in ("mean", "p90"))
        out.append(
            f"| {n} | {fmt(r['recall'])} | {fmt(r['precision'])} | {fmt(r['f1'])} | "
            f"{r['tp']} | {r['fp']} | {fmt(r['ms_mean'], 1)} | {own} | {com} | "
            f"{'—' if bias is None else f'{bias:+.3f}'} | {deb} | "
            + " | ".join(fmt(r["recall_by_side"][b], 1) for b in bins)
            + " |"
        )
    return out


def locus_extensions(runs_dir: Path) -> dict[str, list[str]]:
    """The Locus native module each current Locus run imported (``locus_extension`` in its
    first JSONL record, written by ``tools.bench.sota.run``) -> the run labels that used it."""
    labels = run_labels(runs_dir)
    out: dict[str, list[str]] = {}
    for p in sorted(runs_dir.glob(f"{LOCUS_PREFIX}*.jsonl")):
        if labels is not None and p.stem not in labels:
            continue
        with open(p) as f:
            first = f.readline()
        ext = json.loads(first).get("locus_extension") if first.strip() else None
        key = f"`{ext['path']}` (mtime {ext['mtime']})" if ext else "not recorded"
        out.setdefault(key, []).append(p.stem)
    return out


def render(
    name: str,
    runs_dir: Path,
    unsupported: dict[str, str],
    champion: str = DEFAULT_CHAMPION,
) -> str:
    spec = load_specs()[name.split("@")[0]]
    score = json.loads((runs_dir / "score.json").read_text())
    meta = json.loads((runs_dir / "meta.json").read_text())
    rows = {k: v for k, v in score.items() if not k.startswith("_")}
    order = sorted(rows, key=lambda n: -rows[n].get("f1", rows[n].get("recall", 0.0)))
    out = [f"# SOTA comparison — {name}", ""]
    timing = (
        "valid (detectors run serially)"
        if meta["timing_valid"]
        else "**invalid** (detectors ran concurrently; accuracy only)"
    )
    out += [
        "| Run | Value |",
        "| :-- | :-- |",
        f"| CPU (lscpu) | {meta['cpu_model']} |",
        f"| Kernel / rustc | {meta['kernel']} / {meta['rustc']} |",
        f"| Locus git rev | `{meta['git_rev'][:12]}` |",
        f"| References | OpenCV {meta['opencv']}, aruco_nano `{meta['aruco_nano'][:7]}` (unpatched) |",
        f"| Images | {meta['images']} (stride {meta['stride']}) |",
        f"| Threads | {meta['threads']} per detector (`RAYON_NUM_THREADS` / `cv::setNumThreads`) |",
        f"| Timing | best of {meta['reps']} per image, decode excluded; {timing} |",
    ]
    out += [
        f"| Locus extension | {ext}: {', '.join(runs)} |"
        for ext, runs in locus_extensions(runs_dir).items()
    ]
    out.append("")
    m = score.get("_meta", {})
    if spec.scorer == "euroc":
        out += [
            f"Ground-truth-free protocol (see `tools/bench/sota/score.py`): "
            f"{m.get('reference_frames')} frames with a pooled board fit; leave-one-tag-out "
            f"corner error on {m.get('common_loo_tags')} (frame, tag) pairs common to every reference "
            "detector with >= 20 % recall.",
            "",
        ]
    else:
        out += [
            f"Corner error in the OpenCV pixel convention, one fixed corner relabelling per "
            f"detector; common tags = {m.get('common_tags')} GT tags matched by the references "
            f"{', '.join(m.get('common_detectors', []))}.",
            "",
        ]
    out += _detector_table(spec.scorer, rows, order)
    out += [""]
    if champion in rows:
        out += render_win_table(
            win_table(rows, spec.scorer, champion, bool(meta["timing_valid"])), champion
        )
    else:
        out += [f"No `{champion}` run: win table skipped."]
    if unsupported:
        out += ["", "Unsupported as published (not run, never patched):", ""]
        out += [f"- **{k}** — {v}" for k, v in unsupported.items()]
    if failed := failed_runs(runs_dir):
        out += ["", "Crashed (not scored):", ""]
        out += [f"- **{k}** — {v}" for k, v in failed.items()]
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> None:
    name, runs_dir, *uns = argv
    unsupported = dict(u.split("=", 1) for u in uns)
    md = render(name, Path(runs_dir), unsupported)
    (Path(runs_dir) / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1:])
