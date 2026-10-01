"""Render ``score.json`` + ``meta.json`` of a SOTA run into ``report.md``.

Usage::

    python -m tools.bench.sota.report <dataset> <runs_dir> [name=unsupported-reason ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render(dataset: str, runs_dir: Path, unsupported: dict[str, str]) -> str:
    score = json.loads((runs_dir / "score.json").read_text())
    meta = json.loads((runs_dir / "meta.json").read_text())
    rows = {k: v for k, v in score.items() if not k.startswith("_")}
    order = sorted(rows, key=lambda n: -rows[n].get("f1", rows[n].get("recall", 0.0)))
    out = [f"# SOTA comparison — {dataset}", ""]
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
        "",
    ]
    if dataset == "liu4k":
        bins = list(next(iter(rows.values()))["recall_by_side"])
        out += [
            "| Detector | Recall % | Precision % | F1 | TP | FP | ms/img | "
            + " | ".join(f"R% side {b}" for b in bins)
            + " |",
            "| :-- " + "| --: " * (6 + len(bins)) + "|",
        ]
        for n in order:
            r = rows[n]
            out.append(
                f"| {n} | {_fmt(r['recall'])} | {_fmt(r['precision'])} | {_fmt(r['f1'])} | "
                f"{r['tp']} | {r['fp']} | {_fmt(r['ms_mean'], 1)} | "
                + " | ".join(_fmt(r["recall_by_side"][b], 1) for b in bins)
                + " |"
            )
    else:
        m = score.get("_meta", {})
        out += [
            f"Ground-truth-free protocol (see `tools/bench/sota/score.py`): "
            f"{m.get('reference_frames')} frames with a pooled board fit; leave-one-tag-out "
            f"corner error on {m.get('common_loo_tags')} (frame, tag) pairs common to every "
            "detector with >= 20 % recall.",
            "",
            "| Detector | Recall % | Precision % | FP | ms/img | LOO median px (common) "
            "| LOO p90 px (common) | LOO median px (own) |",
            "| :-- | --: | --: | --: | --: | --: | --: | --: |",
        ]
        for n in order:
            r = rows[n]
            out.append(
                f"| {n} | {_fmt(r['recall'])} | {_fmt(r['precision'])} | {r['fp']} | "
                f"{_fmt(r['ms_mean'], 1)} | {_fmt(r['loo_common_median_px'], 3)} | "
                f"{_fmt(r['loo_common_p90_px'], 3)} | {_fmt(r['loo_own_median_px'], 3)} |"
            )
    if unsupported:
        out += ["", "Unsupported as published (not run, never patched):", ""]
        out += [f"- **{k}** — {v}" for k, v in unsupported.items()]
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> None:
    dataset, runs_dir, *uns = argv
    unsupported = dict(u.split("=", 1) for u in uns)
    md = render(dataset, Path(runs_dir), unsupported)
    (Path(runs_dir) / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1:])
