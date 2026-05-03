"""Aggregate per-stage latencies from JSON telemetry dumps.

Reads `target/profiling/*_events.json` files produced by
`TELEMETRY_MODE=json`, groups span CLOSE events by canonical pipeline
stage, and prints per-stage p50/p95/p99 latency in milliseconds.

Usage:
    uv run python tools/bench/aggregate_telemetry.py target/profiling/

The architecture defines six major stages (Preprocessing, Segmentation,
Quad Extraction, Decoding, Pose Refinement, Phase C.5/board/telemetry
tail). Note that `pipeline::*` spans only cover a subset of these:
segmentation (`label_components_lsl`) is not instrumented at the time
of writing, so a "Segmentation" line is reported as
`uninstrumented` for transparency.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import quantiles

# Map raw `pipeline::*` span names to the architectural stage they belong to.
SPAN_TO_STAGE: dict[str, str] = {
    "pipeline::threshold_compute_stats": "Preprocessing",
    "pipeline::threshold_apply_map": "Preprocessing",
    "pipeline::threshold_integral": "Preprocessing",
    "pipeline::threshold_gradient_window": "Preprocessing",
    "pipeline::quad_extraction": "Quad Extraction",
    "pipeline::quad_extraction_camera": "Quad Extraction",
    "pipeline::homography_pass": "Decoding (Hard)",
    "pipeline::decoding_pass": "Decoding (Hard)",
    "pipeline::decoding_pass_distortion": "Decoding (Hard)",
    "pipeline::pose_refinement": "Pose Refinement",
    "pipeline::estimate_tag_pose": "Pose Refinement",
    "pipeline::estimate_tag_pose_diag": "Pose Refinement",
}

# Architectural stage order for stable reporting.
STAGE_ORDER: list[str] = [
    "Preprocessing",
    "Segmentation",
    "Quad Extraction",
    "Decoding (Hard)",
    "Pose Refinement",
    "Telemetry / Tail",
]

DURATION_RE = re.compile(r"^(?P<value>[0-9.]+)(?P<unit>ns|µs|us|ms|s)$")
UNIT_TO_MS: dict[str, float] = {
    "ns": 1e-6,
    "µs": 1e-3,
    "us": 1e-3,
    "ms": 1.0,
    "s": 1e3,
}


def parse_duration_ms(text: str) -> float:
    """Convert tracing-formatted duration strings to milliseconds."""
    match = DURATION_RE.match(text.strip())
    if not match:
        raise ValueError(f"unrecognised duration: {text!r}")
    value = float(match["value"])
    return value * UNIT_TO_MS[match["unit"]]


def iter_close_events(path: Path) -> Iterable[tuple[str, float, bool]]:
    """Yield (span_name, busy_ms, is_top_level) tuples for every CLOSE event.

    `is_top_level` is True iff the span has no parent (`spans` is empty).
    The CLOSE event's `time.busy` is wall-clock time inside that span,
    *including* any nested children, so summing top-level spans gives an
    accurate per-frame total without double counting.
    """
    with path.open() as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(
                    f"warning: {path.name}:{line_no} not valid JSON ({exc}); skipping",
                    file=sys.stderr,
                )
                continue
            fields = event.get("fields", {})
            if fields.get("message") != "close":
                continue
            span = event.get("span") or {}
            name = span.get("name")
            busy = fields.get("time.busy")
            if not name or not busy:
                continue
            parents = event.get("spans") or []
            is_top_level = len(parents) == 0
            try:
                yield name, parse_duration_ms(busy), is_top_level
            except ValueError as exc:
                print(
                    f"warning: {path.name}:{line_no} {exc}; skipping",
                    file=sys.stderr,
                )


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (0 < pct < 100). Empty -> NaN."""
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    # `quantiles` with n=100 gives us the 1st..99th percentiles directly,
    # which is sufficient for the p50/p95/p99 we report.
    qs = quantiles(values, n=100, method="inclusive")
    idx = max(0, min(len(qs) - 1, round(pct) - 1))
    return qs[idx]


def aggregate(profiling_dir: Path) -> dict[str, dict[str, list[float]]]:
    """Return `{file_stem: {stage: [per_frame_total_ms, ...]}}`.

    Strategy:
      * Use only top-level CLOSE events (no parent). Their `time.busy` already
        includes any children, so summing siblings within a frame gives the
        true per-frame stage total without double counting.
      * Frame boundaries are detected by `pipeline::threshold_compute_stats`,
        which fires once per frame as the very first stage.
    """
    files = sorted(profiling_dir.glob("*_events.json"))
    if not files:
        raise SystemExit(f"no *_events.json files under {profiling_dir}")
    out: dict[str, dict[str, list[float]]] = {}
    for path in files:
        per_frame_totals: dict[str, list[float]] = defaultdict(list)
        current: dict[str, float] = defaultdict(float)
        seen_first_frame = False
        for name, busy_ms, is_top_level in iter_close_events(path):
            if not is_top_level:
                continue  # avoid double-counting nested spans
            stage = SPAN_TO_STAGE.get(name)
            if stage is None:
                continue
            # `threshold_compute_stats` close marks the end of preprocessing
            # for a frame and (because spans close in order) is reliably the
            # earliest of any frame's top-level events.
            if name == "pipeline::threshold_compute_stats":
                if seen_first_frame:
                    for st, total in current.items():
                        per_frame_totals[st].append(total)
                current = defaultdict(float)
                seen_first_frame = True
            current[stage] += busy_ms
        # Flush the last frame.
        if seen_first_frame:
            for st, total in current.items():
                per_frame_totals[st].append(total)
        out[path.stem.replace("_events", "")] = dict(per_frame_totals)
    return out


def report(per_file: dict[str, dict[str, list[float]]]) -> None:
    for stem, stages in per_file.items():
        n_frames = max(
            (len(vals) for vals in stages.values()),
            default=0,
        )
        print(f"\n# {stem}")
        print(f"frames: {n_frames}")
        header = (
            f"{'Stage':<20s} {'frames':>7s} "
            f"{'p50 (ms)':>10s} {'p95 (ms)':>10s} {'p99 (ms)':>10s} "
            f"{'mean (ms)':>10s}"
        )
        print(header)
        print("-" * len(header))
        for stage in STAGE_ORDER:
            vals = stages.get(stage, [])
            if not vals:
                if stage == "Segmentation":
                    note = "uninstrumented (folded into Quad Extraction)"
                    print(f"{stage:<20s} {note:>50s}")
                continue
            sorted_vals = sorted(vals)
            p50 = percentile(sorted_vals, 50)
            p95 = percentile(sorted_vals, 95)
            p99 = percentile(sorted_vals, 99)
            mean = sum(sorted_vals) / len(sorted_vals)
            print(
                f"{stage:<20s} {len(sorted_vals):>7d} "
                f"{p50:>10.3f} {p95:>10.3f} {p99:>10.3f} {mean:>10.3f}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profiling_dir",
        type=Path,
        help="Directory containing `*_events.json` files.",
    )
    args = parser.parse_args(argv)
    if not args.profiling_dir.is_dir():
        print(f"error: {args.profiling_dir} is not a directory", file=sys.stderr)
        return 2
    per_file = aggregate(args.profiling_dir)
    report(per_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
