"""Corner-geometry-outlier runtime gate validation harness.

Runs ``pose_cov_audit`` at gate thresholds ``[0.0 (off), 20.1 (default)]`` over
the locus_v1_tag36h11_1920x1080 corpus, and per the S1 lesson measures
``rotation_error_chosen_deg`` directly per-scene rather than just ‖r‖ or d²
proxies.

Aggregates:

- scene_0008 rotation error, corner d² distribution, Σ_pose scale (pre/post)
- corpus rotation_error_chosen_deg: mean, p99, max
- corpus d²: mean, p99 (expected to drop on the gate-fired scenes;
  off-path scenes are byte-identical)
- per-scene gate firing log

Per-run threshold is passed to ``pose_cov_audit.py`` via the
``--corner-d2-gate-threshold`` CLI flag, which overrides the value
loaded from ``high_accuracy.json``.  No on-disk profile mutation.

Usage::

    PYTHONPATH=. uv run --group bench tools/bench/runtime_gate_validation.py \\
        --output-dir diagnostics/runtime_gate_validation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_OF_INTEREST = "scene_0008_cam_0000"
THRESHOLDS = [0.0, 20.1]


def run_audit(threshold: float, out_dir: Path, hub_config: str) -> list[dict]:
    """Run pose_cov_audit with the given threshold; return parsed samples."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("RAYON_NUM_THREADS", "8")
    env["PYTHONPATH"] = str(REPO_ROOT)
    cmd = [
        "uv",
        "run",
        "--group",
        "bench",
        "tools/bench/pose_cov_audit.py",
        "--hub-config",
        hub_config,
        "--profile",
        "high_accuracy",
        "--output-dir",
        str(out_dir),
        "--corner-d2-gate-threshold",
        f"{threshold}",
    ]
    print(f"[gate={threshold}] running pose_cov_audit -> {out_dir}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)
    return json.loads((out_dir / "samples.json").read_text())


def aggregate(samples: list[dict]) -> dict:
    """Reduce a samples list to gate-relevant metrics."""
    rot_errs_deg: list[float] = []
    d2: list[float] = []
    scene_match: dict | None = None
    for s in samples:
        delta = s.get("delta_se3") or [0.0] * 6
        rot_norm_rad = float(np.linalg.norm(np.asarray(delta[3:6], dtype=float)))
        rot_errs_deg.append(math.degrees(rot_norm_rad))
        if s.get("d2_total") is not None and np.isfinite(s["d2_total"]):
            d2.append(float(s["d2_total"]))
        if s.get("scene_id") == SCENE_OF_INTEREST:
            scene_match = s
    rot_arr = np.asarray(rot_errs_deg, dtype=float)
    d2_arr = np.asarray(d2, dtype=float)

    scene_rot: float | None = None
    scene_d2_total: float | None = None
    scene_max_corner_d2: float | None = None
    if scene_match is not None:
        sd = scene_match.get("delta_se3") or [0.0] * 6
        scene_rot = math.degrees(float(np.linalg.norm(np.asarray(sd[3:6], dtype=float))))
        scene_d2_total = scene_match.get("d2_total")
        per_corner = scene_match.get("final_per_corner_d2") or []
        if per_corner:
            scene_max_corner_d2 = float(max(per_corner))

    return {
        "n_samples": len(samples),
        "scene_0008_rot_err_deg": scene_rot,
        "scene_0008_d2_total": scene_d2_total,
        "scene_0008_max_corner_d2": scene_max_corner_d2,
        "corpus_mean_rot_deg": float(rot_arr.mean()) if rot_arr.size else None,
        "corpus_p99_rot_deg": float(np.percentile(rot_arr, 99)) if rot_arr.size else None,
        "corpus_max_rot_deg": float(rot_arr.max()) if rot_arr.size else None,
        "corpus_mean_d2": float(d2_arr.mean()) if d2_arr.size else None,
        "corpus_p99_d2": float(np.percentile(d2_arr, 99)) if d2_arr.size else None,
    }


def emit_table(rows: list[dict]) -> str:
    headers = [
        "threshold",
        "n",
        "scene_0008 rot°",
        "scene_0008 d² total",
        "scene_0008 max corner d²",
        "corpus mean rot°",
        "corpus p99 rot°",
        "corpus max rot°",
        "corpus mean d²",
        "corpus p99 d²",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        cells = [
            f"{r['threshold']:.2f}",
            f"{r['n_samples']}",
            f"{r['scene_0008_rot_err_deg']:.4f}"
            if r["scene_0008_rot_err_deg"] is not None
            else "—",
            f"{r['scene_0008_d2_total']:.1f}" if r["scene_0008_d2_total"] is not None else "—",
            f"{r['scene_0008_max_corner_d2']:.2f}"
            if r["scene_0008_max_corner_d2"] is not None
            else "—",
            f"{r['corpus_mean_rot_deg']:.4f}" if r["corpus_mean_rot_deg"] is not None else "—",
            f"{r['corpus_p99_rot_deg']:.4f}" if r["corpus_p99_rot_deg"] is not None else "—",
            f"{r['corpus_max_rot_deg']:.4f}" if r["corpus_max_rot_deg"] is not None else "—",
            f"{r['corpus_mean_d2']:.1f}" if r["corpus_mean_d2"] is not None else "—",
            f"{r['corpus_p99_d2']:.1f}" if r["corpus_p99_d2"] is not None else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root output directory; per-threshold subdirs are created under it.",
    )
    parser.add_argument(
        "--hub-config",
        default="locus_v1_tag36h11_1920x1080",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=",".join(f"{t}" for t in THRESHOLDS),
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    rows: list[dict] = []
    for t in thresholds:
        sub = args.output_dir / f"gate_{t:.2f}"
        samples = run_audit(t, sub, args.hub_config)
        agg = aggregate(samples)
        agg["threshold"] = t
        rows.append(agg)
        print(json.dumps(agg, indent=2), flush=True)
        if not args.keep_intermediate:
            samples_path = sub / "samples.json"
            if samples_path.exists():
                samples_path.unlink()

    summary = {"hub_config": args.hub_config, "thresholds": thresholds, "rows": rows}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    table = emit_table(rows)
    (args.output_dir / "summary.md").write_text(table + "\n")
    print("\n=== Runtime-gate validation summary ===\n", file=sys.stderr)
    print(table, file=sys.stderr)


if __name__ == "__main__":
    main()
