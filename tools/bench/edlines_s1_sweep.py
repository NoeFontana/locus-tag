"""S1 corner-exclusion Δ sweep for the EdLines Phase 1 segmentation experiment.

Runs ``pose_cov_audit`` over the locus_v1_tag36h11_1920x1080 corpus for each
``LOCUS_EDLINES_CORNER_EXCLUSION_PX`` value and aggregates:

- scene_0008_cam_0000 corner-1 residual norm
- corpus mean, p99, max of corner residual norms
- corpus mean / p99 of d²_total
- d² KL divergence to χ²(6)

Reproduces the falsification + acceptance checks from §3.4 / §3.5 of
``docs/engineering/edlines_segmentation_design_2026-05-03.md``.

Usage::

    PYTHONPATH=. uv run --group bench tools/bench/edlines_s1_sweep.py \\
        --output-dir diagnostics/edlines_s1_sweep
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE_OF_INTEREST = "scene_0008_cam_0000"
CORNER_OF_INTEREST = 1
DELTAS = [0.0, 1.0, 2.0, 3.0, 5.0]


def run_audit(delta: float, out_dir: Path, hub_config: str) -> list[dict]:
    """Run pose_cov_audit with the given Δ; return parsed samples list."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LOCUS_EDLINES_CORNER_EXCLUSION_PX"] = f"{delta}"
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
    ]
    print(f"[Δ={delta}] running pose_cov_audit -> {out_dir}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)
    samples_path = out_dir / "samples.json"
    return json.loads(samples_path.read_text())


def aggregate(samples: list[dict]) -> dict:
    """Reduce a samples list to the metrics tracked by the design memo."""
    scene_match = next((s for s in samples if s.get("scene_id") == SCENE_OF_INTEREST), None)
    scene_corner1: float | None = None
    if scene_match is not None:
        norms = scene_match.get("corner_residual_norms_px") or []
        if len(norms) > CORNER_OF_INTEREST:
            scene_corner1 = float(norms[CORNER_OF_INTEREST])

    all_norms: list[float] = []
    all_d2: list[float] = []
    for s in samples:
        norms = s.get("corner_residual_norms_px") or []
        all_norms.extend(float(n) for n in norms if n is not None)
        d2 = s.get("d2_total")
        if d2 is not None and np.isfinite(d2):
            all_d2.append(float(d2))

    norms_arr = np.asarray(all_norms, dtype=float)
    d2_arr = np.asarray(all_d2, dtype=float)
    return {
        "n_samples": len(samples),
        "n_corner_norms": int(norms_arr.size),
        "scene_0008_corner1_r": scene_corner1,
        "corpus_mean_r": float(norms_arr.mean()) if norms_arr.size else None,
        "corpus_p99_r": float(np.percentile(norms_arr, 99)) if norms_arr.size else None,
        "corpus_max_r": float(norms_arr.max()) if norms_arr.size else None,
        "corpus_mean_d2": float(d2_arr.mean()) if d2_arr.size else None,
        "corpus_p99_d2": float(np.percentile(d2_arr, 99)) if d2_arr.size else None,
    }


def emit_table(rows: list[dict]) -> str:
    headers = [
        "Δ_px",
        "n",
        "scene_0008 c1 ‖r‖",
        "corpus mean ‖r‖",
        "corpus p99 ‖r‖",
        "corpus max ‖r‖",
        "corpus mean d²",
        "corpus p99 d²",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        cells = [
            f"{r['delta']:.1f}",
            f"{r['n_samples']}",
            f"{r['scene_0008_corner1_r']:.3f}" if r["scene_0008_corner1_r"] is not None else "—",
            f"{r['corpus_mean_r']:.4f}" if r["corpus_mean_r"] is not None else "—",
            f"{r['corpus_p99_r']:.4f}" if r["corpus_p99_r"] is not None else "—",
            f"{r['corpus_max_r']:.3f}" if r["corpus_max_r"] is not None else "—",
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
        help="Root output directory; per-Δ subdirs are created under it.",
    )
    parser.add_argument(
        "--hub-config",
        default="locus_v1_tag36h11_1920x1080",
    )
    parser.add_argument(
        "--deltas",
        type=str,
        default=",".join(f"{d}" for d in DELTAS),
        help="Comma-separated list of Δ values to sweep.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Don't delete per-Δ samples.json after aggregation (large).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]

    rows: list[dict] = []
    for delta in deltas:
        sub = args.output_dir / f"delta_{delta:.1f}"
        samples = run_audit(delta, sub, args.hub_config)
        agg = aggregate(samples)
        agg["delta"] = delta
        rows.append(agg)
        print(json.dumps(agg, indent=2), flush=True)
        if not args.keep_intermediate:
            samples_path = sub / "samples.json"
            if samples_path.exists():
                samples_path.unlink()

    summary = {
        "hub_config": args.hub_config,
        "deltas": deltas,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    table = emit_table(rows)
    (args.output_dir / "summary.md").write_text(table + "\n")
    print("\n=== S1 Δ sweep summary ===\n", file=sys.stderr)
    print(table, file=sys.stderr)


if __name__ == "__main__":
    main()
