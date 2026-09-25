"""Visualize pinhole-baseline detection on real EuRoC cam_april frames with Rerun.

Standalone companion to `tools/cli.py visualize` (which is hard-wired to the
ICRA/hub dataset loader and its per-image ground truth — EuRoC has none, see
`common::euroc::GroundTruth` on the Rust side). Logs the same
`pipeline/0_input` -> `pipeline/1_threshold` -> `pipeline/2_binarized` ->
`pipeline/rejected` / `pipeline/3_detections` layers as that command, with
rejected quads color-coded by `FunnelStatus` (red = contrast-gate reject,
orange = passed contrast but failed decode, grey = unclassified) — see
`locus.FunnelStatus` for what each value actually means in practice.

Also logs `pipeline/board_coverage`: once >= 4 tags decode, fits a 2D affine
map from board-plane to pixel coordinates (see `fit_affine_board_to_image`,
which mirrors `fit_affine_board_to_image` in
`crates/locus-core/tests/regression_euroc.rs::euroc_detection_baseline`
exactly — that's the same fit the relative-recall metric there is built on),
reprojects the full 6x6 grid through it, and color-codes every tag: green =
decoded, yellow = predicted in-frame but never decoded (the real recall
gap), grey = predicted out of frame. This is the single view that answers
"is a given frame's shortfall an extraction problem (yellow tag has no
rejected-quad outline nearby) or a decode problem (yellow tag lines up with
an orange rejected quad)".

Setup:
    bash scripts/fetch_euroc_calibration.sh
    uv run maturin develop --release --manifest-path crates/locus-py/Cargo.toml

Usage:
    # Visualize the single worst-offender frame found by the Rust funnel
    # diagnostic (crates/locus-core/tests/regression_euroc.rs,
    # euroc_pinhole_funnel_diagnostic), saved to a .rrd for offline viewing:
    uv run tools/viz_rerun_euroc.py --frame 1403709431837837056.png \
        --save /tmp/euroc_worst_frame.rrd

    # Or step through a stretch of frames with a live viewer:
    uv run tools/viz_rerun_euroc.py --start-frame 200 --limit 20

    # Also dump a standalone annotated PNG for the last logged frame:
    uv run tools/viz_rerun_euroc.py --frame 1403709431837837056.png \
        --overlay-png /tmp/overlay.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import locus
import numpy as np
import rerun as rr

# Known EuRoC VI-Sensor cam0 calibration (from mav0/cam0/sensor.yaml's
# published values) — mirrors crates/locus-core/tests/common/euroc.rs exactly.
CAM0_FX = 458.654
CAM0_FY = 457.296
CAM0_CX = 367.215
CAM0_CY = 248.375
CAM0_K1 = -0.28340811
CAM0_K2 = 0.07395907
CAM0_P1 = 0.00019359
CAM0_P2 = 1.76187114e-05
TAG_SIZE = 0.088

# AprilGrid layout — mirrors crates/locus-core/tests/common/euroc.rs exactly.
GRID_ROWS = 6
GRID_COLS = 6
TAG_SPACING_RATIO = 0.3


def board_obj_points() -> dict[int, np.ndarray]:
    """Board-plane (x, y) corners [TL, TR, BR, BL] for every tag ID, matching
    `AprilGridTopology::new` in crates/locus-core/src/board.rs exactly."""
    gap = TAG_SPACING_RATIO * TAG_SIZE
    step = TAG_SIZE + gap
    board_w = GRID_COLS * TAG_SIZE + (GRID_COLS - 1) * gap
    board_h = GRID_ROWS * TAG_SIZE + (GRID_ROWS - 1) * gap
    ox, oy = -board_w / 2.0, -board_h / 2.0
    out = {}
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x, y = ox + c * step, oy + r * step
            out[r * GRID_COLS + c] = np.array(
                [[x, y], [x + TAG_SIZE, y], [x + TAG_SIZE, y + TAG_SIZE], [x, y + TAG_SIZE]]
            )
    return out


def fit_affine_board_to_image(
    board_pts: np.ndarray, image_pts: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Ordinary least squares fit of board-plane -> pixel affine coefficients.

    Mirrors `fit_affine_board_to_image` in
    crates/locus-core/tests/regression_euroc.rs — see that function's doc
    comment for why affine (not a full 6-DOF pose) is the right robustness
    tradeoff here. Returns `(coeffs_x, coeffs_y)`, each `[a, b, c]` such that
    `pixel = a*bx + b*by + c`, or `None` if the fit is degenerate.
    """
    a = np.hstack([board_pts, np.ones((len(board_pts), 1))])
    try:
        coeffs_x, *_ = np.linalg.lstsq(a, image_pts[:, 0], rcond=None)
        coeffs_y, *_ = np.linalg.lstsq(a, image_pts[:, 1], rcond=None)
    except np.linalg.LinAlgError:
        return None
    return coeffs_x, coeffs_y


def project_affine(coeffs: tuple[np.ndarray, np.ndarray], board_pt: np.ndarray) -> np.ndarray:
    coeffs_x, coeffs_y = coeffs
    x, y = board_pt
    return np.array(
        [
            coeffs_x[0] * x + coeffs_x[1] * y + coeffs_x[2],
            coeffs_y[0] * x + coeffs_y[1] * y + coeffs_y[2],
        ]
    )


def build_intrinsics(distorted: bool) -> locus.CameraIntrinsics:
    if not distorted:
        return locus.CameraIntrinsics(fx=CAM0_FX, fy=CAM0_FY, cx=CAM0_CX, cy=CAM0_CY)
    return locus.CameraIntrinsics(
        fx=CAM0_FX,
        fy=CAM0_FY,
        cx=CAM0_CX,
        cy=CAM0_CY,
        distortion_model=locus.DistortionModel.BrownConrady,
        dist_coeffs=[CAM0_K1, CAM0_K2, CAM0_P1, CAM0_P2, 0.0],
    )


def log_frame(
    detector: locus.Detector,
    img_path: Path,
    intrinsics: locus.CameraIntrinsics,
    overlay_png: Path | None = None,
) -> None:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skipping unreadable image: {img_path}")
        return
    height, width = img.shape

    batch = detector.detect(img, intrinsics=intrinsics, tag_size=TAG_SIZE, debug_telemetry=True)

    rr.log("pipeline/0_input", rr.Image(img))

    if batch.telemetry is not None:
        rr.log("pipeline/1_threshold", rr.Image(batch.telemetry.threshold_map))
        rr.log("pipeline/2_binarized", rr.Image(batch.telemetry.binarized))

    if batch.rejected_corners is not None and len(batch.rejected_corners) > 0:
        rejected = batch.rejected_corners
        rej_errs = batch.rejected_error_rates
        rej_status = batch.rejected_funnel_status

        colors = []
        labels = []
        for j in range(len(rejected)):
            err = rej_errs[j] if rej_errs is not None else 0.0
            code = (
                int(rej_status[j]) if rej_status is not None else int(locus.FunnelStatus.NoneReason)
            )
            if code == locus.FunnelStatus.RejectedContrast:
                colors.append([255, 0, 0, 128])
                labels.append("Rejected: low contrast")
            elif code == locus.FunnelStatus.PassedContrast:
                colors.append([255, 165, 0, 128])
                labels.append(f"Decode fail: best Hamming {int(err)}")
            else:
                colors.append([128, 128, 128, 128])
                labels.append("Rejected Quad")

        strips = np.concatenate([rejected, rejected[:, :1, :]], axis=1)
        rr.log(
            "pipeline/rejected", rr.LineStrips2D(strips, colors=colors, labels=labels, radii=0.5)
        )
    else:
        rr.log("pipeline/rejected", rr.Clear(recursive=False))

    if len(batch) > 0:
        det_strips = []
        det_labels = []
        for j in range(len(batch)):
            c = batch.corners[j]
            det_strips.append(np.vstack([c, c[0]]))
            det_labels.append(f"ID:{batch.ids[j]}")
        rr.log(
            "pipeline/3_detections",
            rr.LineStrips2D(det_strips, colors=[0, 0, 255, 128], radii=0.5, labels=det_labels),
        )
        rr.log(
            "pipeline/0_input/detections",
            rr.LineStrips2D(det_strips, colors=[0, 0, 255, 128], radii=0.5, labels=det_labels),
        )
    else:
        rr.log("pipeline/3_detections", rr.Clear(recursive=False))
        rr.log("pipeline/0_input/detections", rr.Clear(recursive=False))

    # Board coverage: fit board-plane -> pixel affine from every decoded
    # tag's corners, reproject the full 6x6 layout, classify each tag as
    # decoded / predicted-but-missing / predicted-out-of-frame.
    coverage = None
    if len(batch) >= 4:
        obj_points = board_obj_points()
        board_pts, image_pts = [], []
        for j in range(len(batch)):
            tid = int(batch.ids[j])
            if tid not in obj_points:
                continue
            board_pts.append(obj_points[tid])
            image_pts.append(batch.corners[j])
        board_pts = np.concatenate(board_pts, axis=0)
        image_pts = np.concatenate(image_pts, axis=0)
        coeffs = fit_affine_board_to_image(board_pts, image_pts)
        if coeffs is not None:
            decoded_ids = {int(i) for i in batch.ids}
            missing_strips, missing_labels = [], []
            absent_strips = []
            for tid, corners in obj_points.items():
                if tid in decoded_ids:
                    continue
                proj = np.array([project_affine(coeffs, p) for p in corners])
                in_frame = np.all(
                    (proj[:, 0] >= 0)
                    & (proj[:, 0] < width)
                    & (proj[:, 1] >= 0)
                    & (proj[:, 1] < height)
                )
                strip = np.vstack([proj, proj[:1]])
                if in_frame:
                    missing_strips.append(strip)
                    missing_labels.append(f"predicted ID:{tid}")
                else:
                    absent_strips.append(strip)
            coverage = (len(missing_strips), len(batch))
            if missing_strips:
                rr.log(
                    "pipeline/board_coverage/missing",
                    rr.LineStrips2D(
                        missing_strips, colors=[255, 230, 0, 180], radii=0.5, labels=missing_labels
                    ),
                )
            else:
                rr.log("pipeline/board_coverage/missing", rr.Clear(recursive=False))
            if absent_strips:
                rr.log(
                    "pipeline/board_coverage/out_of_frame",
                    rr.LineStrips2D(absent_strips, colors=[100, 100, 100, 100], radii=0.3),
                )
            else:
                rr.log("pipeline/board_coverage/out_of_frame", rr.Clear(recursive=False))

            if overlay_png is not None:
                out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                for j in range(len(batch)):
                    c = batch.corners[j].astype(np.int32)
                    cv2.polylines(out, [c], True, (0, 255, 0), 2)
                for strip in missing_strips:
                    cv2.polylines(out, [strip.astype(np.int32)], True, (0, 230, 255), 2)
                for strip in absent_strips:
                    cv2.polylines(out, [strip.astype(np.int32)], True, (100, 100, 100), 1)
                cv2.imwrite(str(overlay_png), out)
                print(f"  wrote overlay: {overlay_png}")

    n_rejected = len(batch.rejected_corners) if batch.rejected_corners is not None else 0
    coverage_msg = ""
    if coverage is not None:
        n_missing, n_decoded = coverage
        n_present = n_missing + n_decoded
        coverage_msg = f", relative recall {100.0 * n_decoded / n_present:.1f}% ({n_decoded}/{n_present} present)"
    print(
        f"{img_path.name}: {len(batch) + n_rejected} quads extracted, "
        f"{len(batch)} decoded, {n_rejected} rejected{coverage_msg}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize EuRoC pinhole-baseline detection with Rerun"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("tests/data/euroc/cam_april/mav0/cam0/data"),
        help="Directory of EuRoC cam0 PNGs",
    )
    parser.add_argument("--frame", type=str, help="Visualize this single filename only")
    parser.add_argument(
        "--start-frame", type=int, default=0, help="Start index into the sorted frame list"
    )
    parser.add_argument("--stride", type=int, default=1, help="Step between frames")
    parser.add_argument("--limit", type=int, default=5, help="Number of frames to visualize")
    parser.add_argument(
        "--distorted",
        action="store_true",
        help="Use real Brown-Conrady distortion instead of pinhole",
    )
    parser.add_argument(
        "--save", type=str, help="Save to .rrd file instead of spawning a live viewer"
    )
    parser.add_argument(
        "--overlay-png",
        type=Path,
        help="Also write a standalone annotated PNG for the last logged frame",
    )
    args = parser.parse_args()

    rr.init("locus_debug_pipeline_euroc")
    if args.save:
        rr.save(args.save)
    else:
        rr.spawn()

    intrinsics = build_intrinsics(args.distorted)
    detector = locus.Detector(config=locus.DetectorConfig.from_profile("standard"))

    if args.frame:
        paths = [args.data_dir / args.frame]
    else:
        all_paths = sorted(args.data_dir.glob("*.png"))
        paths = all_paths[args.start_frame :: args.stride][: args.limit]

    for i, img_path in enumerate(paths):
        rr.set_time(timeline="frame_idx", sequence=i)
        is_last = i == len(paths) - 1
        log_frame(detector, img_path, intrinsics, args.overlay_png if is_last else None)

    print(f"Logged {len(paths)} frame(s).")
    if args.save:
        print(f"Saved to {args.save} — open with: rerun {args.save}")


if __name__ == "__main__":
    main()
