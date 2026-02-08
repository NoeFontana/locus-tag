import argparse

import cv2
import locus
import numpy as np
from tqdm import tqdm

try:
    import rerun as rr

    RERUN_AVAILABLE = True
except ImportError:
    print("Rerun not available. Please install 'rerun-sdk'.")
    RERUN_AVAILABLE = False

from scripts.bench.utils import DatasetLoader


def run_visualization(args):
    if not RERUN_AVAILABLE:
        return

    rr.script_setup(args, "locus_debug_pipeline")

    loader = DatasetLoader()
    scenario = args.scenario
    if not loader.prepare_icra(scenario):
        print(f"Failed to prepare scenario {scenario}")
        return

    datasets = loader.find_datasets(scenario, ["tags"])

    detector = locus.Detector(
        threshold_tile_size=args.tile_size,
        quad_min_area=args.min_area,
        enable_bilateral=args.bilateral,
        upscale_factor=args.upscale,
    )

    for ds_name, img_dir, gt_map in datasets:
        print(f"\nVisualizing {ds_name}...")

        img_names = sorted(gt_map.keys())
        if args.limit:
            img_names = img_names[: args.limit]

        for i, img_name in enumerate(tqdm(img_names)):
            img_path = img_dir / img_name
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            rr.set_time(timeline="frame_idx", sequence=i)

            # --- Perform Full Detection (Single Pass) ---
            res = detector.detect_full(img)

            # 1. Input & Ground Truth
            rr.log("pipeline/0_input", rr.Image(img))

            gt_tags = gt_map.get(img_name, [])
            if gt_tags:
                gt_strips = []
                gt_labels = []
                for gt in gt_tags:
                    c = np.vstack([gt.corners, gt.corners[0]])
                    gt_strips.append(c)
                    gt_labels.append(f"GT:{gt.tag_id}")

                rr.log(
                    "pipeline/0_input/ground_truth",
                    rr.LineStrips2D(gt_strips, colors=[0, 255, 0], radii=2.0, labels=gt_labels),
                )

            # 2. Thresholding
            binarized = res.get_binarized()
            if binarized is not None:
                rr.log("pipeline/1_threshold", rr.Image(binarized))

            # 3. Segmentation
            labels = res.get_labels()
            if labels is not None:
                rr.log("pipeline/2_segmentation", rr.SegmentationImage(labels.astype(np.int32)))

            # 4. Candidates
            if res.candidates:
                cand_strips = []
                cand_colors = []
                for cand in res.candidates:
                    c = np.array(cand.corners)
                    c = np.vstack([c, c[0]])
                    cand_strips.append(c)
                    # Color by rejection reason: Red if Hamming > 0, Blue otherwise
                    color = [255, 100, 100] if cand.hamming > 0 else [100, 150, 255]
                    cand_colors.append(color)

                rr.log(
                    "pipeline/3_candidates",
                    rr.LineStrips2D(cand_strips, colors=cand_colors, radii=0.5),
                )

            # 5. Final Detections
            if res.detections:
                det_strips = []
                det_labels = []
                for det in res.detections:
                    c = np.array(det.corners)
                    c = np.vstack([c, c[0]])
                    det_strips.append(c)
                    det_labels.append(f"ID:{det.id}")

                rr.log(
                    "pipeline/4_detections",
                    rr.LineStrips2D(det_strips, colors=[255, 50, 50], radii=1.2, labels=det_labels),
                )

            # 6. Failure Diagnostics (Replacing diagnose_missed.py)
            if gt_tags:
                det_ids = {d.id for d in res.detections}
                for gt in gt_tags:
                    if gt.tag_id not in det_ids:
                        # Find closest candidate
                        best_cand = None
                        min_rmse = float("inf")
                        for cand in res.candidates:
                            cand_corners = np.array(cand.corners)
                            for rot in range(4):
                                shifted = np.roll(cand_corners, rot, axis=0)
                                rmse = np.sqrt(np.mean((shifted - gt.corners) ** 2))
                                if rmse < min_rmse:
                                    min_rmse = rmse
                                    best_cand = cand

                        # Log diagnosis to Rerun
                        diag_path = f"diagnosis/missed_tags/{gt.tag_id}"
                        if best_cand and min_rmse < 20.0:
                            # Reverted to rejected candidate
                            rr.log(
                                f"{diag_path}/reason",
                                rr.TextLog(f"Rejected: Hamming {best_cand.hamming}"),
                            )
                            # Log the extracted bits as a 6x6 image for visual inspection
                            bits = best_cand.bits
                            grid = np.zeros((6, 6), dtype=np.uint8)
                            for r in range(6):
                                for c2 in range(6):
                                    idx = r * 6 + c2
                                    if (bits >> (35 - idx)) & 1:
                                        grid[r, c2] = 255
                            rr.log(f"{diag_path}/extracted_bits", rr.Image(grid))
                        else:
                            rr.log(f"{diag_path}/reason", rr.TextLog("No candidate found near GT"))

            # 7. Timings
            rr.log("performance/latency/total_ms", rr.Scalars(res.stats.total_ms))
            rr.log("performance/latency/threshold_ms", rr.Scalars(res.stats.threshold_ms))
            rr.log("performance/latency/segmentation_ms", rr.Scalars(res.stats.segmentation_ms))
            rr.log("performance/latency/quad_ms", rr.Scalars(res.stats.quad_extraction_ms))
            rr.log("performance/latency/decode_ms", rr.Scalars(res.stats.decoding_ms))


def main():
    parser = argparse.ArgumentParser(description="Locus Advanced Debug Visualization")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--scenario", default="forward")
    parser.add_argument("--tile-size", type=int, default=4)
    parser.add_argument("--min-area", type=int, default=10)
    parser.add_argument("--bilateral", action="store_true")
    parser.add_argument("--upscale", type=int, default=1)
    rr.script_add_args(parser)

    args = parser.parse_args()
    run_visualization(args)


if __name__ == "__main__":
    main()
