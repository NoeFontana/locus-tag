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

from benchmarks.utils import DatasetLoader


def run_visualization(args):
    if not RERUN_AVAILABLE:
        return

    rr.script_setup(args, "locus_debug_pipeline")

    loader = DatasetLoader()
    scenario = "forward"
    if not loader.prepare_scenario(scenario):
        print(f"Failed to prepare scenario {scenario}")
        return

    datasets = loader.find_datasets(scenario, ["tags"])

    # Configure detector with reasonable defaults for debugging
    detector = locus.Detector(
        threshold_tile_size=4,
        quad_min_area=4,
        enable_bilateral=False,
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
            # This captures all intermediate data in one go!
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

            # 2. Thresholding (Binarized Image)
            binarized = res.get_binarized()
            if binarized is not None:
                rr.log("pipeline/1_threshold", rr.Image(binarized))

            # 3. Segmentation (Labels)
            labels = res.get_labels()
            if labels is not None:
                # Cast to int32 for Rerun segmentation image
                rr.log("pipeline/2_segmentation", rr.SegmentationImage(labels.astype(np.int32)))

            # 4. Candidates (All quads found)
            if res.candidates:
                cand_strips = []
                for cand in res.candidates:
                    c = np.array(cand.corners)
                    c = np.vstack([c, c[0]])
                    cand_strips.append(c)

                # Log to dedicated view and overlay on input
                rr.log(
                    "pipeline/3_candidates",
                    rr.LineStrips2D(cand_strips, colors=[100, 150, 255], radii=0.5),
                )
                rr.log(
                    "pipeline/0_input/candidates",
                    rr.LineStrips2D(cand_strips, colors=[100, 150, 255], radii=0.5),
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
                rr.log(
                    "pipeline/0_input/detections",
                    rr.LineStrips2D(det_strips, colors=[255, 50, 50], radii=1.2, labels=det_labels),
                )

            # 6. Performance Statistics
            # Capture individual stage timings
            rr.log("pipeline/stats/timings/threshold_ms", rr.Scalars(res.stats.threshold_ms))
            rr.log("pipeline/stats/timings/segmentation_ms", rr.Scalars(res.stats.segmentation_ms))
            rr.log(
                "pipeline/stats/timings/quad_extraction_ms",
                rr.Scalars(res.stats.quad_extraction_ms),
            )
            rr.log("pipeline/stats/timings/decoding_ms", rr.Scalars(res.stats.decoding_ms))
            rr.log("pipeline/stats/timings/total_ms", rr.Scalars(res.stats.total_ms))

            # Counts
            rr.log("pipeline/stats/counts/candidates", rr.Scalars(float(res.stats.num_candidates)))
            rr.log("pipeline/stats/counts/detections", rr.Scalars(float(res.stats.num_detections)))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Locus Pipeline Stages with Rerun (High-Quality Single Pass)"
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of images to visualize")
    rr.script_add_args(parser)

    args = parser.parse_args()

    run_visualization(args)


if __name__ == "__main__":
    main()
