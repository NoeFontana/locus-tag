import argparse
import csv
import multiprocessing
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import locus
import numpy as np
import rerun as rr  # type: ignore
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Constants
REPO_ID = "NoeFontana/apriltag-validation-data"
CACHE_DIR = Path("tests/data/icra2020")
SCENARIOS = ["circle", "forward", "random", "rotation"]


@dataclass
class TagGroundTruth:
    tag_id: int
    corners: np.ndarray  # 4x2 float32
    fully_visible: bool


class Icra2020Dataset:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def prepare_scenario(self, scenario: str) -> bool:
        """Downloads and extracts the scenario if needed."""
        scenario_dir = self.cache_dir / scenario
        # Crude check if populated
        if scenario_dir.exists() and any(scenario_dir.iterdir()):
            print(f"Scenario '{scenario}' seems already extracted in {scenario_dir}.")
            return True

        filename = f"{scenario}.tar.xz"
        print(f"Downloading {filename}...")
        try:
            archive_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=self.cache_dir,
                local_dir_use_symlinks=False,
            )
            archive_path = Path(archive_path)
        except Exception as e:
            print(f"Failed to download {scenario}: {e}")
            return False

        print(f"Extracting {archive_path} to {self.cache_dir}...")
        try:
            with tarfile.open(archive_path, "r:xz") as tar:
                tar.extractall(path=self.cache_dir)
        except Exception as e:
            print(f"Failed to extract {scenario}: {e}")
            return False

        print(f"Removing archive {archive_path}...")
        try:
            archive_path.unlink()
        except Exception as e:
            print(f"Warning: Failed to delete archive {archive_path}: {e}")

        return True

    def find_datasets(
        self, scenario: str, filter_types: list[str]
    ) -> list[tuple[str, Path, dict[str, list[TagGroundTruth]]]]:
        """
        Returns list of (dataset_name, images_dir, ground_truth_map)
        ground_truth_map: image_name -> [TagGroundTruth, ...]
        """
        search_dir = self.cache_dir / scenario
        if not search_dir.exists():
            search_dir = self.cache_dir  # fallback

        csv_files = list(search_dir.rglob("tags.csv"))
        csv_files = [f for f in csv_files if scenario in str(f)]

        results = []
        for csv_file in csv_files:
            gt_data = self._parse_csv(csv_file)
            if not gt_data:
                continue

            parent_dir = csv_file.parent
            all_subdirs = [d for d in parent_dir.iterdir() if d.is_dir() and "images" in d.name]
            if not all_subdirs:
                all_subdirs = [parent_dir]

            # Filter subdirs
            subdirs = []
            for d in all_subdirs:
                d_name = d.name
                if (
                    "tags" in filter_types
                    and "pure_tags" in d_name
                    or "checkerboard" in filter_types
                    and "checkerboard" in d_name
                    or d == parent_dir
                    and "tags" in filter_types
                ):
                    subdirs.append(d)

            for subdir in subdirs:
                subdir_name = subdir.name
                full_dataset_name = (
                    f"{scenario}/{subdir_name}" if subdir != parent_dir else f"{scenario}/root"
                )
                results.append((full_dataset_name, subdir, gt_data))

        return results

    def _parse_csv(self, csv_file: Path) -> dict[str, list[TagGroundTruth]]:
        gt_map = {}  # image_name -> {tid -> {corners, visible}}

        with open(csv_file) as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            fieldnames = [c.strip() for c in reader.fieldnames] if reader.fieldnames else []
            reader.fieldnames = fieldnames

            required = ["image", "tag_id", "corner", "ground_truth_x", "ground_truth_y"]
            if not all(k in fieldnames for k in required):
                print(f"Skipping {csv_file}: Missing columns {required}")
                return {}

            for row in reader:
                img = row["image"]
                tid = int(row["tag_id"])
                corner_idx = int(row["corner"])
                x = float(row["ground_truth_x"])
                y = float(row["ground_truth_y"])
                full_vis = int(row.get("tag_fully_visible", 1)) == 1

                if img not in gt_map:
                    gt_map[img] = {}
                if tid not in gt_map[img]:
                    gt_map[img][tid] = {"corners": [None] * 4, "fully_visible": full_vis}

                if 0 <= corner_idx < 4:
                    gt_map[img][tid]["corners"][corner_idx] = [x, y]
                    if not full_vis:
                        gt_map[img][tid]["fully_visible"] = False

        # Flatten structure
        final_map = {}
        for img, tags in gt_map.items():
            valid_tags = []
            for tid, data in tags.items():
                if any(c is None for c in data["corners"]):
                    continue

                # We can enforce visibility filtering here or later.
                # Let's keep data and let evaluator decide?
                # Benchmarker uses filtered data for metric, so let's filter here.
                if not data["fully_visible"]:
                    continue

                valid_tags.append(
                    TagGroundTruth(
                        tag_id=tid,
                        corners=np.array(data["corners"], dtype=np.float32),
                        fully_visible=data["fully_visible"],
                    )
                )
            if valid_tags:
                final_map[img] = valid_tags

        return final_map


@dataclass
class EvalResult:
    image_name: str
    tags_gt: int = 0
    correct: int = 0
    corner_error_sum: float = 0.0
    corner_error_count: int = 0
    false_positives: int = 0
    num_candidates: int = 0
    error: str | None = None
    # For visualization
    img_shape: tuple[int, int] | None = None
    detections: list[Any] = field(default_factory=list)
    gt_tags: list[TagGroundTruth] = field(default_factory=list)


def process_image(args: tuple[Path, list[TagGroundTruth], bool]) -> EvalResult:
    img_path, gt_tags, visualize = args
    res = EvalResult(image_name=img_path.name, gt_tags=gt_tags)

    if not img_path.exists() and not img_path.suffix:
        # Try extensions
        for ext in [".png", ".jpg"]:
            p = img_path.with_suffix(ext)
            if p.exists():
                img_path = p
                break

    if not img_path.exists():
        return res

    try:
        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return res

        h, w = img.shape
        # center_x, center_y = w / 2, h / 2

        if visualize:
            res.img_shape = img.shape

        detector = locus.Detector(
            quad_min_area=8,  # Lowered to detect smaller tags
            quad_max_aspect_ratio=20.0,
            quad_min_edge_score=0.3, # Lowered for small tags with weak gradients
            threshold_min_range=2,  # Lower to capture low-contrast edges
            enable_bilateral=False,  # Disabled for speed
            enable_adaptive_window=False,  # Disabled for speed
        )
        # Detect with stats
        detections, stats = detector.detect_with_stats(img)
        res.num_candidates = stats.num_candidates

        # Convert to dicts for pickling
        serializable_dets = []
        for d in detections:
            serializable_dets.append(
                {
                    "id": d.id,
                    "hamming": d.hamming,
                    "decision_margin": d.decision_margin,
                    "corners": list(d.corners),
                    "center": d.center,
                }
            )
        res.detections = serializable_dets

        # Match detections to GT
        # Simple proximity matching: if center within threshold
        res.tags_gt = len(gt_tags)

        # Keep track of matched GTs to avoid double counting
        matched_gt_indices = set()

        for det in detections:
            # Find closest GT
            min_dist = float("inf")
            best_gt_idx = -1

            for idx, gt in enumerate(gt_tags):
                # Assuming TagGroundTruth has a 'center' attribute or we calculate it
                # For now, let's calculate it from corners
                gt_center = np.mean(gt.corners, axis=0)
                dist = np.linalg.norm(np.array(det.center) - gt_center)
                if dist < min_dist:
                    min_dist = dist
                    best_gt_idx = idx

            # Threshold for match: e.g. 10 pixels
            if (
                min_dist < 20.0
                and best_gt_idx != -1
                and gt_tags[best_gt_idx].tag_id == det.id
                and best_gt_idx not in matched_gt_indices
            ):
                matched_gt_indices.add(best_gt_idx)
                res.correct += 1

                # Calculate corner error for matched tag
                gt = gt_tags[best_gt_idx]
                det_corners = np.array(det.corners, dtype=np.float32)

                # Try all corner orderings (rotations + winding) to find best match
                best_err = float("inf")
                best_vecs = None
                for ordering in [det_corners, det_corners[::-1]]:  # CCW and CW
                    for rot in range(4):
                        rotated = np.roll(ordering, rot, axis=0)
                        diffs = gt.corners - rotated
                        err = np.sqrt(np.mean(np.sum(diffs**2, axis=1)))
                        if err < best_err:
                            best_err = err
                            best_vecs = diffs

                res.corner_error_sum += best_err
                res.corner_error_count += 1

        res.false_positives = len(detections) - res.correct

    except Exception:
        pass

    return res


class BenchmarkRunner:
    def __init__(self, dataset: Icra2020Dataset, visualize: bool = False):
        self.dataset = dataset
        self.visualize = visualize
        self.max_workers = multiprocessing.cpu_count()

        if self.visualize:
            rr.init("ICRA2020 Benchmark", spawn=False)
            rr.save("locus_benchmark.rrd")

    def run(self, scenarios: list[str], types: list[str], limit: int | None = None, skip: int = 0):
        print(f"Running benchmark with {self.max_workers} workers. Visualization: {self.visualize}")

        overall = {"gt": 0, "det": 0, "err_sum": 0.0, "err_cnt": 0, "candidates": 0}
        report = []

        for scenario in scenarios:
            if not self.dataset.prepare_scenario(scenario):
                continue

            datasets = self.dataset.find_datasets(scenario, types)
            print(f"Found {len(datasets)} datasets for {scenario}")

            for ds_name, img_dir, gt_map in datasets:
                print(f"Evaluating {ds_name} ({len(gt_map)} images)...")

                tasks = []
                for img_name, tags in gt_map.items():
                    tasks.append((img_dir / img_name, tags, self.visualize))

                # Apply skip/limit if specified
                if skip > 0:
                    tasks = tasks[skip:]
                    print(f"  (Skipped first {skip} images)")
                if limit is not None:
                    tasks = tasks[:limit]
                    print(f"  (Limited to {len(tasks)} images for debugging)")

                run_stats = {"gt": 0, "det": 0, "err_sum": 0.0, "err_cnt": 0, "fp": 0, "candidates": 0}

                # Sequential matching for rerun?
                # Rerun in parallel is tricky due to unrelated timelines if not careful.
                # But we can log based on image name or index.
                # However, usually easier to visualize sequentially or log from main thread.
                # Let's run parallel but return visualization data, then log in main.

                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {executor.submit(process_image, t): t for t in tasks}

                    for _, future in tqdm(
                        enumerate(as_completed(futures)), total=len(futures), desc=f"Eval {ds_name}"
                    ):
                        res = future.result()

                        run_stats["gt"] += res.tags_gt
                        run_stats["det"] += res.correct
                        run_stats["err_sum"] += res.corner_error_sum
                        run_stats["err_cnt"] += res.corner_error_count
                        run_stats["fp"] += res.false_positives
                        run_stats["candidates"] += res.num_candidates

                        if self.visualize:
                            self._log_visualization(ds_name, img_dir, res)

                recall = (run_stats["det"] / run_stats["gt"] * 100) if run_stats["gt"] > 0 else 0
                avg_err = (
                    (run_stats["err_sum"] / run_stats["err_cnt"]) if run_stats["err_cnt"] > 0 else 0
                )

                report.append(
                    (ds_name, run_stats["gt"], run_stats["det"], recall, avg_err, run_stats["fp"], run_stats["candidates"])
                )

                overall["gt"] += run_stats["gt"]
                overall["det"] += run_stats["det"]
                overall["err_sum"] += run_stats["err_sum"]
                overall["err_cnt"] += run_stats["err_cnt"]
                overall["fp"] = overall.get("fp", 0) + run_stats["fp"]
                overall["candidates"] += run_stats["candidates"]

        self._print_report(report, overall)

    def _log_visualization(self, dataset_name: str, img_dir: Path, res: EvalResult):
        # We use dataset/image_name as entity path (without extension for cleaner paths)
        image_stem = Path(res.image_name).stem
        base_path = f"benchmark/{dataset_name}/{image_stem}"

        rr.set_time(timeline="step", sequence=0)

        # Log Image
        img_path = img_dir / res.image_name
        if img_path.exists():
            # cv2 reads BGR, Rerun expects RGB
            img = cv2.imread(str(img_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Log image
                rr.log(base_path, rr.Image(img_rgb))

        # Log GT
        for gt in res.gt_tags:
            corners = np.vstack([gt.corners, gt.corners[0]])
            rr.log(
                f"{base_path}/gt/tag_{gt.tag_id}",
                rr.LineStrips2D(corners, radii=0.5, colors=[0, 255, 0], draw_order=10.0),
            )
            rr.log(
                f"{base_path}/gt/tag_{gt.tag_id}/labels",
                rr.Points2D(gt.corners, labels=["0", "1", "2", "3"], radii=0, draw_order=12.0),
            )

        # Log Det
        for det in res.detections:
            corners = np.array(det["corners"])
            corners = np.vstack([corners, corners[0]])
            tid = det["id"]
            rr.log(
                f"{base_path}/det/tag_{tid}",
                rr.LineStrips2D(corners, radii=0.5, colors=[255, 0, 0], draw_order=11.0),
            )
            # Log center
            rr.log(
                f"{base_path}/det/tag_{tid}/center",
                rr.Points2D([det["center"]], labels=[f"ID {tid}"], colors=[255, 0, 0]),
            )

    def _print_report(self, report, overall):
        print("\n" + "=" * 115)
        print(
            f"{'Dataset':<35} | {'Tags':<6} | {'Det':<6} | {'Recall %':<10} | {'Error (px)':<10} | {'FP':<5} | {'Prec %':<8} | {'Quads':<8}"
        )
        print("-" * 115)
        for row in sorted(report, key=lambda x: x[0]):
            ds, gt, det, rec, err, fp, quads = row
            prec = (det / (det + fp) * 100) if (det + fp) > 0 else 100.0
            print(
                f"{ds:<35} | {gt:<6} | {det:<6} | {rec:<10.2f} | {err:<10.4f} | {fp:<5} | {prec:<8.2f} | {quads:<8}"
            )
        print("-" * 115)

        tot_rec = (overall["det"] / overall["gt"] * 100) if overall["gt"] > 0 else 0
        tot_err = (overall["err_sum"] / overall["err_cnt"]) if overall["err_cnt"] > 0 else 0
        tot_fp = overall.get("fp", 0)
        tot_prec = (
            (overall["det"] / (overall["det"] + tot_fp) * 100)
            if (overall["det"] + tot_fp) > 0
            else 100.0
        )
        print(
            f"{'TOTAL':<35} | {overall['gt']:<6} | {overall['det']:<6} | {tot_rec:<10.2f} | {tot_err:<10.4f} | {tot_fp:<5} | {tot_prec:<8.2f} | {overall['candidates']:<8}"
        )
        print("=" * 115)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ICRA 2020 AprilTag Benchmark")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--types", nargs="+", choices=["tags", "checkerboard"], default=["tags"])
    parser.add_argument("--visualize", action="store_true", help="Enable Rerun visualization")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of images to process (for debugging)"
    )
    parser.add_argument("--skip", type=int, default=0, help="Skip first N images (for debugging)")
    args = parser.parse_args()

    dataset = Icra2020Dataset()
    runner = BenchmarkRunner(dataset, visualize=args.visualize)
    runner.run(args.scenarios, args.types, limit=args.limit, skip=args.skip)
