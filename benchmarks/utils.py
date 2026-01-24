import csv
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import locus
import numpy as np
from huggingface_hub import hf_hub_download
from pupil_apriltags import Detector as AprilTagDetector

REPO_ID = "NoeFontana/apriltag-validation-data"
DEFAULT_CACHE_DIR = Path("tests/data/icra2020")


@dataclass
class TagGroundTruth:
    tag_id: int
    corners: np.ndarray  # 4x2 float32
    fully_visible: bool = True


@dataclass
class EvalResult:
    image_name: str
    gt_tags: list[TagGroundTruth] = field(default_factory=list)
    detections: list[dict[str, Any]] = field(default_factory=list)
    correct: int = 0
    false_positives: int = 0
    corner_error_sum: float = 0.0
    corner_error_count: int = 0
    num_candidates: int = 0
    latency_ms: float = 0.0


class DatasetLoader:
    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def prepare_scenario(self, scenario: str) -> bool:
        scenario_dir = self.cache_dir / scenario
        if scenario_dir.exists() and any(scenario_dir.iterdir()):
            return True

        filename = f"{scenario}.tar.xz"
        try:
            archive_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=self.cache_dir,
                local_dir_use_symlinks=False,
            )
            archive_path = Path(archive_path)
            with tarfile.open(archive_path, "r:xz") as tar:
                tar.extractall(path=self.cache_dir)
            archive_path.unlink()
            return True
        except Exception as e:
            print(f"Error preparing scenario {scenario}: {e}")
            return False

    def find_datasets(
        self, scenario: str, filter_types: list[str]
    ) -> list[tuple[str, Path, dict[str, list[TagGroundTruth]]]]:
        search_dir = self.cache_dir / scenario
        if not search_dir.exists():
            search_dir = self.cache_dir

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

            for subdir in all_subdirs:
                d_name = subdir.name
                match = False
                if "tags" in filter_types and ("pure_tags" in d_name or subdir == parent_dir):
                    match = True
                if "checkerboard" in filter_types and "checkerboard" in d_name:
                    match = True

                if match:
                    name = f"{scenario}/{d_name}" if subdir != parent_dir else f"{scenario}/root"
                    results.append((name, subdir, gt_data))
        return results

    def _parse_csv(self, csv_file: Path) -> dict[str, list[TagGroundTruth]]:
        gt_map = {}
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row["image"]
                tid = int(row["tag_id"])
                corner_idx = int(row["corner"])
                x = float(row["ground_truth_x"])
                y = float(row["ground_truth_y"])
                visible = int(row.get("tag_fully_visible", 1)) == 1

                if img not in gt_map:
                    gt_map[img] = {}
                if tid not in gt_map[img]:
                    gt_map[img][tid] = {"corners": [None] * 4, "visible": True}

                if 0 <= corner_idx < 4:
                    gt_map[img][tid]["corners"][corner_idx] = [x, y]
                if not visible:
                    gt_map[img][tid]["visible"] = False

        final_map = {}
        for img, tags in gt_map.items():
            valid = []
            for tid, data in tags.items():
                if any(c is None for c in data["corners"]) or not data["visible"]:
                    continue
                valid.append(
                    TagGroundTruth(tag_id=tid, corners=np.array(data["corners"], dtype=np.float32))
                )
            if valid:
                final_map[img] = valid
        return final_map


class Metrics:
    @staticmethod
    def compute_corner_error(det_corners: np.ndarray, gt_corners: np.ndarray) -> float:
        min_err = float("inf")
        # Try rotations
        for rot in range(4):
            rotated = np.roll(det_corners, rot, axis=0)
            err = np.sqrt(np.mean(np.sum((rotated - gt_corners) ** 2, axis=1)))
            min_err = min(min_err, err)
        # Try flipped
        flipped = det_corners[::-1]
        for rot in range(4):
            rotated = np.roll(flipped, rot, axis=0)
            err = np.sqrt(np.mean(np.sum((rotated - gt_corners) ** 2, axis=1)))
            min_err = min(min_err, err)
        return min_err

    @staticmethod
    def match_detections(
        detections: list[dict[str, Any]], gt_tags: list[TagGroundTruth], threshold: float = 20.0
    ) -> tuple[int, float, int]:
        correct = 0
        err_sum = 0.0
        matched_gt = set()

        for det in detections:
            det_center = np.array(det["center"])
            best_gt_idx = -1
            min_dist = float("inf")

            for idx, gt in enumerate(gt_tags):
                if idx in matched_gt or gt.tag_id != det["id"]:
                    continue
                gt_center = np.mean(gt.corners, axis=0)
                dist = np.linalg.norm(det_center - gt_center)
                if dist < min_dist:
                    min_dist = dist
                    best_gt_idx = idx

            if best_gt_idx != -1 and min_dist < threshold:
                matched_gt.add(best_gt_idx)
                correct += 1
                err_sum += Metrics.compute_corner_error(
                    np.array(det["corners"]), gt_tags[best_gt_idx].corners
                )

        return correct, err_sum, len(matched_gt)


class LibraryWrapper:
    def __init__(self, name: str):
        self.name = name

    def detect(self, img: np.ndarray) -> tuple[list[dict[str, Any]], Any]:
        raise NotImplementedError


class LocusWrapper(LibraryWrapper):
    def __init__(self, config: Optional[locus.DetectorConfig] = None):
        super().__init__("Locus")
        self.detector = locus.Detector(config) if config else locus.Detector()

    def detect(self, img: np.ndarray) -> tuple[list[dict[str, Any]], Any]:
        detections, stats = self.detector.detect_with_stats(img)
        serializable = []
        for d in detections:
            serializable.append(
                {
                    "id": d.id,
                    "center": d.center,
                    "corners": list(d.corners),
                    "hamming": d.hamming,
                    "margin": d.decision_margin,
                }
            )
        return serializable, stats


class OpenCVWrapper(LibraryWrapper):
    def __init__(self):
        super().__init__("OpenCV")
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def detect(self, img: np.ndarray) -> tuple[list[dict[str, Any]], Any]:
        corners, ids, _ = self.detector.detectMarkers(img)
        detections = []
        if ids is not None:
            for i, tid in enumerate(ids):
                c = corners[i][0]
                detections.append(
                    {
                        "id": int(tid[0]),
                        "center": np.mean(c, axis=0).tolist(),
                        "corners": c.tolist(),
                        "hamming": 0,
                        "margin": 0,
                    }
                )
        return detections, None


class AprilTagWrapper(LibraryWrapper):
    def __init__(self, nthreads: int = 1):
        super().__init__("AprilTag")
        self.detector = AprilTagDetector(families="tag36h11", nthreads=nthreads, quad_decimate=1.0)

    def detect(self, img: np.ndarray) -> tuple[list[dict[str, Any]], Any]:
        raw_dets = self.detector.detect(img)
        detections = []
        for d in raw_dets:
            detections.append(
                {
                    "id": d.tag_id,
                    "center": d.center.tolist(),
                    "corners": d.corners.tolist(),
                    "hamming": d.hamming,
                    "margin": d.decision_margin,
                }
            )
        return detections, None


def generate_synthetic_image(
    num_tags: int, res: tuple[int, int], noise_sigma: float = 0.0
) -> tuple[np.ndarray, list[TagGroundTruth]]:
    img = np.zeros((res[1], res[0]), dtype=np.uint8) + 128
    cols = int(np.ceil(np.sqrt(num_tags)))
    rows = int(np.ceil(num_tags / cols))
    cell_w, cell_h = res[0] // cols, res[1] // rows
    tag_size = int(min(cell_w, cell_h) * 0.6)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    gt_data = []

    for i in range(num_tags):
        r, c = i // cols, i % cols
        x = c * cell_w + (cell_w - tag_size) // 2
        y = r * cell_h + (cell_h - tag_size) // 2

        tag_id = i % 587
        tag_img = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_size)

        padding = tag_size // 5
        img[y - padding : y + tag_size + padding, x - padding : x + tag_size + padding] = 255
        img[y : y + tag_size, x : x + tag_size] = tag_img

        ts = tag_size - 1
        corners = np.array([[x, y], [x + ts, y], [x + ts, y + ts], [x, y + ts]], dtype=np.float32)
        gt_data.append(TagGroundTruth(tag_id=tag_id, corners=corners))

    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img, gt_data
