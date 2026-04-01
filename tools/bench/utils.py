import csv
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import locus
import numpy as np
from huggingface_hub import hf_hub_download
from pupil_apriltags import Detector as AprilTagDetector  # type: ignore

ICRA_REPO_ID = "NoeFontana/apriltag-validation-data"
ICRA_CACHE_DIR = Path("tests/data/icra2020")
HUB_CACHE_DIR = Path("tests/data/hub_cache")


class FamilyMapper:
    """Centralized mapping for tag families across different libraries."""

    @staticmethod
    def to_locus(family: int | None) -> list[locus.TagFamily] | None:
        if family is None:
            return None
        mapping = {
            int(locus.TagFamily.AprilTag16h5): locus.TagFamily.AprilTag16h5,
            int(locus.TagFamily.AprilTag36h11): locus.TagFamily.AprilTag36h11,
            int(locus.TagFamily.ArUco4x4_50): locus.TagFamily.ArUco4x4_50,
            int(locus.TagFamily.ArUco4x4_100): locus.TagFamily.ArUco4x4_100,
            int(locus.TagFamily.ArUco6x6_250): locus.TagFamily.ArUco6x6_250,
        }
        f = mapping.get(family)
        return [f] if f else None

    @staticmethod
    def to_opencv(family: int | None) -> int | None:
        if family is None:
            return None
        mapping = {
            int(locus.TagFamily.AprilTag16h5): cv2.aruco.DICT_APRILTAG_16h5,
            int(locus.TagFamily.AprilTag36h11): cv2.aruco.DICT_APRILTAG_36h11,
            int(locus.TagFamily.ArUco4x4_50): cv2.aruco.DICT_4X4_50,
            int(locus.TagFamily.ArUco4x4_100): cv2.aruco.DICT_4X4_100,
            int(locus.TagFamily.ArUco6x6_250): cv2.aruco.DICT_6X6_250,
        }
        return mapping.get(family)

    @staticmethod
    def to_apriltag(family: int | None) -> str | None:
        if family is None:
            return None
        mapping = {
            int(locus.TagFamily.AprilTag16h5): "tag16h5",
            int(locus.TagFamily.AprilTag36h11): "tag36h11",
            int(locus.TagFamily.ArUco4x4_50): None,
            int(locus.TagFamily.ArUco4x4_100): None,
            int(locus.TagFamily.ArUco6x6_250): None,
        }
        return mapping.get(family)


@dataclass
class TagGroundTruth:
    tag_id: int
    corners: np.ndarray  # 4x2 float32
    fully_visible: bool = True
    # 6-DOF pose if available: [tx, ty, tz, qx, qy, qz, qw] (Scalar-Last)
    pose: np.ndarray | None = None


@dataclass
class DatasetMetadata:
    intrinsics: locus.CameraIntrinsics | None = None
    tag_size: float | None = None


class DatasetLoader:
    def __init__(self, icra_dir: Path = ICRA_CACHE_DIR):
        self.icra_dir = icra_dir

    def prepare_all(self):
        """Prepare ICRA datasets."""
        self.prepare_icra("forward")
        self.prepare_icra("circle")

    def prepare_icra(self, scenario: str) -> bool:
        self.icra_dir.mkdir(parents=True, exist_ok=True)
        scenario_dir = self.icra_dir / scenario
        if scenario_dir.exists() and any(scenario_dir.iterdir()):
            return True

        filename = f"{scenario}.tar.xz"
        try:
            archive_path = str(
                hf_hub_download(
                    repo_id=ICRA_REPO_ID,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=str(self.icra_dir),
                )
            )
            archive_path_obj = Path(archive_path)
            with tarfile.open(archive_path_obj, "r:xz") as tar:
                tar.extractall(path=self.icra_dir)
            archive_path_obj.unlink()
            return True
        except Exception as e:
            print(f"Error preparing ICRA scenario {scenario}: {e}")
            return False

    def find_datasets(
        self, scenario: str, filter_types: list[str]
    ) -> list[tuple[str, Path, dict[str, list[TagGroundTruth]], DatasetMetadata]]:
        # 1. Check for Hub (render-tag) dataset first (rich_truth.json)
        hub_dir = self.icra_dir / scenario
        if not hub_dir.exists():
            hub_dir = HUB_CACHE_DIR / scenario

        if hub_dir.exists():
            rich_truth = hub_dir / "rich_truth.json"
            if rich_truth.exists():
                return self._load_hub_dataset(hub_dir, scenario)

        # 2. Fallback to ICRA (CSV) dataset
        search_dir = self.icra_dir / scenario
        if not search_dir.exists():
            search_dir = self.icra_dir

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

                    # Estimated ICRA defaults if not present
                    meta = DatasetMetadata(
                        intrinsics=locus.CameraIntrinsics(fx=736.6, fy=736.6, cx=960.0, cy=540.0),
                        tag_size=0.16,
                    )
                    results.append((name, subdir, gt_data, meta))
        return results

    def _load_hub_dataset(
        self, hub_dir: Path, scenario: str
    ) -> list[tuple[str, Path, dict[str, list[TagGroundTruth]], DatasetMetadata]]:
        """Loads Hub/render-tag datasets (rich_truth.json)."""
        rich_path = hub_dir / "rich_truth.json"
        prov_path = hub_dir / "provenance.json"
        images_dir = hub_dir / "images"

        with open(rich_path) as f:
            entries = json.load(f)

        gt_map: dict[str, list[TagGroundTruth]] = {}
        for entry in entries:
            img_name = entry.get("image_filename") or entry.get("image_id")
            if img_name is None:
                continue

            if not img_name.endswith(".png"):
                img_name = f"{img_name}.png"

            # Map corners
            corners = np.array(entry["corners"], dtype=np.float32)

            # Map pose [tx, ty, tz, qx, qy, qz, qw]
            # Hub quaternion is [w, x, y, z]
            w, x, y, z = entry["rotation_quaternion"]
            pos = entry["position"]
            pose = np.array([pos[0], pos[1], pos[2], x, y, z, w], dtype=np.float32)

            gt_tags = gt_map.setdefault(img_name, [])
            gt_tags.append(TagGroundTruth(tag_id=int(entry["tag_id"]), corners=corners, pose=pose))

        # Load metadata
        meta = DatasetMetadata()
        if prov_path.exists():
            with open(prov_path) as f:
                prov = json.load(f)

            if "camera_intrinsics" in prov:
                k = prov["camera_intrinsics"]
                meta.intrinsics = locus.CameraIntrinsics(
                    fx=k["fx"], fy=k["fy"], cx=k["cx"], cy=k["cy"]
                )

            if "tag_size_mm" in prov:
                meta.tag_size = prov["tag_size_mm"] / 1000.0

        # Fallback to per-detection metadata if global missing
        if entries and (meta.intrinsics is None or meta.tag_size is None):
            first = entries[0]
            if meta.intrinsics is None and "k_matrix" in first:
                k = first["k_matrix"]
                meta.intrinsics = locus.CameraIntrinsics(
                    fx=k[0][0], fy=k[1][1], cx=k[0][2], cy=k[1][2]
                )
            if meta.tag_size is None and "tag_size_mm" in first:
                meta.tag_size = first["tag_size_mm"] / 1000.0

        return [(scenario, images_dir, gt_map, meta)]

    def _parse_csv(self, csv_file: Path) -> dict[str, list[TagGroundTruth]]:
        """Parses ICRA 2020 ground truth CSV files."""
        # Use primitive types for interim accumulation for performance
        gt_data: dict[str, dict[int, dict[str, Any]]] = {}
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row["image"]
                tid = int(row["tag_id"])

                img_entry = gt_data.setdefault(img, {})
                tag_entry = img_entry.setdefault(tid, {"corners": [None] * 4, "visible": True})

                corner_idx = int(row["corner"])
                if 0 <= corner_idx < 4:
                    tag_entry["corners"][corner_idx] = [
                        float(row["ground_truth_x"]),
                        float(row["ground_truth_y"]),
                    ]

                if int(row.get("tag_fully_visible", 1)) == 0:
                    tag_entry["visible"] = False

        final_map = {}
        for img, tags in gt_data.items():
            valid = []
            for tid, data in tags.items():
                if data["visible"] and all(c is not None for c in data["corners"]):
                    valid.append(
                        TagGroundTruth(
                            tag_id=tid, corners=np.array(data["corners"], dtype=np.float32)
                        )
                    )
            if valid:
                final_map[img] = valid
        return final_map


class HubBenchmarkLoader:
    def __init__(self, repo_id: str = "NoeFontana/locus-tag-bench"):
        self.repo_id = repo_id

    def stream_subset(self, subset_name: str) -> Any:
        """Streams a dataset subset from the Hugging Face Hub.

        Args:
            subset_name: The name of the dataset configuration/subset to load.

        Yields:
            A tuple of (image_name, grayscale_image_np, list_of_ground_truth_tags).
        """
        import datasets

        # Stream the dataset from the hub
        ds = datasets.load_dataset(self.repo_id, subset_name, split="train", streaming=True)

        for idx, item in enumerate(ds):
            # PIL image to grayscale numpy array
            pil_img = item["image"]
            if pil_img.mode != "L":
                pil_img = pil_img.convert("L")
            img_np = np.array(pil_img, dtype=np.uint8)

            # Use native image_id if available, fallback to index
            image_id = item.get("image_id", f"{subset_name}/{idx}")

            # Map tag ground truth
            # Some datasets might have lists for tag_id and corners if multiple per image
            tag_ids = item["tag_id"]
            corners_list = item["corners"]

            if not isinstance(tag_ids, list):
                tag_ids = [tag_ids]
                corners_list = [corners_list]

            gt_tags = []
            for tid, corners in zip(tag_ids, corners_list, strict=True):
                gt_tags.append(
                    TagGroundTruth(tag_id=int(tid), corners=np.array(corners, dtype=np.float32))
                )

            yield image_id, img_np, gt_tags


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
                    min_dist = float(dist)
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

    def detect(
        self,
        img: np.ndarray,
        intrinsics: locus.CameraIntrinsics | None = None,
        tag_size: float | None = None,
    ) -> tuple[list[dict[str, Any]], Any]:
        """Detect tags in an image.
        Returns:
            detections: List of dicts with keys 'id', 'center', 'corners', 'hamming', 'margin'
            stats: Library specific statistics (optional)
        """
        raise NotImplementedError


class LocusWrapper(LibraryWrapper):
    def __init__(
        self,
        name: str = "Locus",
        config: locus.DetectorConfig | None = None,
        decimation: int = 1,
        family: int | None = None,
        detector: locus.Detector | None = None,
    ):
        super().__init__(name)

        if detector:
            self.detector = detector
            return

        families = FamilyMapper.to_locus(family)

        if config:
            self.detector = locus.Detector(
                decimation=decimation,
                families=families,
                decode_mode=config.decode_mode,
                enable_sharpening=config.enable_sharpening,
                upscale_factor=config.upscale_factor,
                refinement_mode=config.refinement_mode,
                threshold_tile_size=config.threshold_tile_size,
                threshold_min_range=config.threshold_min_range,
                adaptive_threshold_constant=config.adaptive_threshold_constant,
                quad_min_area=config.quad_min_area,
                quad_min_fill_ratio=config.quad_min_fill_ratio,
                quad_min_edge_score=config.quad_min_edge_score,
                quad_max_elongation=config.quad_max_elongation,
                quad_min_density=config.quad_min_density,
                quad_extraction_mode=int(config.quad_extraction_mode),
                decoder_min_contrast=config.decoder_min_contrast,
                max_hamming_error=config.max_hamming_error,
            )
        else:
            self.detector = locus.Detector(decimation=decimation, families=families)

    def detect(
        self,
        img: np.ndarray,
        intrinsics: locus.CameraIntrinsics | None = None,
        tag_size: float | None = None,
    ) -> tuple[list[dict[str, Any]], Any]:
        batch = self.detector.detect(img, intrinsics=intrinsics, tag_size=tag_size)

        serializable = []
        for i in range(len(batch)):
            corners = batch.corners[i]
            center = np.mean(corners, axis=0).tolist()
            serializable.append(
                {
                    "id": int(batch.ids[i]),
                    "center": center,
                    "corners": corners.tolist(),
                    "hamming": int(batch.error_rates[i]),
                    "margin": 0.0,  # Not currently exposed
                }
            )
        return serializable, None


class OpenCVWrapper(LibraryWrapper):
    def __init__(self, family: int | None = None):
        super().__init__("OpenCV")
        self.detector: cv2.aruco.ArucoDetector | None = None
        cv_family = FamilyMapper.to_opencv(family)

        if cv_family is not None:
            dictionary = cv2.aruco.getPredefinedDictionary(cv_family)
            parameters = cv2.aruco.DetectorParameters()
            parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            parameters.minMarkerPerimeterRate = 0.005
            parameters.adaptiveThreshConstant = 3
            parameters.adaptiveThreshWinSizeStep = 5
            parameters.minMarkerDistanceRate = 0.01
            parameters.minDistanceToBorder = 1
            parameters.polygonalApproxAccuracyRate = 0.01
            self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        else:
            self.detector = None

    def detect(
        self,
        img: np.ndarray,
        intrinsics: locus.CameraIntrinsics | None = None,
        tag_size: float | None = None,
    ) -> tuple[list[dict[str, Any]], Any]:
        if self.detector is None:
            return [], None
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
    def __init__(self, nthreads: int = 8, quad_decimate: float = 1.0, family: int | None = None):
        super().__init__("AprilTag")
        at_family = FamilyMapper.to_apriltag(family)

        if at_family is not None:
            self.detector = AprilTagDetector(
                families=at_family,
                nthreads=nthreads,
                quad_decimate=quad_decimate,
                quad_sigma=0.0,
                decode_sharpening=0.25,
                refine_edges=True,
            )
        else:
            self.detector = None

    def detect(
        self,
        img: np.ndarray,
        intrinsics: locus.CameraIntrinsics | None = None,
        tag_size: float | None = None,
    ) -> tuple[list[dict[str, Any]], Any]:
        if self.detector is None:
            return [], None
        raw_dets = self.detector.detect(img)
        detections = []
        for d in raw_dets:  # type: ignore[attr-defined]
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
    num_tags: int, res: tuple[int, int], noise_sigma: float = 0.0, family: int | None = None
) -> tuple[np.ndarray, list[TagGroundTruth]]:
    img = np.zeros((res[1], res[0]), dtype=np.uint8) + 128
    cols = int(np.ceil(np.sqrt(num_tags)))
    rows = int(np.ceil(num_tags / cols))
    cell_w, cell_h = res[0] // cols, res[1] // rows
    tag_size = int(min(cell_w, cell_h) * 0.6)

    cv_family = FamilyMapper.to_opencv(family)
    if cv_family is None:
        cv_family = cv2.aruco.DICT_APRILTAG_36h11

    dictionary = cv2.aruco.getPredefinedDictionary(cv_family)

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
