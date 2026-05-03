import csv
import json
import math
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import locus
import numpy as np
from huggingface_hub import hf_hub_download
from pupil_apriltags import Detector as AprilTagDetector
from tqdm import tqdm

ICRA_REPO_ID = "NoeFontana/apriltag-validation-data"
ICRA_CACHE_DIR = Path("tests/data/icra2020")
HUB_CACHE_DIR = Path("tests/data/hub_cache")


@dataclass
class HubDatasetResult:
    images_dir: Path
    gt_map: dict[str, Any]  # img_name -> {"tags": {id: tag_data}, "board_pose": np.ndarray | None}
    board_config_entry: dict[str, Any] | None
    intrinsics: locus.CameraIntrinsics | None
    tag_size: float | None  # in meters


def build_board_refiner(
    board_config_entry: dict[str, Any], tag_family: locus.TagFamily
) -> locus.BoardEstimator:
    """Build a BoardEstimator from a board config dict.

    Raises:
        ValueError: If the board type is unrecognised.
    """
    bt = board_config_entry.get("type", "").lower()
    rows = board_config_entry["rows"]
    cols = board_config_entry["cols"]
    sq_m = board_config_entry["square_size_mm"] / 1000.0
    mk_m = board_config_entry["marker_size_mm"] / 1000.0

    if "charuco" in bt:
        charuco_board = locus.CharucoBoard(rows, cols, sq_m, mk_m, tag_family)
        return locus.BoardEstimator.from_charuco(charuco_board)
    if "aprilgrid" in bt:
        spacing = sq_m - mk_m
        april_grid = locus.AprilGrid(rows, cols, spacing, mk_m, tag_family)
        return locus.BoardEstimator(april_grid)
    raise ValueError(f"Unknown board type: {bt!r}")


class HubDatasetLoader:
    def __init__(self, root: Path = HUB_CACHE_DIR):
        self.root = root

    def load_dataset(self, config_name: str) -> HubDatasetResult:
        hub_dir = self.root / config_name
        rich_path = hub_dir / "rich_truth.json"
        images_dir = hub_dir / "images"

        if not rich_path.exists():
            raise FileNotFoundError(f"Metadata not found at {rich_path}")

        with open(rich_path) as f:
            data = json.load(f)
        # rich_truth.json has two on-disk shapes: v1 = bare list, v2 = {records: [...]}.
        entries = data["records"] if isinstance(data, dict) and "records" in data else data

        gt_map: dict[str, Any] = {}
        board_config_entry = None
        intrinsics = None
        tag_size_mm = None

        for entry in entries:
            img_name = entry.get("image_filename") or entry.get("image_id")
            if img_name is None:
                continue
            if not img_name.endswith(".png"):
                img_name = f"{img_name}.png"

            if intrinsics is None and "k_matrix" in entry and len(entry["k_matrix"]) >= 2:
                k = entry["k_matrix"]
                intrinsics = locus.CameraIntrinsics(fx=k[0][0], fy=k[1][1], cx=k[0][2], cy=k[1][2])

            if tag_size_mm is None and "tag_size_mm" in entry:
                tag_size_mm = entry["tag_size_mm"]

            if img_name not in gt_map:
                gt_map[img_name] = {"tags": {}, "board_pose": None}

            record_type = entry.get("record_type")
            if record_type == "BOARD":
                if board_config_entry is None:
                    board_config_entry = entry.get("board_definition")

                pos = entry["position"]
                quat = entry["rotation_quaternion"]  # [w, x, y, z]
                gt_map[img_name]["board_pose"] = np.array(
                    [pos[0], pos[1], pos[2], quat[1], quat[2], quat[3], quat[0]], dtype=np.float64
                )

            elif record_type == "TAG":
                tid = int(entry["tag_id"])
                tag_data: dict[str, Any] = {"corners": np.array(entry["corners"], dtype=np.float32)}
                if "position" in entry and "rotation_quaternion" in entry:
                    pos = entry["position"]
                    quat = entry["rotation_quaternion"]  # [w, x, y, z]
                    tag_data["pose"] = np.array(
                        [pos[0], pos[1], pos[2], quat[1], quat[2], quat[3], quat[0]],
                        dtype=np.float64,
                    )
                gt_map[img_name]["tags"][tid] = tag_data

        tag_size = tag_size_mm / 1000.0 if tag_size_mm else None
        return HubDatasetResult(
            images_dir=images_dir,
            gt_map=gt_map,
            board_config_entry=board_config_entry,
            intrinsics=intrinsics,
            tag_size=tag_size,
        )


class FamilyMapper:
    """Centralized mapping for tag families across different libraries."""

    @staticmethod
    def to_locus(family: int | locus.TagFamily | None) -> list[locus.TagFamily] | None:
        if family is None:
            return None
        mapping = {
            int(locus.TagFamily.AprilTag16h5): locus.TagFamily.AprilTag16h5,
            int(locus.TagFamily.AprilTag36h11): locus.TagFamily.AprilTag36h11,
            int(locus.TagFamily.ArUco4x4_50): locus.TagFamily.ArUco4x4_50,
            int(locus.TagFamily.ArUco4x4_100): locus.TagFamily.ArUco4x4_100,
            int(locus.TagFamily.ArUco6x6_250): locus.TagFamily.ArUco6x6_250,
        }
        f = mapping.get(int(family))
        return [f] if f else None

    @staticmethod
    def to_opencv(family: int | locus.TagFamily | None) -> int | None:
        if family is None:
            return None
        mapping = {
            int(locus.TagFamily.AprilTag16h5): cv2.aruco.DICT_APRILTAG_16h5,
            int(locus.TagFamily.AprilTag36h11): cv2.aruco.DICT_APRILTAG_36h11,
            int(locus.TagFamily.ArUco4x4_50): cv2.aruco.DICT_4X4_50,
            int(locus.TagFamily.ArUco4x4_100): cv2.aruco.DICT_4X4_100,
            int(locus.TagFamily.ArUco6x6_250): cv2.aruco.DICT_6X6_250,
        }
        return mapping.get(int(family))

    @staticmethod
    def to_apriltag(family: int | locus.TagFamily | None) -> str | None:
        if family is None:
            return None
        mapping = {
            int(locus.TagFamily.AprilTag16h5): "tag16h5",
            int(locus.TagFamily.AprilTag36h11): "tag36h11",
            int(locus.TagFamily.ArUco4x4_50): None,
            int(locus.TagFamily.ArUco4x4_100): None,
            int(locus.TagFamily.ArUco6x6_250): None,
        }
        return mapping.get(int(family))


@dataclass
class TagGroundTruth:
    tag_id: int
    corners: np.ndarray  # 4x2 float32
    fully_visible: bool = True
    # 6-DOF pose if available: [tx, ty, tz, qx, qy, qz, qw] (Scalar-Last)
    pose: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class TagAxes:
    """Raw stratification-axis values for one ground-truth tag.

    Sibling to :class:`TagGroundTruth` rather than an extension —
    ``Metrics.match_detections`` keys on ``TagGroundTruth`` and shouldn't
    widen. ``DatasetLoader.load_axes`` returns these alongside ``gt_map``.

    NaN means "axis not derivable from the source dataset" (e.g. ICRA has no
    per-tag distance metadata).
    """

    tag_id: int
    distance_m: float
    aoi_deg: float
    ppm: float  # derived: max_edge_px(corners) / (tag_size_mm / 1000)
    velocity: float | None  # `None` (not NaN) means "not provided" → static
    shutter_time_ms: float
    resolution_h: int
    occlusion_ratio: float
    tag_size_mm: float


@dataclass
class DatasetMetadata:
    intrinsics: locus.CameraIntrinsics | None = None
    tag_size: float | None = None


def max_edge_px(corners: np.ndarray) -> float:
    """Longest of the four polygon edges in pixels. Used to derive PPM."""
    rolled = np.roll(corners, -1, axis=0)
    return float(np.max(np.linalg.norm(rolled - corners, axis=1)))


@dataclass(frozen=True, slots=True)
class RejectedQuads:
    """Per-frame Locus quads that were rejected before successful decode.

    Returned as the second element of :meth:`LocusWrapper.detect`. ``None``
    for non-Locus wrappers since they don't expose intermediate rejections.

    All three arrays share axis-0 length M.
    """

    corners: np.ndarray  # (M, 4, 2) float32
    funnel_status: np.ndarray  # (M,) uint8 — see locus.FunnelStatus
    error_rates: np.ndarray  # (M,) float32

    @classmethod
    def from_batch(cls, batch: "locus.DetectionBatch") -> "RejectedQuads | None":
        if batch.rejected_corners is None or batch.rejected_funnel_status is None:
            return None
        return cls(
            corners=batch.rejected_corners,
            funnel_status=batch.rejected_funnel_status,
            error_rates=batch.rejected_error_rates
            if batch.rejected_error_rates is not None
            else np.zeros(len(batch.rejected_corners), dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.corners)


def serializable_from_batch(batch: "locus.DetectionBatch") -> list[dict[str, Any]]:
    """Convert a Locus :class:`DetectionBatch` to the wrapper-uniform dict list.

    Shared between :meth:`LocusWrapper.detect` (ICRA flow) and the Hub flow at
    ``tools/cli.py`` which calls ``wrapper.detector.detect()`` directly so it
    can pass ``pose_estimation_mode=Accurate``.
    """
    out: list[dict[str, Any]] = []
    for i in range(len(batch)):
        corners = batch.corners[i]
        det: dict[str, Any] = {
            "id": int(batch.ids[i]),
            "center": np.mean(corners, axis=0).tolist(),
            "corners": corners.tolist(),
            "hamming": int(batch.error_rates[i]),
            "margin": 0.0,
        }
        if batch.poses is not None:
            det["pose"] = batch.poses[i].tolist()
        out.append(det)
    return out


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
            pose = np.array([pos[0], pos[1], pos[2], x, y, z, w], dtype=np.float64)

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

    def load_axes(self, scenario: str) -> dict[str, dict[int, TagAxes]]:
        """Return per-image, per-tag stratification axes for a scenario.

        Lookup shape is ``axes[image_filename][tag_id]`` so callers can join by
        ``tag_id`` rather than by list index — robust against any reordering
        in the underlying CSV / JSON parser.

        For Hub corpora (``rich_truth.json``), every axis is populated from the
        manifest (``ppm`` is derived from corners + ``tag_size_mm``). For ICRA
        2020 (``tags.csv``), only ``resolution_h`` is recoverable; the rest are
        ``NaN`` per :doc:`stratification` §3.
        """
        # 1. Hub layout (preferred when available)
        hub_dir = self.icra_dir / scenario
        if not hub_dir.exists():
            hub_dir = HUB_CACHE_DIR / scenario
        rich_path = hub_dir / "rich_truth.json"
        if rich_path.exists():
            return self._load_hub_axes(rich_path)

        # 2. ICRA fallback — populate `resolution_h` from image header at most;
        # other fields collapse to NaN. Not implemented in v1 of this method
        # because every ICRA stratum is `unk` already and the bench harness
        # doesn't need per-tag axes there.
        return {}

    def _load_hub_axes(self, rich_path: Path) -> dict[str, dict[int, TagAxes]]:
        with open(rich_path) as f:
            data = json.load(f)
        entries = data["records"] if isinstance(data, dict) and "records" in data else data

        out: dict[str, dict[int, TagAxes]] = {}
        for e in entries:
            if e.get("record_type", "TAG") != "TAG":
                continue
            img_name = e.get("image_filename") or e.get("image_id")
            if img_name is None:
                continue
            if not img_name.endswith(".png"):
                img_name = f"{img_name}.png"

            tag_size_mm = float(e["tag_size_mm"])
            corners = np.asarray(e["corners"], dtype=np.float64)
            ppm_raw = float(e.get("ppm", 0.0))
            # `ppm == 0.0` is the manifest sentinel for "not provided" —
            # derive from corners per stratification.md §3.
            ppm = ppm_raw if ppm_raw > 0.0 else max_edge_px(corners) / (tag_size_mm / 1000.0)

            axes = TagAxes(
                tag_id=int(e["tag_id"]),
                distance_m=float(e["distance"]),
                aoi_deg=float(e["angle_of_incidence"]),
                ppm=ppm,
                velocity=e.get("velocity"),
                shutter_time_ms=float(e.get("shutter_time_ms", math.nan)),
                resolution_h=int(e["resolution"][1]),
                occlusion_ratio=float(e.get("occlusion_ratio", 0.0)),
                tag_size_mm=tag_size_mm,
            )
            out.setdefault(img_name, {})[axes.tag_id] = axes
        return out

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


def _pose_from_R_t(R: np.ndarray, t: np.ndarray) -> list[float]:
    """Pack a rotation matrix + translation into [tx, ty, tz, qx, qy, qz, qw]."""
    qw = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) * 0.5
    if qw > 1e-8:
        qx = (R[2, 1] - R[1, 2]) / (4.0 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4.0 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4.0 * qw)
    else:
        # Fallback for near-180° rotations: pick the largest diagonal.
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            qx = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 0.5
            qy = (R[0, 1] + R[1, 0]) / (4.0 * qx)
            qz = (R[0, 2] + R[2, 0]) / (4.0 * qx)
            qw = (R[2, 1] - R[1, 2]) / (4.0 * qx)
        elif i == 1:
            qy = math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) * 0.5
            qx = (R[0, 1] + R[1, 0]) / (4.0 * qy)
            qz = (R[1, 2] + R[2, 1]) / (4.0 * qy)
            qw = (R[0, 2] - R[2, 0]) / (4.0 * qy)
        else:
            qz = math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) * 0.5
            qx = (R[0, 2] + R[2, 0]) / (4.0 * qz)
            qy = (R[1, 2] + R[2, 1]) / (4.0 * qz)
            qw = (R[1, 0] - R[0, 1]) / (4.0 * qz)
    return [float(t[0]), float(t[1]), float(t[2]), float(qx), float(qy), float(qz), float(qw)]


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def rotation_error_deg(det_pose: np.ndarray, gt_quat_xyzw: np.ndarray) -> float:
    """Geodesic angle (degrees) between detected and GT rotations."""
    R_det = _quat_to_rot(*[float(v) for v in det_pose[3:7]])
    qx, qy, qz, qw = (float(v) for v in gt_quat_xyzw)
    R_gt = _quat_to_rot(qx, qy, qz, qw)
    cos_theta = (np.trace(R_det.T @ R_gt) - 1.0) * 0.5
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cos_theta)))))


def accumulate_pose_match(
    stats: dict[str, Any],
    matched: set[int],
    tid: int,
    gt_tags: dict[int, Any],
    pose: Any,
) -> None:
    """Mark a detection matched (id ∈ GT, first hit wins) and record pose error."""
    if tid not in gt_tags or tid in matched:
        return
    matched.add(tid)
    stats["det"] += 1
    gt = gt_tags[tid]
    if "pose" in gt and pose is not None:
        det_pose = np.asarray(pose)
        stats["trans_errs"].append(
            float(np.linalg.norm(np.asarray(det_pose[:3]) - np.asarray(gt["pose"][:3])))
        )
        stats["rot_errs"].append(rotation_error_deg(det_pose, gt["pose"][3:7]))


def new_pose_stats() -> dict[str, Any]:
    """Empty stats dict for `accumulate_pose_match` / `aggregate_pose_stats`."""
    return {
        "gt": 0,
        "det": 0,
        "total_det": 0,
        "latency": [],
        "trans_errs": [],
        "rot_errs": [],
    }


def evaluate_tag_pose(
    wrapper: "LibraryWrapper",
    ds: "HubDatasetResult",
    eval_tag_size: float,
    *,
    pose_estimation_mode: Any = None,
) -> dict[str, Any]:
    """Run a wrapper across a Hub tag-only dataset, returning raw per-detection stats.

    Locus wrappers are driven through the underlying detector to thread the
    optional `pose_estimation_mode`; non-Locus wrappers go through `detect()`.
    """
    is_locus = isinstance(wrapper, LocusWrapper)
    stats = new_pose_stats()

    for img_name in tqdm(sorted(ds.gt_map.keys()), desc=f"{wrapper.name:<24}"):
        img_path = ds.images_dir / img_name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        start = time.perf_counter()
        if is_locus:
            assert ds.intrinsics is not None
            kwargs: dict[str, Any] = {"intrinsics": ds.intrinsics, "tag_size": eval_tag_size}
            if pose_estimation_mode is not None:
                kwargs["pose_estimation_mode"] = pose_estimation_mode
            batch = wrapper.detector.detect(img, **kwargs)
            detections = None
        else:
            detections, _ = wrapper.detect(img, intrinsics=ds.intrinsics, tag_size=eval_tag_size)
            batch = None
        stats["latency"].append((time.perf_counter() - start) * 1000.0)

        gt_tags = ds.gt_map[img_name]["tags"]
        stats["gt"] += len(gt_tags)
        matched: set[int] = set()

        if batch is not None:
            stats["total_det"] += len(batch.ids)
            batch_poses = getattr(batch, "poses", None)
            for i in range(len(batch.ids)):
                pose = batch_poses[i] if batch_poses is not None else None
                accumulate_pose_match(stats, matched, int(batch.ids[i]), gt_tags, pose)
        elif detections is not None:
            stats["total_det"] += len(detections)
            for det in detections:
                accumulate_pose_match(stats, matched, int(det["id"]), gt_tags, det.get("pose"))

    return stats


def aggregate_pose_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Reduce raw stats from `evaluate_tag_pose` to recall/precision/latency + percentiles."""
    trans = np.asarray(stats["trans_errs"], dtype=np.float64)
    rot = np.asarray(stats["rot_errs"], dtype=np.float64)
    out: dict[str, Any] = {
        "recall": (stats["det"] / stats["gt"] * 100) if stats["gt"] else 0.0,
        "precision": (stats["det"] / stats["total_det"] * 100) if stats["total_det"] else 0.0,
        "latency_ms": float(np.mean(stats["latency"])) if stats["latency"] else 0.0,
        "samples": int(trans.size),
    }
    if trans.size:
        out["trans_mean_m"] = float(np.mean(trans))
        out["trans_p50_m"], out["trans_p95_m"], out["trans_p99_m"] = (
            float(v) for v in np.percentile(trans, [50, 95, 99])
        )
    else:
        out["trans_mean_m"] = out["trans_p50_m"] = out["trans_p95_m"] = out["trans_p99_m"] = 0.0
    if rot.size:
        out["rot_mean_deg"] = float(np.mean(rot))
        out["rot_p50_deg"], out["rot_p95_deg"], out["rot_p99_deg"] = (
            float(v) for v in np.percentile(rot, [50, 95, 99])
        )
    else:
        out["rot_mean_deg"] = out["rot_p50_deg"] = out["rot_p95_deg"] = out["rot_p99_deg"] = 0.0
    return out


class Metrics:
    @staticmethod
    def align_pose(gt_pos: np.ndarray, gt_quat: np.ndarray, tag_size: float) -> np.ndarray:
        """Aligns a center-origin ground truth pose to a top-left origin pose.

        Args:
            gt_pos: [x, y, z] position in camera frame.
            gt_quat: [x, y, z, w] quaternion.
            tag_size: Physical side length of the tag.

        Returns:
            shifted_pos: [x, y, z] position of the top-left corner in camera frame.
        """
        r_gt = _quat_to_rot(*[float(v) for v in gt_quat])
        s_half = tag_size * 0.5
        return gt_pos + r_gt @ np.array([-s_half, -s_half, 0.0])

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
        decimation: int = 1,
        family: int | None = None,
        detector: locus.Detector | None = None,
    ):
        super().__init__(name)

        if detector is not None:
            self.detector = detector
        else:
            families = FamilyMapper.to_locus(family)
            self.detector = locus.Detector(decimation=decimation, families=families)

    def detect(
        self,
        img: np.ndarray,
        intrinsics: locus.CameraIntrinsics | None = None,
        tag_size: float | None = None,
    ) -> tuple[list[dict[str, Any]], "RejectedQuads | None"]:
        batch = self.detector.detect(img, intrinsics=intrinsics, tag_size=tag_size)
        return serializable_from_batch(batch), RejectedQuads.from_batch(batch)


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
        if ids is None:
            return detections, None

        camera_matrix = None
        dist_coeffs = np.zeros(5, dtype=np.float64)
        obj_pts = None
        if intrinsics is not None and tag_size is not None:
            camera_matrix = np.array(
                [
                    [intrinsics.fx, 0.0, intrinsics.cx],
                    [0.0, intrinsics.fy, intrinsics.cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            # Center-origin y-down object points (matches Locus + Hub GT convention).
            # cv2.aruco corner order is TL,TR,BR,BL relative to the marker's canonical orientation.
            s = tag_size * 0.5
            obj_pts = np.array(
                [[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]],
                dtype=np.float64,
            )

        for i, tid in enumerate(ids):
            c = corners[i][0]
            det = {
                "id": int(tid[0]),
                "center": np.mean(c, axis=0).tolist(),
                "corners": c.tolist(),
                "hamming": 0,
                "margin": 0,
            }
            if camera_matrix is not None and obj_pts is not None:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    c.astype(np.float64),
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok:
                    R, _ = cv2.Rodrigues(rvec)
                    det["pose"] = _pose_from_R_t(R, tvec.flatten())
            detections.append(det)
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
        camera_params = (
            (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
            if intrinsics is not None
            else None
        )
        estimate_pose = camera_params is not None and tag_size is not None
        raw_dets = self.detector.detect(
            img,
            estimate_tag_pose=estimate_pose,
            camera_params=camera_params,
            tag_size=tag_size,
        )
        detections = []
        for d in raw_dets:  # pyright: ignore[reportGeneralTypeIssues]  # pupil_apriltags stub types Detector.detect as Detection (single), runtime returns list
            det = {
                "id": d.tag_id,
                "center": d.center.tolist(),
                "corners": d.corners.tolist(),
                "hamming": d.hamming,
                "margin": d.decision_margin,
            }
            pose_t = getattr(d, "pose_t", None)
            pose_R = getattr(d, "pose_R", None)
            if pose_t is not None and pose_R is not None:
                # AprilTag-C's tag frame differs from Locus/GT by 180° about z.
                R_corrected = np.asarray(pose_R) @ np.diag([-1.0, -1.0, 1.0])
                det["pose"] = _pose_from_R_t(R_corrected, np.asarray(pose_t).flatten())
            detections.append(det)
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
