"""Locus Hub Synchronization Utility.

Efficiently mirrors Hugging Face dataset subsets to local disk for Rust regression testing.
"""

import argparse
import concurrent.futures
import json
import logging
import os
from pathlib import Path
from typing import Final

import datasets
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from PIL import Image
from tqdm import tqdm

# Configure minimalist logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Resolve project-level defaults
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR: Final[Path] = Path(
    os.getenv("LOCUS_HUB_DATASET_DIR", PROJECT_ROOT / "tests/data/hub_cache")
)
DEFAULT_REPO_ID: Final[str] = "NoeFontana/locus-tag-bench"


def _save_image(img: Image.Image, path: Path) -> None:
    """Helper to save a PIL image to disk if it doesn't exist."""
    if not path.exists():
        img.save(path)


def _download_aux(repo_id: str, subset: str, aux_file: str, target_dir: Path) -> None:
    """Helper to download auxiliary files from HF Hub."""
    try:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"{subset}/{aux_file}",
            local_dir=str(target_dir),
        )
    except EntryNotFoundError:
        logger.debug(f"Auxiliary file {aux_file} not found for subset {subset}, skipping.")
    except Exception as e:
        logger.warning(f"    [!] Failed auxiliary download {aux_file}: {e}")


def sync_subset_to_local(subset: str, target_dir: Path, repo_id: str = DEFAULT_REPO_ID) -> None:
    """Synchronizes a single dataset subset (images + metadata) to local disk."""
    subset_dir: Path = target_dir / subset
    images_dir: Path = subset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--> Syncing: {subset}")

    # 1. Stream dataset and parallelize image saving
    try:
        ds = datasets.load_dataset(repo_id, subset, split="train", streaming=True)
    except Exception as e:
        logger.error(f"Failed to load dataset {subset} from {repo_id}: {e}")
        raise

    jsonl_path: Path = subset_dir / "annotations.jsonl"

    with (
        jsonl_path.open("w", encoding="utf-8") as f,
        concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 1) * 4)
        ) as executor,
    ):
        futures: list[concurrent.futures.Future[None]] = []
        for item in tqdm(ds, desc=f"    {subset} (Stream)", unit="img", leave=False):
            img: Image.Image = item.pop("image")
            image_id: str = (
                item.get("image_id")
                or f"img_{item.get('scene_id', 0)}_{item.get('camera_idx', 0)}_{item.get('tag_id', 0)}"
            )

            img_path: Path = images_dir / f"{image_id}.png"
            futures.append(executor.submit(_save_image, img, img_path))

            item["image_filename"] = img_path.name
            f.write(json.dumps(item) + "\n")

        # Wait for all images to be saved
        if futures:
            concurrent.futures.wait(futures)

    # 2. Download auxiliary schema files in parallel
    aux_files: list[str] = ["coco_labels.json", "rich_truth.json"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(aux_files)) as executor:
        for aux_file in aux_files:
            executor.submit(_download_aux, repo_id, subset, aux_file, target_dir)


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(description="Sync HF datasets to local cache.")
    parser.add_argument(
        "--configs", nargs="*", default=["all"], help="Subsets to sync (default: all)"
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Target directory (default: $LOCUS_HUB_DATASET_DIR or project tests/data/hub_cache)",
    )
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    args = parser.parse_args()

    # Discover configurations
    configs: list[str] = args.configs
    if "all" in configs:
        logger.info(f"Discovering configurations for {args.repo_id}...")
        try:
            configs = datasets.get_dataset_config_names(args.repo_id)
            logger.info(f"Found {len(configs)} configs: {', '.join(configs)}\n")
        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            return

    # Execute sync
    for config in configs:
        try:
            sync_subset_to_local(config, args.target_dir, args.repo_id)
        except Exception as e:
            logger.error(f"Failed to sync {config}: {e}")


if __name__ == "__main__":
    main()
