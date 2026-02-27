"""Locus Hub Synchronization Utility.

Efficiently mirrors Hugging Face dataset subsets to local disk for Rust regression testing.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import datasets
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from tqdm import tqdm

# Configure minimalist logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Resolve project-level defaults
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = Path(os.getenv("LOCUS_HUB_DATASET_DIR", PROJECT_ROOT / "tests/data/hub_cache"))
DEFAULT_REPO_ID = "NoeFontana/locus-tag-bench"

def sync_subset_to_local(subset: str, target_dir: Path, repo_id: str = DEFAULT_REPO_ID) -> None:
    """Synchronizes a single dataset subset (images + metadata) to local disk."""
    subset_dir = target_dir / subset
    images_dir = subset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--> Syncing: {subset}")

    # 1. Stream dataset and save images/annotations
    ds = datasets.load_dataset(repo_id, subset, split="train", streaming=True)
    jsonl_path = subset_dir / "annotations.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in tqdm(ds, desc=f"    {subset}", unit="img", leave=False):
            # Extract image and generate unique ID
            img = item.pop("image")
            image_id = item.get("image_id") or f"img_{item.get('scene_id', 0)}_{item.get('camera_idx', 0)}_{item.get('tag_id', 0)}"
            
            img_path = images_dir / f"{image_id}.png"
            if not img_path.exists():
                img.save(img_path)

            item["image_filename"] = img_path.name
            f.write(json.dumps(item) + "\n")

    # 2. Download auxiliary schema files
    for aux_file in ["coco_labels.json", "rich_truth.json"]:
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=f"{subset}/{aux_file}",
                local_dir=str(target_dir),
            )
        except EntryNotFoundError:
            continue
        except Exception as e:
            logger.warning(f"    [!] Failed aux download {aux_file}: {e}")


def main() -> None:
    """CLI Entrypoint."""
    parser = argparse.ArgumentParser(description="Sync HF datasets to local cache.")
    parser.add_argument("--configs", nargs="*", default=["all"], help="Subsets to sync (default: all)")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Target directory (default: $LOCUS_HUB_DATASET_DIR or project tests/data/hub_cache)")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    args = parser.parse_args()

    # Discover configurations
    configs: List[str] = args.configs
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
