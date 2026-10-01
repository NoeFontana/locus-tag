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
from huggingface_hub.errors import EntryNotFoundError
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR: Final[Path] = Path(
    os.getenv("LOCUS_HUB_DATASET_DIR", PROJECT_ROOT / "tests/data/hub_cache")
)
DEFAULT_REPO_ID: Final[str] = "NoeFontana/locus-tag-bench"


def _save_image(img: Image.Image, path: Path) -> None:
    """Helper to save a PIL image to disk if it doesn't exist."""
    if not path.exists():
        img.save(path)


def _download_aux(
    repo_id: str, subset: str, aux_file: str, target_dir: Path, revision: str | None = None
) -> None:
    """Download one auxiliary file; a file absent from the repo is not an error."""
    try:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"{subset}/{aux_file}",
            local_dir=str(target_dir),
            revision=revision,
        )
    except EntryNotFoundError:
        logger.debug(f"Auxiliary file {aux_file} not found for subset {subset}, skipping.")


def sync_subset_to_local(
    subset: str, target_dir: Path, repo_id: str = DEFAULT_REPO_ID, revision: str | None = None
) -> None:
    """Synchronizes a single dataset subset (images + metadata) to local disk.

    ``revision`` pins the Hugging Face repository commit (``xtask/datasets.toml``);
    ``None`` follows the default branch. ``annotations.jsonl`` is the subset's completion
    marker: it is written as ``annotations.jsonl.part`` and renamed only after every image
    and auxiliary file landed, so an interrupted or failed sync never looks complete.
    Raises on any image-write or auxiliary-download failure.
    """
    subset_dir: Path = target_dir / subset
    images_dir: Path = subset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--> Syncing: {subset}")

    try:
        ds = datasets.load_dataset(
            repo_id, subset, split="train", streaming=True, revision=revision
        )
    except Exception as e:
        logger.warning(f"    [!] Standard load failed for {subset}: {e}")
        logger.info("    [-->] Retrying with explicit data_files fallback...")
        try:
            ds = datasets.load_dataset(
                repo_id,
                data_files={"train": f"{subset}/train-*.parquet"},
                split="train",
                streaming=True,
                revision=revision,
            )
        except Exception as e2:
            logger.error(
                f"Failed to load dataset {subset} from {repo_id} (even with fallback): {e2}"
            )
            raise

    jsonl_path: Path = subset_dir / "annotations.jsonl"
    part_path: Path = subset_dir / "annotations.jsonl.part"
    jsonl_path.unlink(missing_ok=True)

    with (
        part_path.open("w", encoding="utf-8") as f,
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

        for fut in concurrent.futures.as_completed(futures):
            fut.result()  # propagate image-write failures

    aux_files: list[str] = ["coco_labels.json", "rich_truth.json"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(aux_files)) as executor:
        aux = [
            executor.submit(_download_aux, repo_id, subset, aux_file, target_dir, revision)
            for aux_file in aux_files
        ]
        for fut in aux:
            fut.result()  # propagate real download failures (absence is handled inside)

    part_path.replace(jsonl_path)


def main() -> None:
    """CLI entry point; delegates to the pinned registry (``cargo xtask data fetch hub``)."""
    from tools.bench.dataset_registry import fetch  # noqa: PLC0415 - avoid import cycle

    parser = argparse.ArgumentParser(
        description="Sync Hub subsets at the revision pinned in xtask/datasets.toml."
    )
    parser.add_argument(
        "--configs", nargs="*", default=["all"], help="Subsets to sync (default: all)"
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Target directory (default: $LOCUS_HUB_DATASET_DIR or project tests/data/hub_cache)",
    )
    args = parser.parse_args()
    subsets = "all" if "all" in args.configs else ",".join(args.configs)
    fetch("hub", dest=args.target_dir, subsets=subsets)


if __name__ == "__main__":
    main()
