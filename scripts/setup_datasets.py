import tarfile
import urllib.request
from pathlib import Path

DATA_URL = "https://april.eecs.umich.edu/media/apriltag/apriltag_test_images.tar.gz"
TARGET_DIR = Path("tests/data/umich")
archive_path = TARGET_DIR / "apriltag_test_images.tar.gz"


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        print(f"Downloading {DATA_URL}...")
        try:
            urllib.request.urlretrieve(DATA_URL, archive_path)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download: {e}")
            # Fallback URL if the first one fails
            ALT_URL = "https://april.eecs.umich.edu/media/apriltag/test_images.tar.gz"
            print(f"Trying fallback {ALT_URL}...")
            urllib.request.urlretrieve(ALT_URL, archive_path)
            print("Download complete.")

    if archive_path.exists():
        print(f"Extracting {archive_path}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=TARGET_DIR)
        print("Extraction complete.")
        # Optional: Remove archive after extraction to save space,
        # but user mentioned caching in CI, so keeping it might be better
        # inside the target dir if we use it for cache key.
    else:
        print("Failed to find or download dataset.")


if __name__ == "__main__":
    main()
