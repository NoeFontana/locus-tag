import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts.bench.sync_hub import sync_subset_to_local


class TestSyncHub(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("datasets.load_dataset")
    def test_sync_subset_structure(self, mock_load_dataset):
        pil_img = Image.new("L", (100, 100), color=128)

        # Mock a dataset item matching Hugging Face schema
        mock_item = {
            "image": pil_img,
            "image_id": "test_img_001",
            "tag_id": 42,
            "corners": [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]],
            "distance": 1.2,
            "angle_of_incidence": 45.0,
        }

        mock_load_dataset.return_value = iter([mock_item])

        # Run sync
        sync_subset_to_local("subset_name", self.test_dir)

        # Verify structure
        scenario_dir = self.test_dir / "subset_name"
        self.assertTrue(scenario_dir.exists())
        self.assertTrue((scenario_dir / "images").exists())
        self.assertTrue((scenario_dir / "images" / "test_img_001.png").exists())
        self.assertTrue((scenario_dir / "annotations.jsonl").exists())

        # Verify JSONL content
        with open(scenario_dir / "annotations.jsonl") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            data = json.loads(lines[0])
            self.assertEqual(data["image_id"], "test_img_001")
            self.assertEqual(data["tag_id"], 42)
            self.assertEqual(data["distance"], 1.2)
            self.assertEqual(data["image_filename"], "test_img_001.png")
            self.assertNotIn("image", data)


if __name__ == "__main__":
    unittest.main()
