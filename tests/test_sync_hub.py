import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from huggingface_hub.errors import EntryNotFoundError
from PIL import Image

from tools.bench.sync_hub import sync_subset_to_local


class TestSyncHub(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch(
        "tools.bench.sync_hub.hf_hub_download",
        side_effect=EntryNotFoundError("aux file absent from the repo"),
    )
    @patch("datasets.load_dataset")
    def test_sync_subset_structure(self, mock_load_dataset, _mock_aux):
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

    def _item(self):
        return {
            "image": Image.new("L", (8, 8), color=128),
            "image_id": "img_0",
            "tag_id": 1,
        }

    @patch("tools.bench.sync_hub.hf_hub_download", side_effect=ConnectionError("HF 503"))
    @patch("datasets.load_dataset")
    def test_aux_failure_raises_and_leaves_no_completion_marker(self, mock_load, _mock_aux):
        mock_load.return_value = iter([self._item()])
        with self.assertRaises(ConnectionError):
            sync_subset_to_local("subset_name", self.test_dir)
        # annotations.jsonl is the completion marker the registry checks; it must not exist.
        self.assertFalse((self.test_dir / "subset_name" / "annotations.jsonl").exists())

    @patch("tools.bench.sync_hub.hf_hub_download")
    @patch("datasets.load_dataset")
    def test_revision_is_forwarded(self, mock_load, mock_aux):
        mock_load.return_value = iter([self._item()])
        sync_subset_to_local("subset_name", self.test_dir, revision="a" * 40)
        self.assertEqual(mock_load.call_args.kwargs["revision"], "a" * 40)
        self.assertTrue(all(c.kwargs["revision"] == "a" * 40 for c in mock_aux.call_args_list))


if __name__ == "__main__":
    unittest.main()
