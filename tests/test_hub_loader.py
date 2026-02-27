import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.bench.utils import HubBenchmarkLoader, TagGroundTruth


class TestHubBenchmarkLoader(unittest.TestCase):
    def setUp(self):
        self.loader = HubBenchmarkLoader()

    @patch("datasets.load_dataset")
    def test_load_subset_mapping(self, mock_load_dataset):
        # Create a mock PIL image
        pil_img = Image.new("L", (100, 100), color=128)

        # Mock dataset entry
        mock_entry = {
            "tag_id": 42,
            "corners": [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]],
            "image": pil_img,
        }

        # Mock the dataset object (iterable)
        mock_dataset = [mock_entry]
        mock_load_dataset.return_value = mock_dataset

        # Call the loader
        results = list(self.loader.stream_subset("subset_name"))

        # Verify results
        self.assertEqual(len(results), 1)
        name, img, gt_tags = results[0]

        self.assertEqual(name, "subset_name/0")
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (100, 100))
        self.assertEqual(img[0, 0], 128)

        self.assertEqual(len(gt_tags), 1)
        self.assertIsInstance(gt_tags[0], TagGroundTruth)
        self.assertEqual(gt_tags[0].tag_id, 42)
        np.testing.assert_array_almost_equal(
            gt_tags[0].corners,
            np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]], dtype=np.float32),
        )

    @patch("datasets.load_dataset")
    def test_load_subset_multiple_tags(self, mock_load_dataset):
        # In some subsets, corners and tag_ids might be lists if there are multiple tags per image
        # But looking at the schema provided, it seems like 1 tag per row (typical for synthetic hub datasets)
        # If there are multiple tags, the schema usually reflects that.
        # Let's assume 1 tag per row for now as per "single_tag_locus_v1_std41h12"

        pil_img = Image.new("L", (100, 100), color=200)
        mock_dataset = [
            {"tag_id": 1, "corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "image": pil_img},
            {"tag_id": 2, "corners": [[5, 5], [6, 5], [6, 6], [5, 6]], "image": pil_img},
        ]
        mock_load_dataset.return_value = mock_dataset

        results = list(self.loader.stream_subset("subset_name"))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][2][0].tag_id, 1)
        self.assertEqual(results[1][2][0].tag_id, 2)


if __name__ == "__main__":
    unittest.main()
