import io
import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

import numpy as np

from scripts.locus_bench import run_hosted_benchmark


class TestCLIHosted(unittest.TestCase):
    @patch("scripts.locus_bench.HubBenchmarkLoader")
    @patch("scripts.locus_bench.LocusWrapper")
    @patch("scripts.locus_bench.Metrics.match_detections")
    def test_run_hosted_benchmark_flow(self, mock_match, mock_locus, mock_loader_cls):
        # Setup mocks
        mock_loader = mock_loader_cls.return_value
        mock_loader.stream_subset.return_value = iter(
            [("subset/0", np.zeros((100, 100), dtype=np.uint8), [MagicMock()])]
        )

        mock_detector = mock_locus.return_value
        mock_detector.name = "Locus"
        mock_detector.detect.return_value = ([], {})

        mock_match.return_value = (0, 0.0, 1)  # correct, err_sum, matched_gt

        args = Namespace(configs=["subset1"], compare=False, decimation=1, limit=None, skip=0)

        # Capture stdout to avoid noise and verify output if needed
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            run_hosted_benchmark(args)
            output = fake_out.getvalue()

        # Verify calls
        mock_loader_cls.assert_called_once()
        mock_loader.stream_subset.assert_called_with("subset1")
        mock_detector.detect.assert_called_once()
        self.assertIn("Evaluating subset1", output)
        self.assertIn("Locus", output)


if __name__ == "__main__":
    unittest.main()
