import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np

# Pre-create mocks
mock_rr = MagicMock()
mock_locus = MagicMock()
mock_cv2 = MagicMock()


class TestRerunMapping(unittest.TestCase):
    def test_rerun_diagnostic_logging(self):
        # 1. Setup sys.modules mocks
        with patch.dict("sys.modules", {"rerun": mock_rr, "locus": mock_locus, "cv2": mock_cv2}):
            import importlib

            import tools.cli

            importlib.reload(tools.cli)

            # After reload with `rerun` mocked in sys.modules, tools.cli.rr is
            # the mock (not None) so the `--no-rerun` / install-hint guards
            # in `visualize` fall through to the real path under test.

            from typer.testing import CliRunner

            from tools.bench.utils import DatasetMetadata
            from tools.cli import app

            runner = CliRunner()

            with (
                patch("tools.bench.utils.DatasetLoader") as mock_loader,
                patch("pathlib.Path.exists") as mock_exists,
            ):
                mock_exists.return_value = True

                # 2. Setup mock data
                mock_cv2.imread.return_value = np.zeros((100, 100), dtype=np.uint8)

                mock_detector = MagicMock()
                mock_locus.Detector.return_value = mock_detector

                # Simulate detection result with full telemetry
                mock_batch = MagicMock()
                mock_batch.ids = np.array([42], dtype=np.uint32)
                mock_batch.corners = np.array(
                    [[[10, 10], [20, 10], [20, 20], [10, 20]]], dtype=np.float32
                )
                mock_batch.error_rates = np.array([0.0], dtype=np.float32)
                mock_batch.rejected_corners = np.array(
                    [[[50, 50], [60, 50], [60, 60], [50, 60]]], dtype=np.float32
                )
                mock_batch.rejected_error_rates = np.array([10.0], dtype=np.float32)

                mock_telemetry = MagicMock()
                mock_telemetry.subpixel_jitter = np.array(
                    [[[0.1, 0.1], [-0.1, 0.1], [0.1, -0.1], [-0.1, -0.1]]], dtype=np.float32
                )
                mock_telemetry.reprojection_errors = np.array([0.05], dtype=np.float32)
                mock_telemetry.binarized = np.zeros((100, 100), dtype=np.uint8)
                mock_telemetry.threshold_map = np.zeros((100, 100), dtype=np.uint8)

                mock_batch.telemetry = mock_telemetry
                mock_batch.__len__.return_value = 1

                mock_detector.detect.return_value = mock_batch

                # Mock dataset loader
                mock_inst = mock_loader.return_value
                mock_inst.prepare_icra.return_value = True
                mock_inst.find_datasets.return_value = [
                    ("ds1", Path("/tmp"), {"test.png": []}, DatasetMetadata())
                ]

                # 3. Invoke visualize
                runner.invoke(app, ["visualize", "--limit", "1", "--rerun"])

                # 4. Assert Rerun logs were called for each diagnostic stream
                mock_rr.log.assert_any_call("pipeline/2_binarized", ANY)
                mock_rr.log.assert_any_call("pipeline/rejected", ANY)
                mock_rr.log.assert_any_call("pipeline/detections/subpixel_jitter", ANY)
                mock_rr.log.assert_any_call("pipeline/detections/tags/42/repro_err", ANY)

                # Verify 0.30.0 connectivity API
                mock_rr.connect_grpc.assert_called_with("rerun+http://127.0.0.1:9876/proxy")


if __name__ == "__main__":
    unittest.main()
