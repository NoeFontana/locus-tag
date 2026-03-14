from unittest.mock import patch

from typer.testing import CliRunner

from tools.cli import app

runner = CliRunner()


@patch("tools.cli.rr")
def test_bench_real_rerun_flag(mock_rr):
    # This should now pass as --rerun is implemented
    result = runner.invoke(app, ["bench", "real", "--rerun", "--limit", "1"])
    assert result.exit_code == 0
    # Verify that rr.init was called because we passed --rerun
    mock_rr.init.assert_called()


@patch("tools.cli.rr")
def test_visualize_rerun_flag(mock_rr):
    # visualize command now has --rerun flag
    result = runner.invoke(app, ["visualize", "--rerun", "--limit", "1"])
    assert result.exit_code == 0
    mock_rr.init.assert_called()
