"""Coverage for ``tools/bench/metrics.py`` — the recall/precision contract."""

from __future__ import annotations

import pytest

from tools.bench.metrics import compute_precision, compute_recall


@pytest.mark.parametrize(
    ("tp", "fn", "expected"),
    [
        (0, 0, 0.0),  # vacuous: no GT in frame → 0 by convention
        (5, 0, 1.0),  # perfect recall
        (0, 5, 0.0),  # zero recall
        (3, 1, 0.75),
    ],
)
def test_compute_recall(tp: int, fn: int, expected: float) -> None:
    assert compute_recall(tp, fn) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("tp", "fp", "expected"),
    [
        (0, 0, 0.0),  # vacuous: no detections → 0
        (5, 0, 1.0),  # perfect precision
        (0, 5, 0.0),  # all false positives
        (1, 99, 0.01),  # mostly false positives
    ],
)
def test_compute_precision(tp: int, fp: int, expected: float) -> None:
    assert compute_precision(tp, fp) == pytest.approx(expected)
