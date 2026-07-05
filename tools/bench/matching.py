"""Detection ↔ ground-truth matching — the single source of truth.

Both the headline aggregator (``Metrics.match_detections`` in ``utils.py``) and
the Tier-1 record collector (``Collector.observe`` in ``collect.py``) pair each
detection to at most one ground-truth tag by nearest same-id center within a
fixed pixel threshold. Before consolidation the greedy loop lived in two places
with the threshold hard-coded twice; they are now one function so recall/precision
can never silently diverge between the printed summary and the emitted records.

Kept numpy-only (no ``locus`` / ``cv2`` import) so it can be imported from
``utils`` without an import cycle — the ``TagGroundTruth`` annotation is resolved
lazily under ``TYPE_CHECKING`` and the matcher only duck-types ``.tag_id`` /
``.corners``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from tools.bench.utils import TagGroundTruth

# Detection ↔ GT pairing threshold in pixels. A detection whose center lands
# within this distance of a same-id GT center counts as a true positive.
MATCH_DISTANCE_THRESHOLD_PX = 20.0


@dataclass(frozen=True)
class MatchResult:
    """Outcome of :func:`match_detections_to_gt`.

    ``pairs`` holds ``(det_idx, gt_idx)`` tuples, greedily assigned in detection
    order; each ``det_idx`` and each ``gt_idx`` appears at most once. The helper
    views cover the two shapes callers need without re-walking the pairs.
    """

    pairs: list[tuple[int, int]]

    @property
    def matched_det_indices(self) -> set[int]:
        """Detection indices that paired with a GT tag."""
        return {det_idx for det_idx, _ in self.pairs}

    @property
    def matched_gt_indices(self) -> set[int]:
        """GT indices that were claimed by a detection."""
        return {gt_idx for _, gt_idx in self.pairs}

    def gt_index_to_det(self, detections: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        """Map ``gt_idx -> detection dict`` for the matched pairs."""
        return {gt_idx: detections[det_idx] for det_idx, gt_idx in self.pairs}


def match_detections_to_gt(
    detections: list[dict[str, Any]],
    gt_tags: list[TagGroundTruth],
    threshold: float = MATCH_DISTANCE_THRESHOLD_PX,
) -> MatchResult:
    """Greedily pair detections to same-id GT tags by nearest center.

    In detection order, each detection claims the nearest not-yet-used GT tag of
    the same id; the pairing is accepted only if that center distance is strictly
    below ``threshold``. This reproduces the original loops in
    ``Metrics.match_detections`` and ``Collector.observe`` exactly.
    """
    pairs: list[tuple[int, int]] = []
    used_gt: set[int] = set()
    for det_idx, det in enumerate(detections):
        det_center = np.asarray(det["center"], dtype=np.float64)
        best_idx = -1
        min_dist = float("inf")
        for idx, gt in enumerate(gt_tags):
            if idx in used_gt or gt.tag_id != det["id"]:
                continue
            gt_center = np.mean(gt.corners, axis=0)
            dist = float(np.linalg.norm(det_center - gt_center))
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        if best_idx != -1 and min_dist < threshold:
            used_gt.add(best_idx)
            pairs.append((det_idx, best_idx))
    return MatchResult(pairs)
