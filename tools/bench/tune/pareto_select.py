"""Pareto selection with a constraint gate and the render-tag priority guard.

Given the per-cell accuracy summaries from a sweep (and, after serial
verification, latency), this:

1. **Aggregates** each config across the swept datasets (macro-average — the
   exact percentile pooling would need pooled raw records; macro-average is the
   standard multi-dataset selection objective and is exact for a single dataset).
2. **Gates** on hard constraints: ``precision >= floor`` and, when a latency
   budget is set, ``latency_p95 <= budget``. Infeasible configs are retained
   (flagged) for lever analysis but excluded from the frontier.
3. Builds the **Pareto frontier** over ``(maximize recall, minimize tail)`` via
   the shared :func:`tools.bench.pareto_core.pareto_mask`.
4. Applies the **priority guard**: annotates whether each frontier config avoids
   regressing render-tag mean RMSE / p99 tail versus the shipped baselines, so a
   tail regression is never silently promoted (feedback: never trade render-tag
   tail/mean for recall).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from tools.bench.pareto_core import pareto_mask
from tools.bench.tune.executor import CellResult

# Selection tail-metric name → ``ConfigSummary`` field.
TAIL_FIELD = {"trans_p99": "trans_p99_m", "rot_p99": "rot_p99_deg"}
_EPS = 1e-9


@dataclass
class ConfigSummary:
    """One config's metrics, macro-averaged across the swept datasets."""

    library: str
    param_hash: str
    param_values: dict[str, object]
    recall: float
    precision: float
    trans_mean_m: float
    trans_p99_m: float
    rot_p99_deg: float
    latency_p95_ms: float | None
    n_datasets: int


@dataclass
class FrontierEntry:
    """A config plus its feasibility / frontier / guard annotations."""

    summary: ConfigSummary
    feasible: bool
    on_frontier: bool
    no_tail_regression: bool = False
    no_mean_regression: bool = False

    @property
    def promotable(self) -> bool:
        """Derived: safe to ship only if it regresses neither tail nor mean."""
        return self.no_tail_regression and self.no_mean_regression

    def to_dict(self) -> dict[str, object]:
        d = asdict(self.summary)
        d.update(
            feasible=self.feasible,
            on_frontier=self.on_frontier,
            no_tail_regression=self.no_tail_regression,
            no_mean_regression=self.no_mean_regression,
            promotable=self.promotable,
        )
        return d


def _mean_or_nan(values: list[float]) -> float:
    finite = [v for v in values if v == v]  # drop NaN
    return float(np.mean(finite)) if finite else float("nan")


def aggregate_across_datasets(results: list[CellResult]) -> list[ConfigSummary]:
    """Macro-average each ``(library, param_hash)`` config over its datasets."""
    groups: dict[tuple[str, str], list[CellResult]] = {}
    for r in results:
        if r.error or not r.overall:
            continue
        groups.setdefault((r.library, r.param_hash), []).append(r)

    summaries: list[ConfigSummary] = []
    for (library, param_hash), cells in groups.items():
        overalls = [c.overall for c in cells]
        # Latency is intentionally left None here: the accuracy sweep runs in the
        # pinned pool (latency_valid=False), so any latency it carries is
        # contention-poisoned. Trustworthy latency is folded in later by the
        # serial verification phase (orchestrate._merge_latency).
        summaries.append(
            ConfigSummary(
                library=library,
                param_hash=param_hash,
                param_values=cells[0].param_values,
                recall=_mean_or_nan([o["recall"] for o in overalls]),
                precision=_mean_or_nan([o["precision"] for o in overalls]),
                trans_mean_m=_mean_or_nan([o["trans_err_mean_m"] for o in overalls]),
                trans_p99_m=_mean_or_nan([o["trans_err_p99_m"] for o in overalls]),
                rot_p99_deg=_mean_or_nan([o["rot_err_p99_deg"] for o in overalls]),
                latency_p95_ms=None,
                n_datasets=len(cells),
            )
        )
    return summaries


def select_frontier(
    summaries: list[ConfigSummary],
    *,
    precision_floor: float,
    tail_metric: str,
    latency_budget_ms: float | None = None,
) -> list[FrontierEntry]:
    """Gate on constraints, then mark the Pareto frontier over (recall, tail).

    Returns an entry for **every** summary (feasible or not) so infeasible
    configs stay available to the lever analysis; only ``on_frontier`` entries
    are the deployment candidates.
    """
    if not summaries:
        return []
    tail_field = TAIL_FIELD[tail_metric]

    feasible = np.array(
        [
            (s.precision >= precision_floor - _EPS)
            # A non-finite tail (no pose-matched detection, e.g. a pose-less
            # dataset or a zero-recall config) cannot sit on a frontier whose
            # objective is that tail — NaN is never dominated in pareto_mask, so
            # without this gate every such config would be marked on_frontier and
            # selection would silently become a no-op.
            and math.isfinite(getattr(s, tail_field))
            and (
                latency_budget_ms is None
                or (s.latency_p95_ms is not None and s.latency_p95_ms <= latency_budget_ms + _EPS)
            )
            for s in summaries
        ],
        dtype=bool,
    )
    objectives = np.array([[s.recall, getattr(s, tail_field)] for s in summaries], dtype=np.float64)
    # maximize recall, minimize tail.
    on_frontier = pareto_mask(objectives, senses=(1, -1), strict=False, feasible=feasible)
    return [
        FrontierEntry(summary=s, feasible=bool(feasible[i]), on_frontier=bool(on_frontier[i]))
        for i, s in enumerate(summaries)
    ]


def apply_baseline_guard(
    entries: list[FrontierEntry], baselines: list[ConfigSummary]
) -> ConfigSummary | None:
    """Annotate frontier entries with no-regression flags vs shipped baselines.

    Tail and mean are each guarded against the *best* baseline on that metric —
    which may be two different profiles (e.g. ``high_accuracy`` has the lower p99
    tail while ``standard`` has the lower mean RMSE). Judging both against a
    single baseline would let a config regress the other profile's stronger
    metric and still read as ``promotable``. Encodes "never trade render-tag
    tail/mean for recall". Returns the lowest-tail baseline as the reported
    incumbent (or ``None`` if no baselines were provided).
    """
    valid = [b for b in baselines if math.isfinite(b.trans_p99_m)]
    if not valid:
        return None
    best_tail = min(valid, key=lambda b: b.trans_p99_m)
    mean_valid = [b for b in valid if math.isfinite(b.trans_mean_m)]
    best_mean = min(mean_valid, key=lambda b: b.trans_mean_m) if mean_valid else best_tail
    for e in entries:
        if not e.on_frontier:
            continue
        s = e.summary
        e.no_tail_regression = s.trans_p99_m <= best_tail.trans_p99_m + _EPS
        e.no_mean_regression = s.trans_mean_m <= best_mean.trans_mean_m + _EPS
    return best_tail
