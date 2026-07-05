"""End-to-end sweep / tune orchestration (CLI-agnostic, so it is testable).

``run_sweep`` fans a search space across cores (accuracy only) and writes the
tidy result tables. ``run_tune`` adds selection: aggregate → Pareto+constraint
gate → **serial** latency verification of the frontier → latency gate → baseline
priority guard → ``pareto/<library>.json``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from tools.bench.collect import build_provenance
from tools.bench.tune.executor import Cell, CellResult, run_search, verify_latency
from tools.bench.tune.pareto_select import (
    ConfigSummary,
    FrontierEntry,
    aggregate_across_datasets,
    apply_baseline_guard,
    select_frontier,
)
from tools.bench.tune.results import write_results
from tools.bench.tune.search import (
    DEFAULT_SPACE,
    baseline_cells,
    build_cells,
)
from tools.bench.tune.space import SearchSpace, load_space
from tools.bench.utils import resolve_tag_family

# Re-exported so the CLI (and callers) resolve families through the single
# canonical mapping in tools.bench.utils.
resolve_family = resolve_tag_family


def resolve_space(library: str, space_arg: str | None) -> tuple[SearchSpace, str]:
    """Load the space for ``library``; ``space_arg`` overrides the shipped default."""
    ref = space_arg or DEFAULT_SPACE[library]
    space = load_space(ref)
    if space.library != library:
        raise ValueError(f"space {ref!r} declares library={space.library!r}, expected {library!r}")
    return space, ref


@dataclass
class SweepOutput:
    results: list[CellResult]
    results_path: Path
    configs_path: Path


def _space_arg_for(libraries: list[str], space_arg: str | None) -> str | None:
    """An explicit ``--space`` only applies when a single library is swept."""
    return space_arg if len(libraries) == 1 else None


def run_sweep(
    *,
    libraries: list[str],
    space_arg: str | None,
    datasets: list[str],
    family: int,
    strategy: str,
    n: int,
    seed: int,
    workers: int | None,
    data_dir: str,
    out_dir: Path,
    limit: int | None = None,
    skip: int = 0,
) -> SweepOutput:
    """Run a parallel accuracy sweep over one or more libraries.

    All libraries' cells go into a single ``run_search`` and one combined
    ``tune_results.parquet`` — the shape the comparative report consumes.
    """
    all_cells: list[Cell] = []
    for library in libraries:
        space, _ = resolve_space(library, _space_arg_for(libraries, space_arg))
        all_cells.extend(
            build_cells(
                space=space,
                datasets=datasets,
                family=family,
                strategy=strategy,
                n=n,
                seed=seed,
                data_dir=data_dir,
                limit=limit,
                skip=skip,
            )
        )
    results = run_search(all_cells, workers=workers)
    provenance = build_provenance()
    results_path, configs_path = write_results(results, out_dir, provenance)
    return SweepOutput(results=results, results_path=results_path, configs_path=configs_path)


@dataclass
class LibraryTune:
    """Selection outcome for one library."""

    library: str
    entries: list[FrontierEntry]
    reference_baseline: ConfigSummary | None
    pareto_path: Path


@dataclass
class TuneOutput:
    per_library: dict[str, LibraryTune]
    results_path: Path
    configs_path: Path


def _merge_latency(entries: list[FrontierEntry], latency_results: list[CellResult]) -> None:
    """Fold verified per-config latency into the frontier summaries (in place)."""
    lat_by_hash: dict[str, list[float]] = {}
    for r in latency_results:
        if r.error or "latency_p95_ms" not in r.overall:
            continue
        lat_by_hash.setdefault(r.param_hash, []).append(r.overall["latency_p95_ms"])
    for e in entries:
        vals = lat_by_hash.get(e.summary.param_hash)
        if vals:
            e.summary.latency_p95_ms = float(sum(vals) / len(vals))


def run_tune(
    *,
    libraries: list[str],
    space_arg: str | None,
    datasets: list[str],
    family: int,
    strategy: str,
    n: int,
    seed: int,
    workers: int | None,
    data_dir: str,
    out_dir: Path,
    precision_floor: float,
    tail_metric: str,
    latency_budget_ms: float | None,
    protect_baselines: list[str],
    limit: int | None = None,
    skip: int = 0,
) -> TuneOutput:
    """Full tune over one or more libraries.

    One combined sweep feeds a per-library selection loop: accuracy frontier →
    serial latency verification → latency gate → (Locus only) baseline priority
    guard → ``pareto/<library>.json``.
    """
    sweep = run_sweep(
        libraries=libraries,
        space_arg=space_arg,
        datasets=datasets,
        family=family,
        strategy=strategy,
        n=n,
        seed=seed,
        workers=workers,
        data_dir=data_dir,
        out_dir=out_dir,
        limit=limit,
        skip=skip,
    )
    summaries_all = aggregate_across_datasets(sweep.results)

    # The baseline guard reuses one baseline sweep across libraries (Locus only).
    baseline_summaries: list[ConfigSummary] = []
    if protect_baselines:
        base_cells = baseline_cells(
            profiles=protect_baselines,
            datasets=datasets,
            family=family,
            data_dir=data_dir,
            limit=limit,
            skip=skip,
        )
        baseline_summaries = aggregate_across_datasets(run_search(base_cells, workers=workers))

    per_library: dict[str, LibraryTune] = {}
    for library in libraries:
        lib_summaries = [s for s in summaries_all if s.library == library]
        entries = select_frontier(
            lib_summaries, precision_floor=precision_floor, tail_metric=tail_metric
        )
        frontier = [e for e in entries if e.on_frontier]

        # Phase B: serial latency verification of the accuracy frontier.
        if frontier:
            space, _ = resolve_space(library, _space_arg_for(libraries, space_arg))
            latency_cells: list[Cell] = []
            for e in frontier:
                latency_cells.extend(
                    _latency_cells_for(e.summary, space, family, datasets, data_dir, limit, skip)
                )
            _merge_latency(frontier, verify_latency(latency_cells))

        # Phase C: latency budget gate (now that latency is known). A missing or
        # non-finite latency (e.g. a dataset whose frames were all unreadable)
        # is treated as failing the budget, not silently passing it.
        if latency_budget_ms is not None:
            for e in frontier:
                lat = e.summary.latency_p95_ms
                if lat is None or not math.isfinite(lat) or lat > latency_budget_ms:
                    e.feasible = False
                    e.on_frontier = False

        # Priority guard (Locus only — competitors have no shipped profiles).
        reference = (
            apply_baseline_guard(entries, baseline_summaries)
            if (library == "locus" and baseline_summaries)
            else None
        )

        pareto_path = _write_pareto(
            out_dir=out_dir,
            library=library,
            datasets=datasets,
            tail_metric=tail_metric,
            precision_floor=precision_floor,
            latency_budget_ms=latency_budget_ms,
            entries=entries,
            reference=reference,
        )
        per_library[library] = LibraryTune(
            library=library,
            entries=entries,
            reference_baseline=reference,
            pareto_path=pareto_path,
        )

    return TuneOutput(
        per_library=per_library,
        results_path=sweep.results_path,
        configs_path=sweep.configs_path,
    )


def _latency_cells_for(
    summary: ConfigSummary,
    space: SearchSpace,
    family: int,
    datasets: list[str],
    data_dir: str,
    limit: int | None,
    skip: int,
) -> list[Cell]:
    """One explicit-config latency cell per dataset for a frontier summary."""
    space_json = space.model_dump_json()
    return [
        Cell(
            library=summary.library,
            param_hash=summary.param_hash,
            param_values=summary.param_values,
            dataset=dataset,
            family=family,
            space_json=space_json,
            data_dir=data_dir,
            limit=limit,
            skip=skip,
            measure_latency=True,
        )
        for dataset in datasets
    ]


def _write_pareto(
    *,
    out_dir: Path,
    library: str,
    datasets: list[str],
    tail_metric: str,
    precision_floor: float,
    latency_budget_ms: float | None,
    entries: list[FrontierEntry],
    reference: ConfigSummary | None,
) -> Path:
    frontier = [e.to_dict() for e in entries if e.on_frontier]
    doc = {
        "library": library,
        "datasets": datasets,
        "tail_metric": tail_metric,
        "precision_floor": precision_floor,
        "latency_budget_ms": latency_budget_ms,
        "reference_baseline": (
            {
                "param_values": reference.param_values,
                "trans_mean_m": reference.trans_mean_m,
                "trans_p99_m": reference.trans_p99_m,
                "recall": reference.recall,
            }
            if reference is not None
            else None
        ),
        "frontier": frontier,
    }
    pareto_dir = out_dir / "pareto"
    pareto_dir.mkdir(parents=True, exist_ok=True)
    path = pareto_dir / f"{library}.json"
    path.write_text(json.dumps(doc, indent=2, default=str))
    return path
