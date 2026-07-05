"""Search strategies + cell construction.

Turns a :class:`SearchSpace` into the concrete list of :class:`Cell` work items
for a sweep: the Cartesian product of *parameter draws* (grid / random / optional
bayes) and *datasets*. The optuna backend is imported lazily so the base install
never needs it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

from tools.bench.tune.executor import Cell
from tools.bench.tune.materialize import param_hash
from tools.bench.tune.space import SearchSpace

Strategy = Literal["grid", "random", "bayes"]

# library id → shipped default space name.
DEFAULT_SPACE = {
    "locus": "locus_default",
    "opencv_aruco": "opencv_default",
    "apriltag": "apriltag_default",
}


def iter_param_draws(
    space: SearchSpace, strategy: str, n: int, seed: int
) -> Iterator[dict[str, Any]]:
    """Yield parameter draws for the requested strategy."""
    if strategy == "grid":
        yield from space.grid()
    elif strategy == "random":
        yield from space.random_draws(n, seed)
    elif strategy == "bayes":
        # Lazily imported so the optional optuna dependency is never required
        # unless a bayes sweep is actually requested.
        try:
            # optuna itself is the optional [tune] extra; search_optuna imports it
            # lazily and raises ImportError when the extra is not installed.
            from tools.bench.tune.search_optuna import optuna_draws
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise RuntimeError(
                "bayes strategy requires the 'tune' extra: pip install -e '.[tune]'"
            ) from exc
        yield from optuna_draws(space, n, seed)
    else:  # pragma: no cover - guarded by the CLI enum
        raise ValueError(f"unknown strategy {strategy!r}")


def build_cells(
    *,
    space: SearchSpace,
    datasets: list[str],
    family: int,
    strategy: str,
    n: int,
    seed: int,
    data_dir: str,
    limit: int | None = None,
    skip: int = 0,
) -> list[Cell]:
    """Build accuracy ``(config × dataset)`` cells for a sweep.

    Each distinct parameter draw is hashed once (the ``param_hash`` join key) and
    fanned across every dataset. Duplicate draws (grid can't produce them; random
    rarely can) are de-duplicated by hash so no config is evaluated twice. Cells
    are accuracy-only (``measure_latency`` stays False); latency is measured
    separately by the serial verification phase.
    """
    space_json = space.model_dump_json()
    seen_hashes: set[str] = set()
    cells: list[Cell] = []
    for values in iter_param_draws(space, strategy, n, seed):
        ph = param_hash(space.library, values)
        if ph in seen_hashes:
            continue
        seen_hashes.add(ph)
        for dataset in datasets:
            cells.append(
                Cell(
                    library=space.library,
                    param_hash=ph,
                    param_values=values,
                    dataset=dataset,
                    family=family,
                    space_json=space_json,
                    data_dir=data_dir,
                    limit=limit,
                    skip=skip,
                )
            )
    return cells


def baseline_cells(
    *,
    profiles: list[str],
    datasets: list[str],
    family: int,
    data_dir: str,
    limit: int | None = None,
    skip: int = 0,
) -> list[Cell]:
    """Build reference cells for shipped Locus profiles (the priority guard).

    Each profile becomes an empty-``params`` Locus space with that ``base_profile``,
    so ``from_params({})`` yields the shipped detector. Used to compare tuned
    frontier configs against ``standard`` / ``high_accuracy`` on the same data.
    """
    cells: list[Cell] = []
    for profile in profiles:
        space = SearchSpace(library="locus", base_profile=profile, params={})
        space_json = space.model_dump_json()
        # Profile lives in the space's base_profile, so params stay EMPTY (an
        # ``from_params({})`` reproduces the shipped detector). The hash only
        # needs to be unique per profile — it labels rows, it is not applied.
        ph = param_hash("locus", {"__baseline_profile__": profile})
        for dataset in datasets:
            cells.append(
                Cell(
                    library="locus",
                    param_hash=ph,
                    param_values={},
                    dataset=dataset,
                    family=family,
                    space_json=space_json,
                    data_dir=data_dir,
                    profile_label=profile,
                    limit=limit,
                    skip=skip,
                )
            )
    return cells
