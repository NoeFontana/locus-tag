"""Generate the combined per-instance Tier-1 parquet for the comparison.

Runs up to four *series* — ``locus:tuned``, ``locus:shipped``,
``opencv_aruco:tuned``, ``apriltag:tuned`` — across the render-tag datasets and
flushes ONE parquet keyed by ``(binary, profile)``, identical in schema to
``bench real --record-out`` (so every downstream consumer works unchanged).

The tuned configs come from ``pareto/<lib>.json`` (the tuning harness output);
the shipped-Locus series uses a shipped profile (default ``high_accuracy``). All
detector construction goes through the wrappers' ``from_params`` via the shared
``tune.executor`` frame loop (``run_search_records``) — no duplicated loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tools.bench.collect import build_provenance
from tools.bench.records import write_records
from tools.bench.tune.executor import Cell, run_search_records
from tools.bench.tune.levers import chosen_per_library
from tools.bench.tune.materialize import param_hash
from tools.bench.tune.orchestrate import resolve_space
from tools.bench.tune.report_compare import _load_pareto_docs
from tools.bench.tune.space import SearchSpace

Variant = Literal["both", "tuned", "shipped"]

# Series shown in each report section.
TUNED_PROFILE = "tuned"
SHIPPED_PROFILE = "shipped"


@dataclass(frozen=True)
class ChosenConfig:
    """One series to run: a (library, profile) with its resolved config + space."""

    library: str  # binary id: locus | opencv_aruco | apriltag
    profile_label: str  # "tuned" | "shipped"
    param_hash: str
    param_values: dict[str, Any]
    space: SearchSpace
    space_name: str


def _tuned_entry(doc: dict, param_hash_value: str) -> dict:
    for entry in doc.get("frontier", []):
        if entry.get("param_hash") == param_hash_value:
            return entry
    raise ValueError(
        f"param_hash {param_hash_value!r} not found in frontier for {doc.get('library')}"
    )


def load_chosen_configs(
    pareto_dir: Path,
    *,
    variant: Variant = "both",
    shipped_profile: str = "high_accuracy",
    space_overrides: dict[str, str] | None = None,
) -> list[ChosenConfig]:
    """Resolve the series to run from ``pareto/*.json``.

    ``variant`` controls which Locus series are included; competitors are always
    their tuned frontier config. ``space_overrides`` maps ``library -> space ref``
    to faithfully rebuild a tuned config if the tune used a non-default space.
    """
    overrides = space_overrides or {}
    docs = _load_pareto_docs(pareto_dir)
    chosen = chosen_per_library(docs)  # {library: param_hash}
    by_library = {doc["library"]: doc for doc in docs}

    configs: list[ChosenConfig] = []
    want_tuned_locus = variant in ("both", "tuned")
    for library, phash in chosen.items():
        # Locus tuned is gated by variant; competitors are always tuned.
        if library == "locus" and not want_tuned_locus:
            continue
        entry = _tuned_entry(by_library[library], phash)
        space, space_name = resolve_space(library, overrides.get(library))
        configs.append(
            ChosenConfig(
                library=library,
                profile_label=TUNED_PROFILE,
                param_hash=phash,
                param_values=dict(entry["param_values"]),
                space=space,
                space_name=space_name,
            )
        )

    if variant in ("both", "shipped"):
        shipped_space = SearchSpace(library="locus", base_profile=shipped_profile, params={})
        configs.append(
            ChosenConfig(
                library="locus",
                profile_label=SHIPPED_PROFILE,
                param_hash=param_hash("locus", {"__shipped__": shipped_profile}),
                param_values={},
                space=shipped_space,
                space_name=f"shipped:{shipped_profile}",
            )
        )
    return configs


def _cells_for(
    configs: list[ChosenConfig],
    *,
    datasets: list[str],
    family: int,
    data_dir: str,
    limit: int | None,
    skip: int,
) -> list[Cell]:
    return [
        Cell(
            library=c.library,
            param_hash=c.param_hash,
            param_values=c.param_values,
            dataset=dataset,
            family=family,
            space_json=c.space.model_dump_json(),
            data_dir=data_dir,
            profile_label=c.profile_label,
            limit=limit,
            skip=skip,
        )
        for c in configs
        for dataset in datasets
    ]


def generate_instance_records(
    *,
    configs: list[ChosenConfig],
    datasets: list[str],
    family: int,
    data_dir: Path,
    out_path: Path,
    limit: int | None = None,
    skip: int = 0,
    workers: int | None = None,
) -> int:
    """Run every ``(series × dataset)`` cell and write one combined Tier-1 parquet.

    Returns the number of records written. A sidecar ``<out>.configs.json`` records
    the resolved series (library/profile/param_hash/space_name) for the report
    header and R1 auditability.
    """
    cells = _cells_for(
        configs, datasets=datasets, family=family, data_dir=str(data_dir), limit=limit, skip=skip
    )
    record_lists = run_search_records(cells, workers=workers)

    # A cell that ran always emits at least a missed_gt/matched row per GT tag. So an
    # empty cell means it errored or crashed. If SOME cells are empty while others
    # produced records, the empty series would be silently scored as all-missed —
    # refuse to write a corrupted comparison rather than alias a failure to a miss.
    empty = [i for i, records in enumerate(record_lists) if not records]
    nonempty = [i for i, records in enumerate(record_lists) if records]
    if empty and nonempty:
        failed = sorted(
            {(cells[i].library, cells[i].profile_label, cells[i].dataset) for i in empty}
        )
        raise ValueError(
            f"{len(failed)} (series, dataset) cell(s) produced no records while others "
            f"succeeded — a crash/error would silently score them as all-missed. "
            f"Failed cells: {failed}. Refusing to write a corrupted comparison parquet "
            f"(see the WARN lines above for the underlying errors)."
        )

    all_records = [record for records in record_lists for record in records]

    provenance = build_provenance()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_records(all_records, provenance, out_path)

    sidecar = out_path.with_suffix(out_path.suffix + ".configs.json")
    sidecar.write_text(
        json.dumps(
            [
                {
                    "library": c.library,
                    "profile": c.profile_label,
                    "param_hash": c.param_hash,
                    "space_name": c.space_name,
                    "param_values": c.param_values,
                }
                for c in configs
            ],
            indent=2,
            default=str,
        )
    )
    return len(all_records)
