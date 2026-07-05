"""Run one tuning *cell* — a (library, params, dataset) triple — to a summary.

``run_cell`` is the serial core used both directly (single-cell parity checks)
and inside the parallel pool worker (P3). It rebuilds the wrapper from a search
space + parameter draw, drives it across a dataset's frames through the shared
``Collector`` (so the Tier-1 records are identical to ``bench real``'s), and
reduces them via ``tune.aggregate.summarize``.

Latency is measured **only** when ``cell.measure_latency`` is set (the serial
verification phase). In the parallel search phase it stays off and
``CellResult.latency_valid`` is ``False`` so no contention-poisoned timing can
leak into selection.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from tools.bench.collect import Collector
from tools.bench.tune.aggregate import summarize
from tools.bench.tune.space import SearchSpace
from tools.bench.utils import (
    HUB_CACHE_DIR,
    DatasetLoader,
    HubDatasetLoader,
    LibraryWrapper,
    LocusWrapper,
    OpenCVWrapper,
    TagAxes,
    TagGroundTruth,
)
from tools.bench.utils import AprilTagWrapper as _AprilTagWrapper

# library_id -> wrapper class, so a cell can rebuild the right detector.
_WRAPPERS: dict[str, type[LibraryWrapper]] = {
    "locus": LocusWrapper,
    "opencv_aruco": OpenCVWrapper,
    "apriltag": _AprilTagWrapper,
}


@dataclass(frozen=True)
class Cell:
    """One unit of tuning work — picklable so it can cross a process boundary."""

    library: str
    param_hash: str
    param_values: dict[str, Any]
    dataset: str
    family: int
    # Self-contained ``SearchSpace`` JSON so a worker rebuilds the exact space
    # (incl. base_profile + fixed) without a filesystem lookup.
    space_json: str
    data_dir: str = str(HUB_CACHE_DIR)
    profile_label: str = "tuned"
    # Latency is measured only for serial verification cells; the parallel search
    # must leave this False (run_search asserts it — see below).
    measure_latency: bool = False
    skip: int = 0
    limit: int | None = None


@dataclass
class CellResult:
    """Metric summary for one cell."""

    library: str
    param_hash: str
    dataset: str
    param_values: dict[str, Any]
    overall: dict[str, float]
    per_stratum: dict[str, dict[str, float]]
    latency_valid: bool
    n_frames: int
    error: str | None = None


def _gt_list(gt_tags: dict[int, dict[str, Any]]) -> list[TagGroundTruth]:
    """Hub ``gt_map[img]["tags"]`` dict → the ``TagGroundTruth`` list Collector wants."""
    return [
        TagGroundTruth(
            tag_id=tid,
            corners=np.asarray(d["corners"], dtype=np.float32),
            pose=d.get("pose"),
        )
        for tid, d in gt_tags.items()
    ]


def _select_images(names: list[str], skip: int, limit: int | None) -> list[str]:
    names = sorted(names)
    if skip:
        names = names[skip:]
    if limit is not None:  # `limit == 0` means zero frames, not "no limit"
        names = names[:limit]
    return names


def run_cell(cell: Cell) -> CellResult:
    """Execute one cell serially and return its metric summary.

    Errors constructing the detector or loading data are captured into
    ``CellResult.error`` (with empty metrics) rather than raised, so one bad
    cell never aborts a whole sweep.
    """
    try:
        return _run_cell_inner(cell)
    except Exception as exc:  # noqa: BLE001 — a bad cell must not kill the sweep
        return CellResult(
            library=cell.library,
            param_hash=cell.param_hash,
            dataset=cell.dataset,
            param_values=cell.param_values,
            overall={},
            per_stratum={},
            latency_valid=False,
            n_frames=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _run_cell_inner(cell: Cell) -> CellResult:
    space = SearchSpace.model_validate_json(cell.space_json)
    wrapper_cls = _WRAPPERS[cell.library]
    wrapper = wrapper_cls.from_params(family=cell.family, params=cell.param_values, space=space)

    data_dir = Path(cell.data_dir)
    ds = HubDatasetLoader(root=data_dir).load_dataset(cell.dataset)
    if ds.intrinsics is None:
        raise ValueError(f"dataset {cell.dataset!r} has no intrinsics (k_matrix)")
    eval_tag_size = ds.tag_size if ds.tag_size is not None else 1.0

    # Per-tag stratification axes (mirrors the bench_real hub path).
    axes: dict[str, dict[int, TagAxes]] = DatasetLoader(icra_dir=data_dir).load_axes(cell.dataset)

    img_names = _select_images(list(ds.gt_map.keys()), cell.skip, cell.limit)
    collector = Collector.new(cell.param_hash, cell.library, cell.profile_label, cell.dataset)

    n_frames = 0
    for img_name in img_names:
        img_path = ds.images_dir / img_name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        n_frames += 1

        start = time.perf_counter()
        detections, rejected = wrapper.detect(img, intrinsics=ds.intrinsics, tag_size=eval_tag_size)
        latency_ms = (time.perf_counter() - start) * 1000.0

        collector.observe(
            image_id=img_name,
            detections=detections,
            gt_tags=_gt_list(ds.gt_map[img_name]["tags"]),
            axes_lookup=axes.get(img_name, {}),
            frame_latency_ms=latency_ms,
            rejected=rejected,
            resolution_h=int(img.shape[0]),
            intrinsics=ds.intrinsics,
        )

    overall, per_stratum = summarize(collector.records, include_latency=cell.measure_latency)
    return CellResult(
        library=cell.library,
        param_hash=cell.param_hash,
        dataset=cell.dataset,
        param_values=cell.param_values,
        overall=overall,
        per_stratum=per_stratum,
        latency_valid=cell.measure_latency,
        n_frames=n_frames,
    )


def _init_worker() -> None:
    """Pin every thread pool to 1 thread *before* any detection runs.

    Locus drives the **global** rayon pool (``rayon::current_num_threads``); with
    the ``spawn`` start method the child's pool is unbuilt at this point, so
    setting ``RAYON_NUM_THREADS`` here means the first ``detect()`` builds a
    single-threaded pool. Combined with ``cv2.setNumThreads(0)`` and
    ``AprilTagWrapper.from_params``' forced ``nthreads=1``, this keeps N workers ×
    1 thread = N threads — no ``n_cores²`` oversubscription. Single-threaded rayon
    is also deterministic, so worker count cannot perturb accuracy metrics.
    """
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENCV_NUM_THREADS"] = "1"
    with contextlib.suppress(Exception):  # best-effort; env vars already cap threads
        cv2.setNumThreads(0)


def _cell_key(cell: Cell) -> tuple[str, str, str]:
    return (cell.library, cell.param_hash, cell.dataset)


def _error_result(cell: Cell, message: str) -> CellResult:
    return CellResult(
        library=cell.library,
        param_hash=cell.param_hash,
        dataset=cell.dataset,
        param_values=cell.param_values,
        overall={},
        per_stratum={},
        latency_valid=False,
        n_frames=0,
        error=message,
    )


def _run_isolated(cell: Cell) -> CellResult:
    """Run a single cell in its own fresh 1-worker pool, surviving native crashes.

    Competitor detectors (e.g. pupil_apriltags with some params) can *segfault*,
    which kills the worker and poisons a shared pool. Isolating the cell means a
    crash becomes an errored ``CellResult`` for that one config instead of taking
    down the sweep.
    """
    ctx = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=_init_worker) as pool:
            return pool.submit(run_cell, cell).result()
    except Exception as exc:  # noqa: BLE001 — includes BrokenProcessPool (native crash)
        return _error_result(cell, f"worker crashed (likely native): {type(exc).__name__}")


def _pool_round(
    cells: list[Cell],
    *,
    n_workers: int,
    ctx: BaseContext,
    progress: bool,
    desc: str,
) -> dict[tuple[str, str, str], CellResult]:
    """Run cells in one shared pool; return whatever completed before a crash.

    A native crash poisons the pool, so results collected before the break are
    returned and the caller retries the survivors.
    """
    done: dict[tuple[str, str, str], CellResult] = {}
    with (
        contextlib.suppress(BrokenProcessPool),
        ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx, initializer=_init_worker
        ) as pool,
    ):
        futures = {pool.submit(run_cell, cell): _cell_key(cell) for cell in cells}
        for future in tqdm(
            as_completed(futures), total=len(futures), disable=not progress, desc=desc
        ):
            try:
                done[futures[future]] = future.result()
            except BrokenProcessPool:
                break  # pool poisoned by a native crash — survivors retried by caller
    return done


def run_search(
    cells: list[Cell], *, workers: int | None = None, progress: bool = True
) -> list[CellResult]:
    """Fan cells across a process pool, returning results in **input order**.

    Every cell runs in a ``spawn`` worker pinned by :func:`_init_worker`, so the
    result is independent of worker count and completion order (results are keyed
    and re-ordered to match ``cells``). ``workers`` defaults to ``os.cpu_count()``;
    since each worker is single-threaded this saturates cores without
    oversubscription.

    A native crash in one worker poisons the whole pool. Rather than collapse to
    fully-serial execution, the survivors are retried in a fresh **shared** pool
    (keeping parallelism); only when a round makes zero progress — i.e. the head
    cell reliably crashes any pool it enters — is that single cell run in
    isolation and recorded as one errored ``CellResult``. So a handful of bad
    configs cost a handful of isolated runs, not the whole sweep's parallelism.
    """
    if not cells:
        return []
    # Structural guard for the accuracy-parallel / latency-serial split: timing
    # measured under N-way pool contention is invalid, so the pinned pool must
    # never carry latency-measuring cells. Latency is measured only by the serial
    # ``verify_latency`` path.
    if any(cell.measure_latency for cell in cells):
        raise ValueError(
            "run_search received measure_latency=True cells; parallel latency is "
            "contention-poisoned — use verify_latency for latency measurement"
        )
    n_workers = workers or (os.cpu_count() or 1)
    ctx = multiprocessing.get_context("spawn")
    by_key: dict[tuple[str, str, str], CellResult] = {}
    remaining = list(cells)
    while remaining:
        before = len(remaining)
        by_key.update(
            _pool_round(remaining, n_workers=n_workers, ctx=ctx, progress=progress, desc="sweep")
        )
        remaining = [cell for cell in remaining if _cell_key(cell) not in by_key]
        if len(remaining) == before:
            # Zero progress: the first survivor crashes any pool it enters.
            # Isolate exactly that cell (it runs alone, so the crash is attributable).
            culprit = remaining.pop(0)
            by_key[_cell_key(culprit)] = _run_isolated(culprit)

    # Re-order deterministically to match the input cell list.
    return [by_key[_cell_key(cell)] for cell in cells]


def _run_latency_isolated(cell: Cell) -> CellResult:
    """Run one latency cell in a fresh, **un-pinned** 1-worker subprocess.

    Unlike :func:`_run_isolated` (which pins threads for the accuracy sweep),
    this uses *no* initializer, so the child's rayon/OpenCV pools use all cores —
    production threading, the point of latency verification. Running in a child
    (not the main process) means a native crash — heap corruption / segfault in a
    competitor detector for some frontier config — becomes one errored
    ``CellResult`` instead of taking down the whole tune.
    """
    ctx = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
            return pool.submit(run_cell, cell).result()
    except Exception as exc:  # noqa: BLE001 — includes BrokenProcessPool (native crash)
        return _error_result(cell, f"latency worker crashed (likely native): {type(exc).__name__}")


def verify_latency(cells: list[Cell], *, progress: bool = True) -> list[CellResult]:
    """Measure latency **serially** with production threading, crash-isolated.

    Each cell runs one at a time (no cross-cell CPU contention → trustworthy
    timing) in its own un-pinned subprocess so rayon uses all cores as in
    deployment, and a native crash in one frontier config is contained to that
    cell rather than killing the tune. Each cell should set
    ``measure_latency=True``.
    """
    results: list[CellResult] = []
    for cell in tqdm(cells, disable=not progress, desc="verify-latency"):
        results.append(_run_latency_isolated(cell))
    return results
