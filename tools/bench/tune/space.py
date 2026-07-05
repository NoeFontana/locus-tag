"""Search-space specification — a serializable declaration of what to tune.

A :class:`SearchSpace` is a per-library declaration mapping *dotted paths* (into
that library's native config) to :class:`ParamSpec` distributions. The tuner
never needs to understand a parameter's semantics — only how to enumerate a grid
axis or draw a random value; the wrapper's ``from_params`` (and, for Locus,
``materialize_locus``) does the actual "set this value" step.

Path conventions (the dict key in ``SearchSpace.params``):
- **locus** — dotted path into ``DetectorConfig.model_dump()`` nested groups,
  e.g. ``"decoder.min_contrast"``, ``"quad.extraction_mode"``.
- **opencv_aruco** — attribute name on ``cv2.aruco.DetectorParameters``,
  e.g. ``"adaptiveThreshWinSizeMax"``.
- **apriltag** — kwarg name on ``pupil_apriltags.Detector``, e.g. ``"quad_decimate"``.

Categorical enum values are stored as plain strings (``"Erf"``, ``"EdLines"``)
and coerced to the library's native representation at materialization time.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

LibraryId = Literal["locus", "opencv_aruco", "apriltag"]
ParamKind = Literal["float", "int", "categorical", "bool"]

# Shipped default search spaces live alongside this module.
SPACES_DIR = Path(__file__).parent / "spaces"


class ParamSpec(BaseModel):
    """One tunable parameter: a distribution the tuner can grid or sample.

    The dotted path/attr name is the *key* under which this spec is stored in
    :attr:`SearchSpace.params`, so it is not repeated here.
    """

    model_config = ConfigDict(extra="forbid")

    kind: ParamKind
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: list[Any] | None = None
    # Explicit grid values; when set they win over low/high/step for grids.
    grid_values: list[Any] | None = None

    @model_validator(mode="after")
    def _check(self) -> ParamSpec:
        if self.kind in ("categorical", "bool"):
            if self.choices is None and self.kind == "categorical":
                raise ValueError("categorical ParamSpec requires 'choices'")
        else:  # float / int
            if self.grid_values is None and (self.low is None or self.high is None):
                raise ValueError(
                    f"{self.kind} ParamSpec requires 'low' and 'high' (or grid_values)"
                )
            if self.log and (self.low is not None and self.low <= 0):
                raise ValueError("log-scale ParamSpec requires low > 0")
        return self

    def _bool_choices(self) -> list[Any]:
        return self.choices if self.choices is not None else [False, True]

    def grid_axis(self) -> list[Any]:
        """Enumerate the discrete values this parameter takes in a grid sweep."""
        if self.kind == "categorical":
            assert self.choices is not None
            return list(self.choices)
        if self.kind == "bool":
            return self._bool_choices()
        if self.grid_values is not None:
            return list(self.grid_values)
        assert self.low is not None and self.high is not None
        step = self.step if self.step else (self.high - self.low)
        if step <= 0:
            return [self.low]
        values: list[Any] = []
        # Integer-count the steps to avoid float drift accumulating past `high`.
        n = int(math.floor((self.high - self.low) / step + 1e-9)) + 1
        for i in range(n):
            v = self.low + i * step
            values.append(int(round(v)) if self.kind == "int" else float(v))
        return values

    def sample(self, rng: random.Random) -> Any:
        """Draw one value for random search (log-uniform when ``log`` is set)."""
        if self.kind == "categorical":
            assert self.choices is not None
            return rng.choice(self.choices)
        if self.kind == "bool":
            return rng.choice(self._bool_choices())
        assert self.low is not None and self.high is not None
        if self.log:
            v = math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        else:
            v = rng.uniform(self.low, self.high)
        return int(round(v)) if self.kind == "int" else float(v)

    def optuna_suggest(self, trial: Any, name: str) -> Any:
        """Suggest a value via an optuna ``trial`` (used by the optional backend)."""
        if self.kind == "categorical":
            assert self.choices is not None
            return trial.suggest_categorical(name, self.choices)
        if self.kind == "bool":
            return trial.suggest_categorical(name, self._bool_choices())
        assert self.low is not None and self.high is not None
        if self.kind == "int":
            return trial.suggest_int(name, int(self.low), int(self.high), log=self.log)
        return trial.suggest_float(name, self.low, self.high, log=self.log)


class SearchSpace(BaseModel):
    """A per-library tunable space plus pinned (non-swept) overrides."""

    model_config = ConfigDict(extra="forbid")

    library: LibraryId
    # Locus profile to start from; ignored for competitors (they have no profiles).
    base_profile: str | None = None
    params: dict[str, ParamSpec]
    # Non-swept overrides pinned for fairness/determinism (e.g. apriltag nthreads=1).
    fixed: dict[str, Any] = {}
    # Incompatible parameter combinations to skip. Each entry is a partial
    # assignment; a draw is excluded if it matches ALL keys of any entry (e.g.
    # {"quad.extraction_mode": "EdLines", "decoder.refinement_mode": "Erf"} — a
    # combination the Locus config rejects). Lets grid/random avoid generating
    # cells that would only ever error.
    exclusions: list[dict[str, Any]] = []

    def _is_excluded(self, draw: dict[str, Any]) -> bool:
        return any(
            all(draw.get(key) == value for key, value in rule.items()) for rule in self.exclusions
        )

    def grid(self) -> Iterator[dict[str, Any]]:
        """Yield every non-excluded point in the Cartesian product of the axes.

        Keys are iterated in sorted order so the enumeration is deterministic.
        """
        names = sorted(self.params)
        axes = [self.params[n].grid_axis() for n in names]
        for combo in itertools.product(*axes):
            draw = dict(zip(names, combo, strict=True))
            if not self._is_excluded(draw):
                yield draw

    def grid_size(self) -> int:
        """Number of non-excluded grid points.

        Falls back to enumeration only when exclusions are present; otherwise it
        is the cheap product of axis lengths (the full grid can be enormous).
        """
        product = 1
        for name in self.params:
            product *= len(self.params[name].grid_axis())
        if not self.exclusions:
            return product
        return sum(1 for _ in self.grid())

    def random_draws(self, n: int, seed: int) -> Iterator[dict[str, Any]]:
        """Yield up to ``n`` seeded, non-excluded random draws.

        Excluded draws are resampled (bounded) rather than counted, so a space
        with exclusions still yields close to ``n`` valid configs. Reproducible
        for a given seed.
        """
        rng = random.Random(seed)
        names = sorted(self.params)
        yielded = 0
        attempts = 0
        cap = n * 50 + 100  # bound in case exclusions cover most of the space
        while yielded < n and attempts < cap:
            attempts += 1
            draw = {name: self.params[name].sample(rng) for name in names}
            if self._is_excluded(draw):
                continue
            yielded += 1
            yield draw


def resolve_space_ref(ref: str) -> Path:
    """Resolve a space reference to a JSON path.

    ``ref`` is either a shipped-space name (``"locus_default"`` →
    ``spaces/locus_default.json``) or an explicit filesystem path. This is the
    single resolver both the CLI and the parallel worker use so a child process
    can rebuild the identical space from a lightweight string.
    """
    p = Path(ref)
    if p.suffix == ".json" and p.exists():
        return p
    candidate = SPACES_DIR / f"{ref}.json"
    if candidate.exists():
        return candidate
    if p.exists():
        return p
    raise FileNotFoundError(f"search space not found: {ref!r} (looked in {SPACES_DIR})")


def load_space(ref: str) -> SearchSpace:
    """Load and validate a :class:`SearchSpace` from a name or path."""
    path = resolve_space_ref(ref)
    return SearchSpace.model_validate(json.loads(path.read_text()))
