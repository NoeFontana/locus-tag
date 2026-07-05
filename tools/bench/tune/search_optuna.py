"""Optional optuna-backed sampling for the ``bayes`` strategy.

Isolated in its own module so the base tuner never imports optuna: ``search.py``
imports ``optuna_draws`` lazily and turns a missing ``optuna`` into a friendly
"install the [tune] extra" error. Install with ``pip install -e '.[tune]'``.

The parallel executor evaluates a *batch* of cells at once, which does not fit a
strictly sequential ask→evaluate→tell loop. So this uses optuna's sampler to draw
``n`` points from the (unconditioned) search distribution — a drop-in, seeded,
sampler-quality alternative to uniform random search. A future enhancement can
close the loop by feeding realized objectives back per round.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

# Module-level import (not lazy inside the generator) so that a missing optuna
# raises ImportError at *import time* of this module — which is inside the
# ``try`` in ``search.iter_param_draws`` and becomes the friendly "install the
# [tune] extra" message. A lazy import would defer the error to iteration, past
# that try, and leak a raw ImportError.
import optuna  # pyright: ignore[reportMissingImports]  # optional [tune] extra

from tools.bench.tune.space import SearchSpace


def optuna_draws(space: SearchSpace, n: int, seed: int) -> Iterator[dict[str, Any]]:
    """Yield ``n`` seeded optuna-sampled parameter draws for ``space``.

    Raises ``ImportError`` (surfaced by ``search.iter_param_draws`` as a friendly
    message) when the optional ``optuna`` dependency is absent.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    names = sorted(space.params)
    for _ in range(n):
        trial = study.ask()
        draw = {name: space.params[name].optuna_suggest(trial, name) for name in names}
        # No realized objective in the batch model; tell a constant so the study
        # stays consistent and the sampler advances deterministically.
        study.tell(trial, 0.0)
        yield draw
