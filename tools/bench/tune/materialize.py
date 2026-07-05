"""Turn a ``{path: value}`` parameter draw into a concrete library config.

For Locus this means building a validated ``locus.DetectorConfig``; the
``param_hash`` here is the stable join key that ties a config back to its metric
rows across the whole tuning pipeline.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

import locus

from tools.bench.tune.space import SearchSpace

# Scalar/JSON-native leaf types that are assigned verbatim. Anything else found
# at a leaf in ``model_dump()`` is treated as a PyO3 enum instance and a string
# value is coerced to the matching variant (see ``_set_dotted``).
_PLAIN_LEAF = (bool, int, float, str, type(None), dict, list)


def _set_dotted(tree: dict[str, Any], path: str, value: Any) -> None:
    """Set ``path`` (dotted) in a nested ``model_dump()`` dict, coercing enums.

    When the existing leaf is a PyO3 enum instance (e.g. ``CornerRefinementMode``)
    and ``value`` is a string, it is mapped to the matching variant via
    ``getattr(type(existing), value)`` — so JSON spaces can store enum choices as
    plain strings without a per-path registry.
    """
    parts = path.split(".")
    node = tree
    for key in parts[:-1]:
        node = node[key]
    leaf = parts[-1]
    if leaf not in node:
        raise KeyError(f"unknown config path {path!r} (no key {leaf!r})")
    existing = node[leaf]
    if isinstance(value, str) and not isinstance(existing, _PLAIN_LEAF):
        try:
            value = getattr(type(existing), value)
        except AttributeError as exc:
            raise ValueError(
                f"{path!r}: {value!r} is not a variant of {type(existing).__name__}"
            ) from exc
    node[leaf] = value


def materialize_locus(space: SearchSpace, param_values: dict[str, Any]) -> locus.DetectorConfig:
    """Build a validated ``DetectorConfig`` from a base profile + parameter draw.

    Reuses the proven mutate-dict-then-validate pattern (``tools/cli.py`` bench
    loop): dump the base profile, apply ``space.fixed`` then ``param_values`` by
    dotted path, and re-validate. ``param_values`` wins over ``fixed`` on conflict.
    """
    # ``base_profile`` is dynamic (from a JSON space), so it is not one of the
    # statically-known ``ProfileName`` literals; the cast documents that.
    base_profile = space.base_profile or "standard"
    tree = locus.DetectorConfig.from_profile(cast(Any, base_profile)).model_dump()
    for path, value in {**space.fixed, **param_values}.items():
        _set_dotted(tree, path, value)
    return locus.DetectorConfig.model_validate(tree)


def _jsonable(value: Any) -> Any:
    """Best-effort JSON-stable representation of a parameter value for hashing."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    # PyO3 enums and misc objects: stringify (repr is stable within a build).
    return str(value)


def param_hash(library: str, param_values: dict[str, Any]) -> str:
    """Stable short hash of ``(library, param_values)`` — the cross-table join key.

    Deterministic across processes/runs for the same inputs (sorted keys,
    canonical separators), so a config computed in a worker matches the same
    config referenced in the report.
    """
    payload = {
        "library": library,
        "params": {k: _jsonable(v) for k, v in sorted(param_values.items())},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=8).hexdigest()
