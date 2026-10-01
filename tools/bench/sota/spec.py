"""Benchmark definitions for ``cargo xtask sota`` (the ``[sota.*]`` tables of ``xtask/datasets.toml``).

xtask stays dependency-free, so it does not parse TOML: it asks this module instead.

Usage::

    python -m tools.bench.sota.spec list
    python -m tools.bench.sota.spec show <name>                 # key=value lines for xtask
    python -m tools.bench.sota.spec images <name> <stride> <out.txt>
    python -m tools.bench.sota.spec fetch <name>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.bench.dataset_registry import MANIFEST, SOTA_TABLE, Dataset, fetch, load_manifest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 fallback (tomli is in the lock for < 3.11)
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

SCORERS = ("liu4k", "euroc", "gt-csv", "gt-hub")
CONVENTIONS = ("opencv", "locus")


@dataclass(frozen=True)
class Spec:
    """One ``[sota.<name>]`` table (field reference: ``xtask/datasets.toml``)."""

    name: str
    data: str
    images: str
    family: str
    opencv_dict: str
    border_bits: int
    scorer: str
    gt_convention: str
    subset: str | None = None
    gt: str | None = None
    apriltag_family: str | None = None
    unsupported: dict[str, str] = field(default_factory=dict)

    def root(self, registry: dict[str, Dataset] | None = None) -> Path:
        """Directory that ``images`` and ``gt`` are relative to."""
        dest = (registry or load_manifest())[self.data].dest_path()
        return dest / self.subset if self.subset else dest

    def gt_path(self, registry: dict[str, Dataset] | None = None) -> Path:
        if self.gt is None:
            raise ValueError(f"{self.name}: scorer {self.scorer} has no `gt` file")
        return self.root(registry) / self.gt

    def image_paths(self, registry: dict[str, Dataset] | None = None) -> list[Path]:
        return sorted(self.root(registry).glob(self.images))


def load_specs(
    path: Path = MANIFEST, registry: dict[str, Dataset] | None = None
) -> dict[str, Spec]:
    """Parse and validate every ``[sota.*]`` table; raises ``ValueError`` on an invalid one."""
    with open(path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f).get(SOTA_TABLE, {})
    registry = registry if registry is not None else load_manifest(path)
    out = {}
    for name, entry in raw.items():
        spec = Spec(name=name, **entry)
        if spec.data not in registry:
            raise ValueError(f"sota.{name}: unknown dataset {spec.data!r}")
        hub = registry[spec.data].kind == "hf-hub-subsets"
        if hub != (spec.subset is not None):
            raise ValueError(f"sota.{name}: `subset` is required for, and only for, hub datasets")
        if spec.scorer not in SCORERS:
            raise ValueError(f"sota.{name}: unknown scorer {spec.scorer!r} (one of {SCORERS})")
        if spec.scorer.startswith("gt-") and not spec.gt:
            raise ValueError(f"sota.{name}: scorer {spec.scorer} needs `gt`")
        if spec.gt_convention not in CONVENTIONS:
            raise ValueError(f"sota.{name}: gt_convention must be one of {CONVENTIONS}")
        if spec.border_bits < 1:
            raise ValueError(f"sota.{name}: border_bits must be >= 1")
        out[name] = spec
    return out


def show(spec: Spec) -> str:
    """``key=value`` lines (values never contain newlines) for the Rust orchestrator."""
    lines = [
        f"family={spec.family}",
        f"opencv_dict={spec.opencv_dict}",
        f"apriltag_family={spec.apriltag_family or ''}",
        f"border_bits={spec.border_bits}",
    ]
    lines += [f"unsupported.{k}={' '.join(v.split())}" for k, v in spec.unsupported.items()]
    return "\n".join(lines) + "\n"


def write_image_list(spec: Spec, stride: int, out: Path) -> int:
    paths = spec.image_paths()[:: max(1, stride)]
    if not paths:
        raise SystemExit(
            f"{spec.name}: no images match {spec.root() / spec.images} "
            f"(run `cargo xtask sota fetch {spec.name}`)"
        )
    out.write_text("".join(f"{p}\n" for p in paths))
    return len(paths)


def main(argv: list[str]) -> None:
    specs = load_specs()
    match argv:
        case ["list"]:
            print("\n".join(specs))
        case ["show", name]:
            sys.stdout.write(show(_get(specs, name)))
        case ["images", name, stride, out]:
            print(write_image_list(_get(specs, name), int(stride), Path(out)))
        case ["fetch", name]:
            spec = _get(specs, name)
            fetch(spec.data, subsets=spec.subset)
        case _:
            raise SystemExit(__doc__)


def _get(specs: dict[str, Spec], name: str) -> Spec:
    if name not in specs:
        raise SystemExit(f"unknown sota dataset {name!r}; known: {', '.join(specs)}")
    return specs[name]


if __name__ == "__main__":
    main(sys.argv[1:])
