"""Dataset provisioning driven by ``xtask/datasets.toml`` (the single source of truth).

Usage (normally via ``cargo xtask data ...``)::

    python -m tools.bench.dataset_registry list
    python -m tools.bench.dataset_registry fetch euroc liu4k [--dest DIR] [--force]
    python -m tools.bench.dataset_registry fetch hub [--subsets all|a,b,c]
    python -m tools.bench.dataset_registry verify [name ...]

Every fetch is idempotent (skips datasets whose ``ready`` markers exist), pinned (Hugging
Face ``revision`` / URL checksum from the manifest) and records what it fetched in a
``.locus-dataset.json`` stamp inside ``dest``; ``verify`` compares that stamp with the
manifest so a local copy fetched from an older pin is reported instead of silently used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 fallback (tomli is in the lock for < 3.11)
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "xtask" / "datasets.toml"
STAMP = ".locus-dataset.json"
KINDS = ("hf-hub-subsets", "hf-archive", "url-zip")


@dataclass(frozen=True)
class Dataset:
    """One manifest entry (see the field reference at the top of ``xtask/datasets.toml``)."""

    name: str
    kind: str
    dest: str
    license: str
    description: str = ""
    repo: str | None = None
    revision: str | None = None
    files: tuple[str, ...] = ()
    subsets: tuple[str, ...] = ()
    url: str | None = None
    md5: str | None = None
    size: int | None = None
    ready: tuple[str, ...] = ()
    citation: str = ""
    source_url: str = ""
    consumers: tuple[str, ...] = field(default=())

    @property
    def pin(self) -> str:
        """The immutable identifier a local copy must match (revision or checksum)."""
        return self.revision or (f"md5:{self.md5}" if self.md5 else "")

    def dest_path(self, override: Path | None = None) -> Path:
        return override if override is not None else ROOT / self.dest


def load_manifest(path: Path = MANIFEST) -> dict[str, Dataset]:
    """Parse and validate the manifest; raises ``ValueError`` on an invalid entry."""
    with open(path, "rb") as f:
        raw: dict[str, dict[str, Any]] = tomllib.load(f)
    out = {}
    for name, entry in raw.items():
        tuples = {k: tuple(v) for k, v in entry.items() if isinstance(v, list)}
        ds = Dataset(name=name, **{**entry, **tuples})
        if ds.kind not in KINDS:
            raise ValueError(f"{name}: unknown kind {ds.kind!r} (expected one of {KINDS})")
        if ds.kind.startswith("hf-") and not (ds.repo and ds.revision and len(ds.revision) == 40):
            raise ValueError(
                f"{name}: Hugging Face sources need `repo` and a 40-char `revision` pin"
            )
        if ds.kind == "hf-archive" and not (ds.files and ds.ready):
            raise ValueError(f"{name}: hf-archive needs `files` and `ready`")
        if ds.kind == "url-zip" and not (ds.url and ds.md5 and ds.ready):
            raise ValueError(f"{name}: url-zip needs `url`, `md5` and `ready`")
        if not ds.dest.startswith("tests/data/"):
            raise ValueError(f"{name}: dest must live under tests/data/ (gitignored)")
        out[name] = ds
    return out


# ── stamps ───────────────────────────────────────────────────────────────────


def _read_stamp(dest: Path) -> dict[str, Any]:
    try:
        return json.loads((dest / STAMP).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_stamp(ds: Dataset, dest: Path, extra: dict[str, Any] | None = None) -> None:
    stamp = _read_stamp(dest)
    stamp[ds.name] = {
        "pin": ds.pin,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        stamp[ds.name].update(extra)
    (dest / STAMP).write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n")


def is_ready(ds: Dataset, dest: Path, subsets: list[str] | None = None) -> bool:
    if ds.kind == "hf-hub-subsets":
        return all((dest / s / "annotations.jsonl").exists() for s in (subsets or ds.subsets))
    return all((dest / r).exists() for r in ds.ready)


# ── extraction (path-traversal safe) ─────────────────────────────────────────


def _safe_extract(archive: Path, dest: Path) -> None:
    root = dest.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if not (root / member).resolve().is_relative_to(root):
                    raise RuntimeError(f"unsafe path in {archive.name}: {member}")
            zf.extractall(root)  # noqa: S202 - member paths validated above
        return
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            if not (root / member.name).resolve().is_relative_to(root) or member.issym():
                raise RuntimeError(f"unsafe member in {archive.name}: {member.name}")
        tf.extractall(root)  # noqa: S202 - members validated above


# ── fetchers ─────────────────────────────────────────────────────────────────


def _fetch_hf_archive(ds: Dataset, dest: Path) -> None:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415 - optional bench dependency

    assert ds.repo is not None
    for name in ds.files:
        print(f"[{ds.name}] downloading {ds.repo}@{ds.revision[:8] if ds.revision else '?'}:{name}")
        archive = Path(
            hf_hub_download(
                repo_id=ds.repo,
                filename=name,
                repo_type="dataset",
                revision=ds.revision,
                local_dir=str(dest),
            )
        )
        print(f"[{ds.name}] extracting {name}")
        _safe_extract(archive, dest)
        archive.unlink()
    shutil.rmtree(dest / ".cache", ignore_errors=True)  # huggingface_hub local_dir metadata


def _fetch_url_zip(ds: Dataset, dest: Path) -> None:
    assert ds.url is not None and ds.md5 is not None
    archive = dest / f"{ds.name}.zip"
    part = archive.with_suffix(".zip.part")
    h = hashlib.md5(usedforsecurity=False)
    size = f" ({ds.size / 1e9:.1f} GB)" if ds.size else ""
    print(f"[{ds.name}] downloading {ds.url}{size}, license {ds.license}")
    with urllib.request.urlopen(ds.url) as resp, open(part, "wb") as out:  # noqa: S310 - pinned https URL
        while chunk := resp.read(1 << 20):
            h.update(chunk)
            out.write(chunk)
    if h.hexdigest() != ds.md5:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{ds.name}: md5 mismatch, got {h.hexdigest()}, expected {ds.md5}")
    part.replace(archive)
    print(f"[{ds.name}] extracting")
    _safe_extract(archive, dest)
    archive.unlink()


def hub_subsets(ds: Dataset, spec: str | None) -> list[str]:
    """Resolve ``--subsets`` (``None`` = manifest default, ``all`` = every config, or a list)."""
    if spec is None:
        return list(ds.subsets)
    if spec != "all":
        return [s for s in spec.split(",") if s]
    assert ds.repo is not None
    import datasets  # noqa: PLC0415

    try:
        return list(datasets.get_dataset_config_names(ds.repo, revision=ds.revision))
    except Exception:  # config discovery can fail on schema drift; fall back to the tree
        from huggingface_hub import HfApi  # noqa: PLC0415

        tree = HfApi().list_repo_tree(ds.repo, repo_type="dataset", revision=ds.revision)
        return [
            f.path.rstrip("/")
            for f in tree
            if "/" not in f.path.rstrip("/")
            and not f.path.startswith(".")
            and f.path.lower() != "readme.md"
        ]


def _fetch_hub(ds: Dataset, dest: Path, subsets: list[str], force: bool) -> None:
    from tools.bench.sync_hub import sync_subset_to_local  # noqa: PLC0415

    assert ds.repo is not None
    for subset in subsets:
        if not force and (dest / subset / "annotations.jsonl").exists():
            continue
        sync_subset_to_local(subset, dest, ds.repo, revision=ds.revision)


def fetch(
    name: str,
    dest: Path | None = None,
    subsets: str | None = None,
    force: bool = False,
    manifest: dict[str, Dataset] | None = None,
) -> Path:
    """Fetch one dataset (idempotent) and return its destination directory."""
    ds = (manifest or load_manifest())[name]
    out = ds.dest_path(dest)
    out.mkdir(parents=True, exist_ok=True)
    if ds.kind == "hf-hub-subsets":
        resolved = hub_subsets(ds, subsets)
        if not force and is_ready(ds, out, resolved):
            print(f"[{name}] present at {out}")
            return out
        _fetch_hub(ds, out, resolved, force)
        prior = _read_stamp(out).get(name, {}).get("subsets", [])
        _write_stamp(ds, out, {"subsets": sorted(set(prior) | set(resolved))})
        return out
    if not force and is_ready(ds, out):
        print(f"[{name}] present at {out}")
        return out
    if ds.kind == "hf-archive":
        _fetch_hf_archive(ds, out)
    else:
        _fetch_url_zip(ds, out)
    if not is_ready(ds, out):
        raise RuntimeError(f"{name}: fetched but readiness markers {ds.ready} are missing in {out}")
    _write_stamp(ds, out)
    return out


def verify(names: list[str], manifest: dict[str, Dataset] | None = None) -> list[str]:
    """Return one problem line per dataset that is missing or not at the manifest's pin."""
    m = manifest or load_manifest()
    problems = []
    for name in names:
        ds = m[name]
        dest = ds.dest_path()
        if not is_ready(ds, dest):
            problems.append(f"{name}: missing (run `cargo xtask data fetch {name}`)")
            continue
        got = _read_stamp(dest).get(name, {}).get("pin")
        if got is None:
            problems.append(
                f"{name}: present but unstamped (fetched before pinning); refetch with --force to verify"
            )
        elif got != ds.pin:
            problems.append(
                f"{name}: local copy is at {got}, manifest pins {ds.pin}; refetch with --force"
            )
    return problems


def _list(m: dict[str, Dataset]) -> None:
    for ds in m.values():
        dest = ds.dest_path()
        state = "present" if is_ready(ds, dest) else "missing"
        print(f"{ds.name:16s} {state:8s} {ds.kind:15s} {ds.dest:28s} {ds.license}")
        if ds.description:
            print(f"{'':16s} {ds.description}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="cargo xtask data", description="Pinned dataset provisioning."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    f = sub.add_parser("fetch")
    f.add_argument("names", nargs="*")
    f.add_argument("--all", action="store_true")
    f.add_argument("--dest", type=Path, default=None)
    f.add_argument("--subsets", default=None, help="hub only: 'all' or comma-separated configs")
    f.add_argument("--force", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("names", nargs="*")
    args = ap.parse_args(argv)
    m = load_manifest()
    if args.cmd == "list":
        _list(m)
        return 0
    names = (
        list(m)
        if (args.cmd == "verify" and not args.names) or getattr(args, "all", False)
        else args.names
    )
    unknown = [n for n in names if n not in m]
    if unknown or not names:
        ap.error(
            f"unknown or missing dataset name(s): {unknown or '(none)'}; known: {', '.join(m)}"
        )
    if args.cmd == "fetch":
        if args.dest is not None and len(names) > 1:
            ap.error("--dest applies to a single dataset")
        for n in names:
            fetch(n, dest=args.dest, subsets=args.subsets, force=args.force, manifest=m)
        return 0
    problems = verify(names, m)
    for p in problems:
        print(p)
    hard = [p for p in problems if "unstamped" not in p]
    print(f"{len(names) - len(problems)}/{len(names)} datasets present at their pinned version")
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
