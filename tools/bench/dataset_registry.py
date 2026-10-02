"""Dataset provisioning driven by ``xtask/datasets.toml`` (the single source of truth).

Usage (normally via ``cargo xtask data ...``)::

    python -m tools.bench.dataset_registry list
    python -m tools.bench.dataset_registry fetch euroc liu4k [--dest DIR] [--force]
    python -m tools.bench.dataset_registry fetch hub [--subsets all|a,b,c]
    python -m tools.bench.dataset_registry verify [name ...]

Guarantees:

* **Pinned.** Hugging Face sources by repository commit, URLs by checksum.
* **Idempotent and atomic.** A dataset counts as present only through its readiness markers,
  and markers appear only on success: archives are extracted into a staging directory and
  renamed into place; a Hub subset's ``annotations.jsonl`` is renamed in last.
* **Auditable.** Each successful fetch records its pin in ``.locus-dataset.json`` inside the
  destination (per subset for the Hub). ``verify`` reports copies that are missing or fetched
  at another pin (exit 1), and copies that predate stamping (warning).
* **Overridable.** ``env`` names the variable that relocates a dataset (the same variables
  the Rust tests read); ``--dest`` overrides both.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 fallback (tomli is in the lock for < 3.11)
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "xtask" / "datasets.toml"
STAMP = ".locus-dataset.json"
KINDS = ("hf-hub-subsets", "hf-archive", "url-zip")
# Reserved top-level table: benchmark definitions (`tools/bench/sota/spec.py`), not datasets.
SOTA_TABLE = "sota"
URL_TIMEOUT_S = 60  # per socket operation; a stalled server fails instead of hanging


@dataclass(frozen=True)
class Dataset:
    """One manifest entry (see the field reference at the top of ``xtask/datasets.toml``)."""

    name: str
    kind: str
    dest: str
    license: str
    description: str = ""
    env: str | None = None
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
        """``override`` > ``$env`` > manifest ``dest``.

        A relative ``$env`` value resolves like the Rust test helpers
        (``tests/common/mod.rs``): against the repo root when that directory exists, else as
        given (cwd-relative) when that exists, else against the repo root (first fetch).
        """
        if override is not None:
            return override
        from_env = os.environ.get(self.env) if self.env else None
        if from_env:
            p = Path(from_env)
            if p.is_absolute():
                return p
            if (ROOT / p).is_dir() or not p.is_dir():
                return ROOT / p
            return p.resolve()
        return ROOT / self.dest


def load_manifest(path: Path = MANIFEST) -> dict[str, Dataset]:
    """Parse and validate the manifest; raises ``ValueError`` on an invalid entry."""
    with open(path, "rb") as f:
        raw: dict[str, dict[str, Any]] = tomllib.load(f)
    out = {}
    for name, entry in raw.items():
        if name == SOTA_TABLE:
            continue
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


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_stamp(ds: Dataset, dest: Path, subsets: list[str] | None = None) -> None:
    """Record ``ds.pin`` for this dataset, or only for the given Hub ``subsets``."""
    stamp = _read_stamp(dest)
    if subsets is None:
        stamp[ds.name] = {"pin": ds.pin, "fetched_at": _now()}
    else:
        entry = stamp.setdefault(ds.name, {}).setdefault("subsets", {})
        entry.update({s: {"pin": ds.pin, "fetched_at": _now()} for s in subsets})
    (dest / STAMP).write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n")


def _subset_ready(dest: Path, subset: str) -> bool:
    return (dest / subset / "annotations.jsonl").exists()


def is_ready(ds: Dataset, dest: Path, subsets: list[str] | None = None) -> bool:
    if ds.kind == "hf-hub-subsets":
        return all(_subset_ready(dest, s) for s in (subsets or ds.subsets))
    return all((dest / r).exists() for r in ds.ready)


# ── extraction (staged, path-traversal safe) ─────────────────────────────────


def _extract_to(archive: Path, staging: Path) -> None:
    """Extract ``archive`` into ``staging``, refusing anything that could escape it."""
    root = staging.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if not (root / member).resolve().is_relative_to(root):
                    raise RuntimeError(f"unsafe path in {archive.name}: {member}")
            zf.extractall(root)  # noqa: S202 - member paths validated above
        return
    with tarfile.open(archive) as tf:
        if hasattr(tarfile, "data_filter"):  # 3.12+, and backported to 3.10.12 / 3.11.4
            try:
                tf.extractall(root, filter="data")
            except tarfile.TarError as e:  # FilterError: links, devices, escaping paths
                raise RuntimeError(f"unsafe member in {archive.name}: {e}") from e
            return
        for m in tf.getmembers():  # pragma: no cover - interpreters without extraction filters
            unsafe = m.issym() or m.islnk() or m.isdev()
            if unsafe or not (root / m.name).resolve().is_relative_to(root):
                raise RuntimeError(f"unsafe member in {archive.name}: {m.name}")
        tf.extractall(root)  # noqa: S202 # pragma: no cover


def _install_archive(ds: Dataset, archive: Path, dest: Path) -> None:
    """Extract into ``dest/.staging-<name>`` then rename each top-level entry into ``dest``.

    Readiness markers therefore only ever appear complete; a crash leaves the staging
    directory (cleared on the next attempt), never a half-extracted dataset.
    """
    staging = dest / f".staging-{ds.name}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        _extract_to(archive, staging)
        for entry in staging.iterdir():
            target = dest / entry.name
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            entry.replace(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ── fetchers ─────────────────────────────────────────────────────────────────


def _fetch_hf_archive(ds: Dataset, dest: Path) -> None:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415 - optional bench dependency

    assert ds.repo is not None and ds.revision is not None
    for name in ds.files:
        print(f"[{ds.name}] downloading {ds.repo}@{ds.revision[:8]}:{name}")
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
        _install_archive(ds, archive, dest)
        archive.unlink()
    shutil.rmtree(dest / ".cache", ignore_errors=True)  # huggingface_hub local_dir metadata


def _md5(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _fetch_url_zip(ds: Dataset, dest: Path) -> None:
    assert ds.url is not None and ds.md5 is not None
    archive = dest / f"{ds.name}.zip"
    if archive.exists() and _md5(archive) == ds.md5:
        print(f"[{ds.name}] reusing verified {archive.name}")
    else:
        part = archive.with_suffix(".zip.part")
        h = hashlib.md5(usedforsecurity=False)
        size = f" ({ds.size / 1e9:.1f} GB)" if ds.size else ""
        print(f"[{ds.name}] downloading {ds.url}{size}, license {ds.license}")
        with (
            urllib.request.urlopen(ds.url, timeout=URL_TIMEOUT_S) as resp,  # noqa: S310 - pinned https
            open(part, "wb") as out,
        ):
            while chunk := resp.read(1 << 20):
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != ds.md5:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"{ds.name}: md5 mismatch, got {h.hexdigest()}, expected {ds.md5}")
        part.replace(archive)
    print(f"[{ds.name}] extracting")
    _install_archive(ds, archive, dest)
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
        from huggingface_hub.hf_api import RepoFolder  # noqa: PLC0415

        tree = HfApi().list_repo_tree(ds.repo, repo_type="dataset", revision=ds.revision)
        return [f.path for f in tree if isinstance(f, RepoFolder) and not f.path.startswith(".")]


def _fetch_hub(ds: Dataset, dest: Path, subsets: list[str], force: bool) -> None:
    """Sync each missing subset; stamp the successes; raise listing any failures."""
    from tools.bench.sync_hub import sync_subset_to_local  # noqa: PLC0415

    assert ds.repo is not None
    fetched, failed = [], []
    for subset in subsets:
        if not force and _subset_ready(dest, subset):
            continue
        try:
            sync_subset_to_local(subset, dest, ds.repo, revision=ds.revision)
            fetched.append(subset)
        except Exception as e:  # keep going: one broken config must not block the rest
            print(f"[{ds.name}] {subset}: FAILED: {e}")
            failed.append(subset)
    if fetched:
        _write_stamp(ds, dest, subsets=fetched)
    if failed:
        raise RuntimeError(f"{ds.name}: {len(failed)} subset(s) failed: {', '.join(failed)}")


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


# ── verify ───────────────────────────────────────────────────────────────────

Status = Literal["missing", "stale", "unstamped"]
FAILING: frozenset[Status] = frozenset({"missing", "stale"})


@dataclass(frozen=True)
class Issue:
    name: str
    status: Status
    detail: str


def verify(names: list[str], manifest: dict[str, Dataset] | None = None) -> list[Issue]:
    """One :class:`Issue` per dataset (or Hub subset) not present at the manifest's pin."""
    m = manifest or load_manifest()
    issues: list[Issue] = []
    for name in names:
        ds = m[name]
        dest = ds.dest_path()
        stamp = _read_stamp(dest).get(name, {})
        if ds.kind == "hf-hub-subsets":
            stamped: dict[str, Any] = stamp.get("subsets", {})
            for s in sorted(set(ds.subsets) | set(stamped)):
                label = f"{name}/{s}"
                pin = stamped.get(s, {}).get("pin")
                if not _subset_ready(dest, s):
                    issues.append(
                        Issue(
                            label,
                            "missing",
                            f"fetch with `cargo xtask data fetch {name} --subsets {s}`",
                        )
                    )
                elif pin is None:
                    issues.append(
                        Issue(
                            label, "unstamped", "predates stamping; refetch with --force to verify"
                        )
                    )
                elif pin != ds.pin:
                    issues.append(
                        Issue(
                            label,
                            "stale",
                            f"at {pin}, manifest pins {ds.pin}; refetch with --force",
                        )
                    )
            continue
        if not is_ready(ds, dest):
            issues.append(Issue(name, "missing", f"fetch with `cargo xtask data fetch {name}`"))
        elif "pin" not in stamp:
            issues.append(
                Issue(name, "unstamped", "predates stamping; refetch with --force to verify")
            )
        elif stamp["pin"] != ds.pin:
            issues.append(
                Issue(
                    name,
                    "stale",
                    f"at {stamp['pin']}, manifest pins {ds.pin}; refetch with --force",
                )
            )
    return issues


def _list(m: dict[str, Dataset]) -> None:
    for ds in m.values():
        dest = ds.dest_path()
        state = "present" if is_ready(ds, dest) else "missing"
        print(f"{ds.name:16s} {state:8s} {ds.kind:15s} {dest!s:28s} {ds.license}")
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
    want_all = (args.cmd == "verify" and not args.names) or getattr(args, "all", False)
    names = list(m) if want_all else args.names
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
    issues = verify(names, m)
    for i in issues:
        print(f"{i.name}: {i.status}: {i.detail}")
    failing = [i for i in issues if i.status in FAILING]
    print(
        f"{len(failing)} failing, {len(issues) - len(failing)} warning(s) across {len(names)} dataset(s)"
    )
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
