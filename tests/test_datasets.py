"""Unit tests for ``tools/bench/dataset_registry.py`` (offline: no network, no real datasets)."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.bench import dataset_registry as dsm


def _write_manifest(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "datasets.toml"
    p.write_text(body)
    return p


def test_shipped_manifest_is_valid_and_pinned() -> None:
    m = dsm.load_manifest()
    assert {"hub", "icra2020-forward", "euroc", "liu4k"} <= set(m)
    for ds in m.values():
        assert ds.pin, f"{ds.name} has no pin"
        assert ds.license, f"{ds.name} has no license"
        assert ds.dest.startswith("tests/data/")


@pytest.mark.parametrize(
    ("entry", "msg"),
    [
        ('kind = "ftp"\ndest = "tests/data/x"\nlicense = "MIT"', "unknown kind"),
        (
            'kind = "hf-archive"\nrepo = "a/b"\nrevision = "main"\nfiles = ["x.zip"]\n'
            'ready = ["x"]\ndest = "tests/data/x"\nlicense = "MIT"',
            "40-char",
        ),
        (
            'kind = "url-zip"\nurl = "https://e/x.zip"\nready = ["x"]\ndest = "tests/data/x"\n'
            'license = "MIT"',
            "md5",
        ),
        (
            'kind = "url-zip"\nurl = "https://e/x.zip"\nmd5 = "0"\nready = ["x"]\n'
            'dest = "/etc/x"\nlicense = "MIT"',
            "tests/data",
        ),
    ],
)
def test_manifest_validation_rejects(tmp_path: Path, entry: str, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        dsm.load_manifest(_write_manifest(tmp_path, f"[bad]\n{entry}\n"))


# ── url-zip ──────────────────────────────────────────────────────────────────


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _url_manifest(tmp_path: Path, payload: bytes, md5: str | None = None) -> dict[str, dsm.Dataset]:
    md5 = md5 or hashlib.md5(payload, usedforsecurity=False).hexdigest()
    return dsm.load_manifest(
        _write_manifest(
            tmp_path,
            f'[toy]\nkind = "url-zip"\nurl = "https://example.invalid/toy.zip"\nmd5 = "{md5}"\n'
            'ready = ["toy"]\ndest = "tests/data/toy"\nlicense = "MIT"\n',
        )
    )


def test_url_zip_fetch_is_idempotent_and_stamped(tmp_path: Path) -> None:
    payload = _zip_bytes({"toy/a.txt": b"hello"})
    m = _url_manifest(tmp_path, payload)
    dest = tmp_path / "out"
    with patch("urllib.request.urlopen", return_value=io.BytesIO(payload)) as op:
        out = dsm.fetch("toy", dest=dest, manifest=m)
        dsm.fetch("toy", dest=dest, manifest=m)  # second call: already present
    assert op.call_count == 1
    assert op.call_args.kwargs["timeout"] == dsm.URL_TIMEOUT_S
    assert (out / "toy" / "a.txt").read_bytes() == b"hello"
    assert not (out / "toy.zip").exists()
    assert json.loads((out / dsm.STAMP).read_text())["toy"]["pin"] == m["toy"].pin


def test_url_zip_md5_mismatch_raises_and_cleans_up(tmp_path: Path) -> None:
    payload = _zip_bytes({"toy/a.txt": b"hello"})
    m = _url_manifest(tmp_path, payload, md5="0" * 32)
    dest = tmp_path / "out"
    with (
        patch("urllib.request.urlopen", return_value=io.BytesIO(payload)),
        pytest.raises(RuntimeError, match="md5 mismatch"),
    ):
        dsm.fetch("toy", dest=dest, manifest=m)
    assert list(dest.iterdir()) == []


def test_url_zip_reuses_verified_archive(tmp_path: Path) -> None:
    payload = _zip_bytes({"toy/a.txt": b"hello"})
    m = _url_manifest(tmp_path, payload)
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "toy.zip").write_bytes(payload)  # left behind by an earlier failed extraction
    with patch("urllib.request.urlopen") as op:
        dsm.fetch("toy", dest=dest, manifest=m)
    op.assert_not_called()
    assert (dest / "toy" / "a.txt").exists()


def test_interrupted_extraction_never_looks_ready(tmp_path: Path) -> None:
    payload = _zip_bytes({"toy/a.txt": b"hello"})
    m = _url_manifest(tmp_path, payload)
    dest = tmp_path / "out"

    def partial_then_crash(self: zipfile.ZipFile, path: Path, *a: object, **k: object) -> None:
        (Path(path) / "toy").mkdir()  # the marker directory appears in staging...
        raise OSError("disk full")  # ...and extraction dies

    with (
        patch("urllib.request.urlopen", return_value=io.BytesIO(payload)),
        patch.object(zipfile.ZipFile, "extractall", partial_then_crash),
        pytest.raises(OSError, match="disk full"),
    ):
        dsm.fetch("toy", dest=dest, manifest=m)
    assert not dsm.is_ready(m["toy"], dest)
    assert not any(p.name.startswith(".staging") for p in dest.iterdir())


# ── archive safety ───────────────────────────────────────────────────────────


def test_extract_rejects_zip_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    archive.write_bytes(_zip_bytes({"../escape.txt": b"x"}))
    (tmp_path / "out").mkdir()
    with pytest.raises(RuntimeError, match="unsafe"):
        dsm._extract_to(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_extract_rejects_tar_links_outside(tmp_path: Path, link_type: bytes) -> None:
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("link")
        info.type = link_type
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    (tmp_path / "out").mkdir()
    with pytest.raises(RuntimeError, match="unsafe"):
        dsm._extract_to(archive, tmp_path / "out")


# ── destinations ─────────────────────────────────────────────────────────────


def test_dest_honours_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = dsm.load_manifest()["hub"]
    assert hub.env == "LOCUS_HUB_DATASET_DIR"
    monkeypatch.setenv("LOCUS_HUB_DATASET_DIR", str(tmp_path))
    assert hub.dest_path() == tmp_path
    assert hub.dest_path(tmp_path / "explicit") == tmp_path / "explicit"
    monkeypatch.delenv("LOCUS_HUB_DATASET_DIR")
    assert hub.dest_path() == dsm.ROOT / hub.dest


# ── hub ──────────────────────────────────────────────────────────────────────


def test_hub_subsets_resolution() -> None:
    hub = dsm.load_manifest()["hub"]
    assert dsm.hub_subsets(hub, None) == list(hub.subsets)
    assert dsm.hub_subsets(hub, "a,b,") == ["a", "b"]
    with patch("datasets.get_dataset_config_names", return_value=["x", "y"]) as names:
        assert dsm.hub_subsets(hub, "all") == ["x", "y"]
    assert names.call_args.kwargs["revision"] == hub.revision


def test_hub_discovery_fallback_keeps_folders_only() -> None:
    from huggingface_hub.hf_api import RepoFile, RepoFolder

    hub = dsm.load_manifest()["hub"]
    tree = [
        RepoFolder(path="cfg_a", oid="1"),
        RepoFile(path="LICENSE", size=1, oid="2"),
        RepoFolder(path=".github", oid="3"),
        RepoFolder(path="cfg_b", oid="4"),
    ]
    with (
        patch("datasets.get_dataset_config_names", side_effect=ValueError("schema drift")),
        patch("huggingface_hub.HfApi.list_repo_tree", return_value=tree),
    ):
        assert dsm.hub_subsets(hub, "all") == ["cfg_a", "cfg_b"]


def test_hub_stamps_only_subsets_it_fetched(tmp_path: Path) -> None:
    hub = dsm.load_manifest()["hub"]
    (tmp_path / "have").mkdir()
    (tmp_path / "have" / "annotations.jsonl").write_text("")  # pre-existing, unpinned copy
    with patch("tools.bench.sync_hub.sync_subset_to_local") as sync:
        dsm.fetch("hub", dest=tmp_path, subsets="have,need", manifest={"hub": hub})
    sync.assert_called_once_with("need", tmp_path, hub.repo, revision=hub.revision)
    stamped = json.loads((tmp_path / dsm.STAMP).read_text())["hub"]["subsets"]
    assert set(stamped) == {"need"}
    assert stamped["need"]["pin"] == hub.pin


def test_hub_partial_failure_stamps_successes_and_raises(tmp_path: Path) -> None:
    hub = dsm.load_manifest()["hub"]

    def sync(subset: str, *_a: object, **_k: object) -> None:
        if subset == "bad":
            raise ConnectionError("HF 503")

    with (
        patch("tools.bench.sync_hub.sync_subset_to_local", side_effect=sync),
        pytest.raises(RuntimeError, match="1 subset"),
    ):
        dsm.fetch("hub", dest=tmp_path, subsets="good,bad,also_good", manifest={"hub": hub})
    stamped = json.loads((tmp_path / dsm.STAMP).read_text())["hub"]["subsets"]
    assert set(stamped) == {"good", "also_good"}


# ── verify ───────────────────────────────────────────────────────────────────


def test_verify_statuses(tmp_path: Path) -> None:
    m = _url_manifest(tmp_path, b"unused")
    ds = m["toy"]
    dest = tmp_path / "data"
    with patch.object(dsm.Dataset, "dest_path", lambda self, override=None: dest):
        assert [i.status for i in dsm.verify(["toy"], m)] == ["missing"]
        (dest / "toy").mkdir(parents=True)
        assert [i.status for i in dsm.verify(["toy"], m)] == ["unstamped"]
        dsm._write_stamp(ds, dest)
        assert dsm.verify(["toy"], m) == []
        stale = json.loads((dest / dsm.STAMP).read_text())
        stale["toy"]["pin"] = "md5:old"
        (dest / dsm.STAMP).write_text(json.dumps(stale))
        assert [i.status for i in dsm.verify(["toy"], m)] == ["stale"]


def test_verify_hub_is_per_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = dsm.load_manifest()["hub"]
    monkeypatch.setenv("LOCUS_HUB_DATASET_DIR", str(tmp_path))
    first, second, *rest = hub.subsets
    for s in (first, second):
        (tmp_path / s).mkdir()
        (tmp_path / s / "annotations.jsonl").write_text("")
    dsm._write_stamp(hub, tmp_path, subsets=[first])
    by_status: dict[str, set[str]] = {}
    for issue in dsm.verify(["hub"], {"hub": hub}):
        by_status.setdefault(issue.status, set()).add(issue.name)
    assert by_status["unstamped"] == {f"hub/{second}"}
    assert by_status["missing"] == {f"hub/{s}" for s in rest}
    assert f"hub/{first}" not in set().union(*by_status.values())


def test_verify_exit_code_follows_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dsm, "verify", lambda n, m: [dsm.Issue("x", "unstamped", "says 'stale' in prose")]
    )
    assert dsm.main(["verify", "liu4k"]) == 0
    monkeypatch.setattr(dsm, "verify", lambda n, m: [dsm.Issue("x", "stale", "says 'unstamped'")])
    assert dsm.main(["verify", "liu4k"]) == 1


# ── bench prepare ────────────────────────────────────────────────────────────


def test_bench_prepare_continues_past_a_failing_dataset() -> None:
    from tools.cli import bench_prepare

    calls = []

    def fetch(name: str, **_k: object) -> Path:
        calls.append(name)
        if name == "icra2020-forward":
            raise ConnectionError("offline")
        return Path()

    with patch("tools.bench.dataset_registry.fetch", side_effect=fetch):
        bench_prepare()
    assert calls == ["icra2020-forward", "icra2020-circle", "hub"]
