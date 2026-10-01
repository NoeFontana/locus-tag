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
    assert (out / "toy" / "a.txt").read_bytes() == b"hello"
    assert not (out / "toy.zip").exists()
    stamp = json.loads((out / dsm.STAMP).read_text())
    assert stamp["toy"]["pin"] == m["toy"].pin


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


def test_safe_extract_rejects_zip_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    archive.write_bytes(_zip_bytes({"../escape.txt": b"x"}))
    (tmp_path / "out").mkdir()
    with pytest.raises(RuntimeError, match="unsafe"):
        dsm._safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_rejects_tar_symlink(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
    (tmp_path / "out").mkdir()
    with pytest.raises(RuntimeError, match="unsafe"):
        dsm._safe_extract(archive, tmp_path / "out")


def test_verify_reports_missing_unstamped_and_stale(tmp_path: Path) -> None:
    payload = _zip_bytes({"toy/a.txt": b"hello"})
    m = _url_manifest(tmp_path, payload)
    ds = m["toy"]
    dest = tmp_path / "data"
    with patch.object(dsm.Dataset, "dest_path", lambda self, override=None: dest):
        assert "missing" in dsm.verify(["toy"], m)[0]
        (dest / "toy").mkdir(parents=True)
        assert "unstamped" in dsm.verify(["toy"], m)[0]
        dsm._write_stamp(ds, dest)
        assert dsm.verify(["toy"], m) == []
        stale = json.loads((dest / dsm.STAMP).read_text())
        stale["toy"]["pin"] = "md5:old"
        (dest / dsm.STAMP).write_text(json.dumps(stale))
        assert "manifest pins" in dsm.verify(["toy"], m)[0]


def test_hub_subsets_resolution() -> None:
    hub = dsm.load_manifest()["hub"]
    assert dsm.hub_subsets(hub, None) == list(hub.subsets)
    assert dsm.hub_subsets(hub, "a,b,") == ["a", "b"]
    with patch("datasets.get_dataset_config_names", return_value=["x", "y"]) as names:
        assert dsm.hub_subsets(hub, "all") == ["x", "y"]
    assert names.call_args.kwargs["revision"] == hub.revision


def test_hub_fetch_passes_pin_and_skips_present(tmp_path: Path) -> None:
    hub = dsm.load_manifest()["hub"]
    (tmp_path / "have").mkdir()
    (tmp_path / "have" / "annotations.jsonl").write_text("")
    with patch("tools.bench.sync_hub.sync_subset_to_local") as sync:
        dsm.fetch("hub", dest=tmp_path, subsets="have,need", manifest={"hub": hub})
    sync.assert_called_once_with("need", tmp_path, hub.repo, revision=hub.revision)
    assert json.loads((tmp_path / dsm.STAMP).read_text())["hub"]["subsets"] == ["have", "need"]
