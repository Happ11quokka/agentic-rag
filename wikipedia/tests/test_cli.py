import json
from pathlib import Path

import pytest

from wikipedia import cli
from wikipedia.bundle import BundlePaths


class FakeDatabase:
    def __init__(self, config: object) -> None:
        self.config = config

    def __enter__(self) -> "FakeDatabase":
        return self

    def __exit__(self, *_: object) -> None:
        pass


def _prepare(paths: BundlePaths) -> dict[str, object]:
    previous = (
        json.loads(paths.manifest_path.read_text(encoding="utf-8"))
        if paths.manifest_path.is_file()
        else {}
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "incomplete",
        "dataset": {
            "resolved_revision": "dataset-sha",
            "shards": ["dataset/data/en/part-000.parquet"],
        },
        "model": {"resolved_revision": "model-sha"},
    }
    if "qdrant" in previous:
        manifest["qdrant"] = previous["qdrant"]
    return manifest


def test_qdrant_ingest_saves_and_reuses_runtime_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    paths.bundle_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_path.write_text(
        json.dumps({"schema_version": 1, "status": "complete"}),
        encoding="utf-8",
    )
    storage = tmp_path / "qdrant"
    runtime_calls: list[dict[str, object]] = []
    ingest_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        cli.BundlePaths,
        "resolve",
        classmethod(lambda cls, bundle_dir=None: paths),
    )
    monkeypatch.setattr(cli, "prepare_bundle", lambda *args, **kwargs: _prepare(paths))
    monkeypatch.setattr(cli, "download_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli,
        "ensure_qdrant",
        lambda url, **kwargs: runtime_calls.append({"url": url, **kwargs}) or "ready",
    )
    monkeypatch.setattr(cli, "QdrantVectorDB", FakeDatabase)
    monkeypatch.setattr(
        cli,
        "ingest_wikipedia",
        lambda database, bundle_dir, **kwargs: ingest_calls.append(kwargs)
        or 123,
    )
    for name in ("QDRANT_URL", "QDRANT_COLLECTION", "QDRANT_IMAGE"):
        monkeypatch.delenv(name, raising=False)

    cli.main(["qdrant"])
    cli.main(["qdrant"])

    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    saved = manifest["qdrant"]
    assert saved["storage_dir"] == str(storage.resolve())
    assert saved["batch_size"] == 256
    assert saved["prefer_grpc"] is True
    assert saved["grpc_port"] == 6334
    assert len(runtime_calls) == 2
    assert runtime_calls[1]["storage_dir"] == storage.resolve()
    assert len(ingest_calls) == 2
    assert ingest_calls[1]["batch_size"] == 256


def test_bundle_dir_must_be_absolute() -> None:
    with pytest.raises(SystemExit) as relative:
        cli.main(["qdrant", "--bundle-dir", "relative/path"])
    assert relative.value.code == 2


def test_explicit_bundle_dir_is_absolute_and_remembered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remembered: list[Path] = []
    runtime_calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "write_bundle_marker", lambda path: remembered.append(path))
    monkeypatch.setattr(
        cli,
        "prepare_bundle",
        lambda paths, **kwargs: _prepare(paths),
    )
    monkeypatch.setattr(
        cli,
        "ensure_qdrant",
        lambda url, **kwargs: runtime_calls.append({"url": url, **kwargs}) or "ready",
    )
    monkeypatch.setattr(cli, "QdrantVectorDB", FakeDatabase)
    monkeypatch.setattr(cli, "ingest_wikipedia", lambda *args, **kwargs: 0)
    monkeypatch.setattr(cli, "download_model", lambda *args, **kwargs: None)

    bundle = tmp_path / "bundle"
    cli.main(["qdrant", "--bundle-dir", str(bundle)])

    assert remembered == [bundle.resolve()]
    assert runtime_calls[0]["storage_dir"] == (bundle / "qdrant").resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["qdrant"]["storage_dir"] == str((bundle / "qdrant").resolve())
