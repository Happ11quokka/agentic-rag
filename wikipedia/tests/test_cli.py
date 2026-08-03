import json
from pathlib import Path

import pytest

from wikipedia import cli
from wikipedia.bundle import BundlePaths
from wikipedia.download import MODEL_REQUIRED_FILES


class FakeDatabase:
    def __init__(self, config: object) -> None:
        self.config = config
        self.flushed = 0

    def __enter__(self) -> "FakeDatabase":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def flush(self) -> None:
        self.flushed += 1

    def wait_for_index(self, **_: object) -> dict[str, object]:
        return {
            "index_type": "DISKANN",
            "state": "Finished",
            "total_rows": 0,
            "indexed_rows": 0,
            "pending_rows": 0,
            "reason": "",
        }


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


def _write_complete_model(paths: BundlePaths) -> None:
    for filename in MODEL_REQUIRED_FILES:
        destination = paths.model_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("complete", encoding="utf-8")
    (paths.model_dir / "pytorch_model.bin").write_bytes(b"weights")


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
    assert saved["float16"] is False
    assert saved["prefer_grpc"] is True
    assert saved["grpc_port"] == 6334
    assert len(runtime_calls) == 2
    assert runtime_calls[1]["storage_dir"] == storage.resolve()
    assert len(ingest_calls) == 2
    assert ingest_calls[1]["batch_size"] == 256
    assert ingest_calls[1]["max_workers"] == 4
    assert callable(ingest_calls[1]["worker_database_factory"])


def test_embedded_qdrant_keeps_ingest_serial_but_model_download_uses_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    ingest_calls: list[dict[str, object]] = []
    model_calls: list[dict[str, object]] = []
    events: list[str] = []

    def ingest(database: object, bundle_dir: Path, **kwargs: object) -> int:
        events.append("ingest")
        ingest_calls.append(kwargs)
        return 0

    def download(paths: BundlePaths, **kwargs: object) -> None:
        events.append("model")
        model_calls.append(kwargs)

    monkeypatch.setattr(
        cli.BundlePaths,
        "resolve",
        classmethod(lambda cls, bundle_dir=None: paths),
    )
    monkeypatch.setattr(cli, "prepare_bundle", lambda *args, **kwargs: _prepare(paths))
    monkeypatch.setattr(
        cli,
        "QdrantVectorDB",
        lambda config: events.append("database") or FakeDatabase(config),
    )
    monkeypatch.setattr(
        cli,
        "ingest_wikipedia",
        ingest,
    )
    monkeypatch.setattr(cli, "download_model", download)

    cli.main(
        [
            "qdrant",
            "--path",
            str(tmp_path / "embedded-qdrant"),
            "--max-workers",
            "7",
        ]
    )

    assert ingest_calls[0]["max_workers"] == 1
    assert ingest_calls[0]["worker_database_factory"] is None
    assert model_calls[0]["max_workers"] == 7
    assert events == ["model", "database", "ingest"]


def test_complete_matching_model_skips_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    paths.bundle_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "incomplete",
                "model": {"resolved_revision": "model-sha"},
            }
        ),
        encoding="utf-8",
    )
    _write_complete_model(paths)
    monkeypatch.setattr(
        cli.BundlePaths,
        "resolve",
        classmethod(lambda cls, bundle_dir=None: paths),
    )
    monkeypatch.setattr(cli, "prepare_bundle", lambda *args, **kwargs: _prepare(paths))
    monkeypatch.setattr(
        cli,
        "download_model",
        lambda *args, **kwargs: pytest.fail("complete model should be reused"),
    )
    monkeypatch.setattr(cli, "QdrantVectorDB", FakeDatabase)
    monkeypatch.setattr(cli, "ingest_wikipedia", lambda *args, **kwargs: 0)

    cli.main(["qdrant", "--path", str(tmp_path / "qdrant")])


@pytest.mark.parametrize(
    "recorded_model",
    [{"resolved_revision": "old-model-sha"}, None],
    ids=("changed", "unknown"),
)
def test_model_revision_mismatch_downloads_before_database_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_model: dict[str, str] | None,
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    paths.bundle_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "model": recorded_model,
            }
        ),
        encoding="utf-8",
    )
    _write_complete_model(paths)
    events: list[str] = []
    monkeypatch.setattr(
        cli.BundlePaths,
        "resolve",
        classmethod(lambda cls, bundle_dir=None: paths),
    )
    monkeypatch.setattr(cli, "prepare_bundle", lambda *args, **kwargs: _prepare(paths))
    monkeypatch.setattr(
        cli,
        "download_model",
        lambda *args, **kwargs: events.append("model"),
    )
    monkeypatch.setattr(
        cli,
        "QdrantVectorDB",
        lambda config: events.append("database") or FakeDatabase(config),
    )
    monkeypatch.setattr(
        cli,
        "ingest_wikipedia",
        lambda *args, **kwargs: events.append("ingest") or 0,
    )

    cli.main(["qdrant", "--path", str(tmp_path / "qdrant")])

    assert events == ["model", "database", "ingest"]


def test_model_download_failure_does_not_start_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    database_started = False
    monkeypatch.setattr(
        cli.BundlePaths,
        "resolve",
        classmethod(lambda cls, bundle_dir=None: paths),
    )
    monkeypatch.setattr(cli, "prepare_bundle", lambda *args, **kwargs: _prepare(paths))

    def fail_download(*args: object, **kwargs: object) -> None:
        raise RuntimeError("model transfer failed")

    def start_database(config: object) -> FakeDatabase:
        nonlocal database_started
        database_started = True
        return FakeDatabase(config)

    monkeypatch.setattr(cli, "download_model", fail_download)
    monkeypatch.setattr(cli, "QdrantVectorDB", start_database)

    with pytest.raises(RuntimeError, match="model transfer failed"):
        cli.main(["qdrant", "--path", str(tmp_path / "qdrant")])

    assert database_started is False


def test_default_max_workers_is_four() -> None:
    args = cli.build_parser().parse_args(["qdrant"])

    assert args.max_workers == 4


def test_keyboard_interrupt_exits_cleanly_with_status_130(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    model_downloaded = False
    monkeypatch.setattr(
        cli.BundlePaths,
        "resolve",
        classmethod(lambda cls, bundle_dir=None: paths),
    )
    monkeypatch.setattr(cli, "prepare_bundle", lambda *args, **kwargs: _prepare(paths))
    monkeypatch.setattr(cli, "ensure_qdrant", lambda *args, **kwargs: "ready")
    monkeypatch.setattr(cli, "QdrantVectorDB", FakeDatabase)

    def interrupt(*args: object, **kwargs: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "ingest_wikipedia", interrupt)

    def download_model(*args: object, **kwargs: object) -> None:
        nonlocal model_downloaded
        model_downloaded = True

    monkeypatch.setattr(cli, "download_model", download_model)

    with pytest.raises(SystemExit) as error:
        cli.main(["qdrant"])

    assert error.value.code == 130
    assert "incomplete downloads and shards retained" in capsys.readouterr().err
    assert model_downloaded is True


def test_milvus_keeps_ingest_serial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    ingest_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli.BundlePaths,
        "resolve",
        classmethod(lambda cls, bundle_dir=None: paths),
    )
    monkeypatch.setattr(cli, "prepare_bundle", lambda *args, **kwargs: _prepare(paths))
    monkeypatch.setattr(cli, "ensure_milvus", lambda *args, **kwargs: "ready")
    monkeypatch.setattr(cli, "MilvusVectorDB", FakeDatabase)
    monkeypatch.setattr(
        cli,
        "ingest_wikipedia",
        lambda database, bundle_dir, **kwargs: ingest_calls.append(kwargs) or 0,
    )
    monkeypatch.setattr(cli, "download_model", lambda *args, **kwargs: None)

    cli.main(["milvus", "--max-workers", "6"])

    assert ingest_calls[0]["max_workers"] == 1
    assert ingest_calls[0]["worker_database_factory"] is None


def test_qdrant_float16_is_saved_and_passed_to_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    configs: list[object] = []
    monkeypatch.setattr(
        cli.BundlePaths,
        "resolve",
        classmethod(lambda cls, bundle_dir=None: paths),
    )
    monkeypatch.setattr(cli, "prepare_bundle", lambda *args, **kwargs: _prepare(paths))
    monkeypatch.setattr(cli, "download_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "ensure_qdrant", lambda *args, **kwargs: "ready")
    monkeypatch.setattr(
        cli,
        "QdrantVectorDB",
        lambda config: configs.append(config) or FakeDatabase(config),
    )
    monkeypatch.setattr(cli, "ingest_wikipedia", lambda *args, **kwargs: 0)

    cli.main(["qdrant", "--float16"])

    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    assert manifest["qdrant"]["float16"] is True
    assert configs[0].float16 is True


@pytest.mark.parametrize(
    ("saved_float16", "arguments"),
    [
        (True, ["qdrant"]),
        (False, ["qdrant", "--float16"]),
    ],
)
def test_qdrant_float16_must_match_saved_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    saved_float16: bool,
    arguments: list[str],
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    paths.bundle_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_path.write_text(
        json.dumps({"qdrant": {"float16": saved_float16}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli.BundlePaths,
        "resolve",
        classmethod(lambda cls, bundle_dir=None: paths),
    )
    monkeypatch.setattr(cli, "prepare_bundle", lambda *args, **kwargs: _prepare(paths))

    with pytest.raises(SystemExit) as error:
        cli.main(arguments)

    assert error.value.code == 2
    assert "clean up existing Qdrant storage" in capsys.readouterr().err


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
