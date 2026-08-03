import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wikipedia import docker_runtime, inspect, qdrant_runtime
from wikipedia.bundle import BundlePaths
from wikipedia.types import SearchResult


class FakeEncoder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def encode(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.0] * 1024


class FakeDatabase:
    def __init__(self, config: object | None = None, *, has_collection: bool = True) -> None:
        self.client = self
        self.has_collection = has_collection
        self.search_count = 0

    def __enter__(self) -> "FakeDatabase":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def collection_exists(self, collection: str) -> bool:
        return self.has_collection

    def get_collection(self, collection: str) -> SimpleNamespace:
        return SimpleNamespace(points_count=47_018_430, status="green")

    def search(self, vector: object, *, limit: int = 5) -> list[SearchResult]:
        self.search_count += 1
        return [
            SearchResult(
                source_id="1",
                score=0.9,
                url="https://example.com",
                title="Aurora",
                text="Charged particles produce auroras.",
            )
        ][:limit]


def _summary(
    tmp_path: Path, *, complete: bool = True, model_bytes: int = 2 * 1024**3
) -> inspect.BundleSummary:
    return inspect.BundleSummary(
        paths=BundlePaths.from_dir(tmp_path),
        complete=complete,
        shard_count=471,
        completed_shards=471,
        dataset_bytes=0,
        model_bytes=model_bytes,
        qdrant={
            "url": qdrant_runtime.DEFAULT_QDRANT_URL,
            "storage_dir": str(tmp_path / "qdrant"),
            "container": qdrant_runtime.DEFAULT_QDRANT_CONTAINER,
            "image": qdrant_runtime.DEFAULT_QDRANT_IMAGE,
            "collection": inspect.DEFAULT_COLLECTION,
        },
    )


@pytest.mark.parametrize(
    ("status", "complete"), [("complete", True), ("incomplete", False)]
)
def test_inspect_bundle_accepts_incomplete_manifest_and_counts_checkpoint(
    tmp_path: Path, status: str, complete: bool
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    paths.bundle_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True)
    (paths.model_dir / "model.safetensors").write_bytes(b"model")
    checkpoint = paths.state_dir / "checkpoint.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(
        json.dumps({"completed_shards": ["dataset/data/en/000.parquet"]}),
        encoding="utf-8",
    )
    paths.manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": status,
                "dataset": {"shards": ["dataset/data/en/000.parquet"]},
                "qdrant": {"checkpoint": str(checkpoint)},
            }
        ),
        encoding="utf-8",
    )

    summary = inspect.inspect_bundle(tmp_path)

    assert summary.complete is complete
    assert summary.shard_count == 1
    assert summary.completed_shards == 1
    assert summary.dataset_bytes == 0
    assert summary.model_bytes == 5


def test_benchmark_query_runs_ten_times_and_reports_empty_results() -> None:
    database = FakeDatabase()
    ticks = iter(value / 1000 for value in range(0, 200, 10))

    latencies, results = inspect.benchmark_query(
        database,
        FakeEncoder(),
        "aurora",
        runs=10,
        limit=5,
        timer=lambda: next(ticks),
    )

    assert database.search_count == 10
    assert latencies == pytest.approx([10.0] * 10)
    assert results[0].title == "Aurora"

    database.search = lambda vector, limit: []  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="returned no chunks"):
        inspect.benchmark_query(database, FakeEncoder(), "missing", runs=1, limit=5)


def test_main_prints_bundle_database_latency_and_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    database = FakeDatabase()
    encoder = FakeEncoder()
    monkeypatch.setattr(
        inspect.BundlePaths,
        "resolve",
        classmethod(lambda cls, bundle_dir=None: BundlePaths.from_dir(tmp_path)),
    )
    monkeypatch.setattr(
        inspect, "inspect_bundle", lambda bundle_dir, **kwargs: _summary(tmp_path)
    )
    monkeypatch.setattr(
        inspect, "ensure_qdrant", lambda url, **kwargs: "already running"
    )
    monkeypatch.setattr(inspect, "QdrantVectorDB", lambda config: database)
    monkeypatch.setattr(
        inspect,
        "Encoder",
        lambda bundle_dir, *, require_complete: encoder,
    )

    inspect.main([])

    output = capsys.readouterr().out
    assert database.search_count == 10
    assert encoder.queries == [inspect.DEFAULT_QUERY] * 10
    assert "471/471 shards ingested" in output
    assert "points=47018430" in output
    assert "summary: runs=10" in output
    assert "title=Aurora" in output


def test_main_warns_and_queries_incomplete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    database = FakeDatabase()
    encoder = FakeEncoder()
    require_complete_values: list[bool] = []
    monkeypatch.setattr(
        inspect.BundlePaths,
        "resolve",
        classmethod(lambda cls, bundle_dir=None: BundlePaths.from_dir(tmp_path)),
    )
    monkeypatch.setattr(
        inspect,
        "inspect_bundle",
        lambda bundle_dir, **kwargs: _summary(tmp_path, complete=False),
    )
    monkeypatch.setattr(
        inspect, "ensure_qdrant", lambda url, **kwargs: "already running"
    )
    monkeypatch.setattr(inspect, "QdrantVectorDB", lambda config: database)

    def fake_encoder(bundle_dir: Path, *, require_complete: bool) -> FakeEncoder:
        require_complete_values.append(require_complete)
        return encoder

    monkeypatch.setattr(inspect, "Encoder", fake_encoder)

    inspect.main([])

    captured = capsys.readouterr()
    assert "bundle: incomplete, 471/471 shards ingested" in captured.out
    assert "warning: bundle is incomplete" in captured.err
    assert database.search_count == 10
    assert require_complete_values == [False]


def test_main_skips_query_when_incomplete_bundle_has_no_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    database = FakeDatabase()
    monkeypatch.setattr(
        inspect.BundlePaths,
        "resolve",
        classmethod(lambda cls, bundle_dir=None: BundlePaths.from_dir(tmp_path)),
    )
    monkeypatch.setattr(
        inspect,
        "inspect_bundle",
        lambda bundle_dir, **kwargs: _summary(
            tmp_path, complete=False, model_bytes=0
        ),
    )
    monkeypatch.setattr(
        inspect, "ensure_qdrant", lambda url, **kwargs: "already running"
    )
    monkeypatch.setattr(inspect, "QdrantVectorDB", lambda config: database)
    monkeypatch.setattr(
        inspect,
        "Encoder",
        lambda *args, **kwargs: pytest.fail("encoder should not load without a model"),
    )

    inspect.main([])

    captured = capsys.readouterr()
    assert "points=47018430" in captured.out
    assert "skipping query benchmark" in captured.err
    assert database.search_count == 0


def test_ensure_qdrant_reuses_running_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "qdrant"

    def fake_docker(*args: str) -> SimpleNamespace:
        stdout = str(storage.resolve()) if "--format" in args else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(qdrant_runtime, "_url_ready", lambda url: True)
    monkeypatch.setattr(qdrant_runtime.shutil, "which", lambda command: "docker")
    monkeypatch.setattr(qdrant_runtime, "_docker", fake_docker)

    assert (
        qdrant_runtime.ensure_qdrant(
            qdrant_runtime.DEFAULT_QDRANT_URL,
            storage_dir=storage,
        )
        == "already running"
    )


def test_ensure_qdrant_creates_bind_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_docker(*args: str) -> SimpleNamespace:
        calls.append(args)
        exists = args[:2] == ("container", "inspect")
        return SimpleNamespace(returncode=1 if exists else 0, stdout="", stderr="")

    monkeypatch.setattr(qdrant_runtime, "_url_ready", lambda url: False)
    monkeypatch.setattr(qdrant_runtime, "_wait_until_ready", lambda url: True)
    monkeypatch.setattr(
        qdrant_runtime.shutil, "which", lambda command: "/usr/local/bin/docker"
    )
    monkeypatch.setattr(qdrant_runtime, "_docker", fake_docker)

    storage = tmp_path / "qdrant"
    assert (
        qdrant_runtime.ensure_qdrant(
            qdrant_runtime.DEFAULT_QDRANT_URL,
            storage_dir=storage,
        )
        == "started"
    )
    run = next(args for args in calls if args and args[0] == "run")
    assert qdrant_runtime.DEFAULT_QDRANT_CONTAINER in run
    assert f"{storage.resolve()}:/qdrant/storage" in run


def test_ensure_qdrant_rejects_non_posix_storage_before_starting_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = tmp_path / "qdrant"
    monkeypatch.setattr(docker_runtime, "filesystem_type", lambda path: "exfat")
    monkeypatch.setattr(
        qdrant_runtime,
        "_ensure_docker",
        lambda: pytest.fail("Docker should not start for unsupported storage"),
    )

    with pytest.raises(RuntimeError, match="POSIX-compatible filesystem"):
        qdrant_runtime.ensure_qdrant(
            qdrant_runtime.DEFAULT_QDRANT_URL,
            storage_dir=storage,
        )

    assert not storage.exists()


def test_filesystem_type_reads_macos_mount_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        docker_runtime.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"/dev/disk4s1 on {tmp_path} (exfat, local, nodev)\n",
        ),
    )

    assert docker_runtime.filesystem_type(tmp_path / "qdrant") == "exfat"


def test_main_rejects_missing_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        inspect.BundlePaths,
        "resolve",
        classmethod(lambda cls, bundle_dir=None: BundlePaths.from_dir(tmp_path)),
    )
    monkeypatch.setattr(
        inspect, "inspect_bundle", lambda bundle_dir, **kwargs: _summary(tmp_path)
    )
    monkeypatch.setattr(
        inspect, "ensure_qdrant", lambda url, **kwargs: "already running"
    )
    monkeypatch.setattr(
        inspect,
        "QdrantVectorDB",
        lambda config: FakeDatabase(has_collection=False),
    )

    with pytest.raises(SystemExit) as error:
        inspect.main([])

    assert error.value.code == 2
    assert "collection does not exist" in capsys.readouterr().err
