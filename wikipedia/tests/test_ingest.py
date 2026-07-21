import json
import os
import signal
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wikipedia import ingest
from wikipedia.bundle import BundlePaths


class FakeDatabase:
    dimension = 1024

    def __init__(self, paths: BundlePaths, *, short_ack: bool = False) -> None:
        self.paths = paths
        self.short_ack = short_ack
        self.ensure_calls = 0
        self.upserted = 0
        self.max_local_shards = 0

    def ensure_collection(self) -> None:
        self.ensure_calls += 1

    def upsert(self, records: list[Any]) -> int:
        local = list(self.paths.dataset_dir.glob("*.parquet"))
        self.max_local_shards = max(self.max_local_shards, len(local))
        acknowledged = len(records) - 1 if self.short_ack else len(records)
        self.upserted += max(acknowledged, 0)
        return acknowledged


def _write_shard(path: Path, rows: int = 2, prefix: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "id": [f"{prefix}{index}" for index in range(rows)],
                "url": [f"https://example.com/{index}" for index in range(rows)],
                "title": [f"Title {index}" for index in range(rows)],
                "text": [f"Text {index}" for index in range(rows)],
                "embedding": [[0.0] * 1024 for _ in range(rows)],
            }
        ),
        path,
    )


def _write_manifest(paths: BundlePaths, names: list[str]) -> None:
    paths.bundle_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "incomplete",
                "dataset": {
                    "resolved_revision": "dataset-sha",
                    "shards": [f"dataset/{name}" for name in names],
                },
            }
        ),
        encoding="utf-8",
    )


def _run(
    database: FakeDatabase,
    paths: BundlePaths,
    checkpoint: Path,
    **kwargs: Any,
) -> int:
    progress_interval = kwargs.pop("progress_interval", 0)
    return ingest.ingest_wikipedia(
        database,
        paths.bundle_dir,
        backend="qdrant",
        endpoint="http://localhost:6333",
        collection="wikipedia",
        metric="DOT",
        batch_size=1,
        checkpoint_path=checkpoint,
        progress_interval=progress_interval,
        **kwargs,
    )


def test_ingest_downloads_checkpoints_and_deletes_one_shard_at_a_time(
    tmp_path: Path,
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    names = ["data/en/part-000.parquet", "data/en/part-001.parquet"]
    _write_manifest(paths, names)
    checkpoint = paths.state_dir / "checkpoint.json"
    download_calls: list[str] = []

    def downloader(**kwargs: object) -> str:
        assert not list(paths.dataset_dir.glob("*.parquet"))
        filename = str(kwargs["filename"])
        destination = Path(kwargs["local_dir"]) / filename
        _write_shard(destination)
        download_calls.append(filename)
        return str(destination)

    database = FakeDatabase(paths)
    count = _run(database, paths, checkpoint, downloader=downloader)

    assert count == 4
    assert download_calls == names
    assert database.ensure_calls == 1
    assert database.max_local_shards == 1
    assert not list(paths.dataset_dir.glob("*.parquet"))
    completed = json.loads(checkpoint.read_text(encoding="utf-8"))["completed_shards"]
    assert completed == [f"dataset/{name}" for name in names]


def test_existing_multi_shard_bundle_is_pruned_to_one_before_ingest(
    tmp_path: Path,
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    names = ["data/en/part-000.parquet", "data/en/part-001.parquet"]
    _write_manifest(paths, names)
    for name in names:
        _write_shard(paths.bundle_dir / "dataset" / name)
    checkpoint = paths.state_dir / "checkpoint.json"
    downloaded: list[str] = []

    def downloader(**kwargs: object) -> str:
        assert not list(paths.dataset_dir.glob("*.parquet"))
        filename = str(kwargs["filename"])
        destination = Path(kwargs["local_dir"]) / filename
        _write_shard(destination)
        downloaded.append(filename)
        return str(destination)

    database = FakeDatabase(paths)
    assert _run(database, paths, checkpoint, downloader=downloader) == 4

    assert downloaded == [names[1]]
    assert database.max_local_shards == 1
    assert not list(paths.dataset_dir.glob("*.parquet"))


def test_parallel_ingest_bounds_shards_and_uses_dedicated_worker_databases(
    tmp_path: Path,
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    names = [f"data/en/part-{index:03d}.parquet" for index in range(4)]
    _write_manifest(paths, names)
    for index, name in enumerate(names):
        _write_shard(paths.bundle_dir / "dataset" / name, prefix=f"{index}-")
    checkpoint = paths.state_dir / "checkpoint.json"
    barrier = threading.Barrier(2, timeout=2)
    state_lock = threading.Lock()
    active_upserts = 0
    max_active_upserts = 0
    max_local_shards = 0
    closed = 0
    worker_threads: set[int] = set()
    downloaded: list[str] = []

    class WorkerDatabase:
        dimension = 1024

        def __init__(self) -> None:
            self.first_upsert = True

        def __enter__(self) -> "WorkerDatabase":
            return self

        def __exit__(self, *_: object) -> None:
            nonlocal closed
            with state_lock:
                closed += 1

        def upsert(self, records: list[Any]) -> int:
            nonlocal active_upserts, max_active_upserts, max_local_shards
            if self.first_upsert:
                self.first_upsert = False
                barrier.wait()
            with state_lock:
                worker_threads.add(threading.get_ident())
                active_upserts += 1
                max_active_upserts = max(max_active_upserts, active_upserts)
                max_local_shards = max(
                    max_local_shards,
                    len(list(paths.dataset_dir.glob("*.parquet"))),
                )
            try:
                time.sleep(0.01)
                return len(records)
            finally:
                with state_lock:
                    active_upserts -= 1

    def downloader(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        destination = Path(kwargs["local_dir"]) / filename
        _write_shard(destination, prefix=f"downloaded-{filename}-")
        with state_lock:
            downloaded.append(filename)
        return str(destination)

    database = FakeDatabase(paths)
    count = _run(
        database,
        paths,
        checkpoint,
        max_workers=2,
        worker_database_factory=WorkerDatabase,
        downloader=downloader,
    )

    assert count == 8
    assert database.ensure_calls == 1
    assert len(worker_threads) == 2
    assert max_active_upserts == 2
    assert max_local_shards <= 2
    assert closed == 2
    assert set(downloaded) == set(names[2:])
    assert not list(paths.dataset_dir.glob("*.parquet"))
    completed = json.loads(checkpoint.read_text(encoding="utf-8"))["completed_shards"]
    assert completed == [f"dataset/{name}" for name in names]


def test_parallel_ingest_uses_one_shared_progress_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    names = ["data/en/part-000.parquet", "data/en/part-001.parquet"]
    _write_manifest(paths, names)
    for index, name in enumerate(names):
        _write_shard(paths.bundle_dir / "dataset" / name, prefix=f"{index}-")
    heartbeat_calls: list[tuple[str, float]] = []

    @contextmanager
    def fake_heartbeat(
        label: str, interval: float, detail: object = None
    ) -> Any:
        heartbeat_calls.append((label, interval))
        yield

    class WorkerDatabase:
        dimension = 1024

        def __enter__(self) -> "WorkerDatabase":
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def upsert(self, records: list[Any]) -> int:
            return len(records)

    monkeypatch.setattr(ingest, "progress_heartbeat", fake_heartbeat)

    assert _run(
        FakeDatabase(paths),
        paths,
        paths.state_dir / "checkpoint.json",
        max_workers=2,
        worker_database_factory=WorkerDatabase,
        progress_interval=5,
    ) == 4

    assert heartbeat_calls.count(("parallel ingest", 5)) == 1
    worker_heartbeats = [
        call for call in heartbeat_calls if call[0].startswith("ingest shard")
    ]
    assert len(worker_heartbeats) == 2
    assert all(interval == 0 for _, interval in worker_heartbeats)


def test_parallel_record_limit_is_exact_and_retains_incomplete_shards(
    tmp_path: Path,
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    names = [f"data/en/part-{index:03d}.parquet" for index in range(3)]
    _write_manifest(paths, names)
    checkpoint = paths.state_dir / "checkpoint.json"
    state_lock = threading.Lock()
    upserted = 0

    class WorkerDatabase:
        dimension = 1024

        def __enter__(self) -> "WorkerDatabase":
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def upsert(self, records: list[Any]) -> int:
            nonlocal upserted
            with state_lock:
                upserted += len(records)
            return len(records)

    def downloader(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        destination = Path(kwargs["local_dir"]) / filename
        _write_shard(destination, prefix=f"{filename}-")
        return str(destination)

    count = _run(
        FakeDatabase(paths),
        paths,
        checkpoint,
        max_workers=3,
        worker_database_factory=WorkerDatabase,
        downloader=downloader,
        max_records=3,
    )

    assert count == 3
    assert upserted == 3
    retained = list(paths.dataset_dir.glob("*.parquet"))
    assert 1 <= len(retained) <= 3
    completed = (
        json.loads(checkpoint.read_text(encoding="utf-8"))["completed_shards"]
        if checkpoint.exists()
        else []
    )
    for relative in completed:
        assert not (paths.bundle_dir / relative).exists()


def test_failed_parallel_checkpoint_is_not_committed_by_another_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    names = ["data/en/part-000.parquet", "data/en/part-001.parquet"]
    relatives = [f"dataset/{name}" for name in names]
    _write_manifest(paths, names)
    for index, name in enumerate(names):
        _write_shard(
            paths.bundle_dir / "dataset" / name,
            rows=1,
            prefix=f"{index}-",
        )
    checkpoint = paths.state_dir / "checkpoint.json"
    barrier = threading.Barrier(2, timeout=2)
    original_atomic_json = ingest.atomic_json

    class WorkerDatabase:
        dimension = 1024

        def __enter__(self) -> "WorkerDatabase":
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def upsert(self, records: list[Any]) -> int:
            barrier.wait()
            if records[0].source_id.startswith("1-"):
                time.sleep(0.05)
            return len(records)

    def fail_first_checkpoint(path: Path, data: dict[str, Any]) -> None:
        if relatives[0] in data["completed_shards"]:
            raise OSError("checkpoint failed")
        original_atomic_json(path, data)

    monkeypatch.setattr(ingest, "atomic_json", fail_first_checkpoint)

    with pytest.raises(OSError, match="checkpoint failed"):
        _run(
            FakeDatabase(paths),
            paths,
            checkpoint,
            max_workers=2,
            worker_database_factory=WorkerDatabase,
        )

    completed = json.loads(checkpoint.read_text(encoding="utf-8"))["completed_shards"]
    assert completed == [relatives[1]]
    assert (paths.bundle_dir / relatives[0]).is_file()
    assert not (paths.bundle_dir / relatives[1]).exists()


def test_parallel_download_failure_stops_before_claiming_more_shards(
    tmp_path: Path,
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    names = [f"data/en/part-{index:03d}.parquet" for index in range(4)]
    _write_manifest(paths, names)
    checkpoint = paths.state_dir / "checkpoint.json"
    barrier = threading.Barrier(2, timeout=2)
    download_calls: list[str] = []
    calls_lock = threading.Lock()
    closed = 0

    class WorkerDatabase:
        dimension = 1024

        def __enter__(self) -> "WorkerDatabase":
            return self

        def __exit__(self, *_: object) -> None:
            nonlocal closed
            with calls_lock:
                closed += 1

        def upsert(self, records: list[Any]) -> int:
            return len(records)

    def downloader(**kwargs: object) -> str:
        filename = str(kwargs["filename"])
        with calls_lock:
            download_calls.append(filename)
        barrier.wait()
        if filename == names[0]:
            raise RuntimeError("download failed")
        destination = Path(kwargs["local_dir"]) / filename
        _write_shard(destination, prefix=f"{filename}-")
        return str(destination)

    with pytest.raises(RuntimeError, match="download failed"):
        _run(
            FakeDatabase(paths),
            paths,
            checkpoint,
            max_workers=2,
            worker_database_factory=WorkerDatabase,
            downloader=downloader,
        )

    assert set(download_calls) == set(names[:2])
    assert closed == 2
    assert not checkpoint.exists()


def test_keyboard_interrupt_stops_all_parallel_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    names = [f"data/en/part-{index:03d}.parquet" for index in range(4)]
    _write_manifest(paths, names)
    checkpoint = paths.state_dir / "checkpoint.json"
    all_started = threading.Event()
    release_downloads = threading.Event()
    state_lock = threading.Lock()
    download_calls: list[str] = []
    started = 0
    closed = 0
    exited = 0

    class WorkerDatabase:
        dimension = 1024

        def __enter__(self) -> "WorkerDatabase":
            return self

        def __exit__(self, *_: object) -> None:
            nonlocal exited
            with state_lock:
                exited += 1

        def close(self) -> None:
            nonlocal closed
            with state_lock:
                closed += 1

        def upsert(self, records: list[Any]) -> int:
            return len(records)

    def downloader(**kwargs: object) -> str:
        nonlocal started
        filename = str(kwargs["filename"])
        with state_lock:
            download_calls.append(filename)
            started += 1
            if started == 2:
                all_started.set()
        if not release_downloads.wait(2):
            raise TimeoutError("downloads were not cancelled")
        raise RuntimeError("download cancelled")

    def send_interrupt() -> None:
        if all_started.wait(2):
            os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(ingest, "cancel_transfers", release_downloads.set)
    interrupt = threading.Thread(target=send_interrupt, daemon=True)
    interrupt.start()

    with pytest.raises(KeyboardInterrupt):
        _run(
            FakeDatabase(paths),
            paths,
            checkpoint,
            max_workers=2,
            worker_database_factory=WorkerDatabase,
            downloader=downloader,
        )
    interrupt.join(timeout=2)

    assert set(download_calls) == set(names[:2])
    assert closed == 2
    assert exited == 2
    assert not checkpoint.exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_workers": 0}, "max_workers"),
        ({"max_workers": 2}, "worker_database_factory"),
    ],
)
def test_parallel_ingest_rejects_invalid_worker_configuration(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    _write_manifest(paths, ["data/en/part-000.parquet"])

    with pytest.raises(ValueError, match=message):
        _run(FakeDatabase(paths), paths, paths.state_dir / "checkpoint.json", **kwargs)


def test_record_limit_retains_and_replays_incomplete_shard(tmp_path: Path) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    name = "data/en/part-000.parquet"
    _write_manifest(paths, [name])
    checkpoint = paths.state_dir / "checkpoint.json"

    def downloader(**kwargs: object) -> str:
        destination = Path(kwargs["local_dir"]) / str(kwargs["filename"])
        _write_shard(destination)
        return str(destination)

    first = FakeDatabase(paths)
    assert _run(first, paths, checkpoint, downloader=downloader, max_records=1) == 1
    assert (paths.bundle_dir / "dataset" / name).is_file()
    assert not checkpoint.exists()

    second = FakeDatabase(paths)
    assert _run(second, paths, checkpoint, downloader=downloader) == 2
    assert not (paths.bundle_dir / "dataset" / name).exists()
    assert checkpoint.is_file()


def test_short_backend_ack_retains_shard_without_checkpoint(tmp_path: Path) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    name = "data/en/part-000.parquet"
    _write_manifest(paths, [name])
    checkpoint = paths.state_dir / "checkpoint.json"

    def downloader(**kwargs: object) -> str:
        destination = Path(kwargs["local_dir"]) / str(kwargs["filename"])
        _write_shard(destination)
        return str(destination)

    with pytest.raises(RuntimeError, match="acknowledged"):
        _run(
            FakeDatabase(paths, short_ack=True),
            paths,
            checkpoint,
            downloader=downloader,
        )

    assert (paths.bundle_dir / "dataset" / name).is_file()
    assert not checkpoint.exists()


def test_checkpoint_failure_retains_fully_ingested_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    name = "data/en/part-000.parquet"
    _write_manifest(paths, [name])
    shard = paths.bundle_dir / "dataset" / name
    _write_shard(shard)
    checkpoint = paths.state_dir / "checkpoint.json"

    def fail_checkpoint(*args: object) -> None:
        raise OSError("disk")

    monkeypatch.setattr(ingest, "atomic_json", fail_checkpoint)

    with pytest.raises(OSError, match="disk"):
        _run(FakeDatabase(paths), paths, checkpoint)

    assert shard.is_file()
    assert not checkpoint.exists()


def test_delete_failure_happens_after_completed_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    name = "data/en/part-000.parquet"
    _write_manifest(paths, [name])
    shard = paths.bundle_dir / "dataset" / name
    _write_shard(shard)
    checkpoint = paths.state_dir / "checkpoint.json"
    original_unlink = Path.unlink

    def fail_shard_delete(self: Path, *args: object, **kwargs: object) -> None:
        if self == shard:
            raise OSError("delete failed")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_shard_delete)

    with pytest.raises(OSError, match="delete failed"):
        _run(FakeDatabase(paths), paths, checkpoint)

    assert shard.is_file()
    completed = json.loads(checkpoint.read_text(encoding="utf-8"))["completed_shards"]
    assert completed == [f"dataset/{name}"]


def test_download_failure_leaves_partial_for_hugging_face_resume(tmp_path: Path) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    name = "data/en/part-000.parquet"
    _write_manifest(paths, [name])
    checkpoint = paths.state_dir / "checkpoint.json"
    partial = paths.bundle_dir / "dataset/.cache/huggingface/download/shard.incomplete"

    def fail_download(**kwargs: object) -> None:
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"partial")
        raise RuntimeError("network stopped")

    with pytest.raises(RuntimeError, match="network stopped"):
        _run(
            FakeDatabase(paths),
            paths,
            checkpoint,
            downloader=fail_download,
        )

    assert partial.is_file()
    assert not checkpoint.exists()


def test_completed_checkpoint_cleans_leftover_shard_without_redownload(
    tmp_path: Path,
) -> None:
    paths = BundlePaths.from_dir(tmp_path)
    name = "data/en/part-000.parquet"
    relative = f"dataset/{name}"
    _write_manifest(paths, [name])
    shard = paths.bundle_dir / relative
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(b"already ingested")
    checkpoint = paths.state_dir / "checkpoint.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_revision": "dataset-sha",
                "backend": "qdrant",
                "endpoint_fingerprint": ingest.endpoint_fingerprint(
                    "http://localhost:6333"
                ),
                "collection": "wikipedia",
                "dimension": 1024,
                "metric": "DOT",
                "completed_shards": [relative],
            }
        ),
        encoding="utf-8",
    )

    def unexpected_download(**kwargs: object) -> None:
        raise AssertionError("completed shard must not download")

    assert _run(
        FakeDatabase(paths), paths, checkpoint, downloader=unexpected_download
    ) == 0
    assert not shard.exists()
