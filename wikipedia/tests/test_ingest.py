import json
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


def _write_shard(path: Path, rows: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "id": [str(index) for index in range(rows)],
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
    return ingest.ingest_wikipedia(
        database,
        paths.bundle_dir,
        backend="qdrant",
        endpoint="http://localhost:6333",
        collection="wikipedia",
        metric="DOT",
        batch_size=1,
        checkpoint_path=checkpoint,
        progress_interval=0,
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
