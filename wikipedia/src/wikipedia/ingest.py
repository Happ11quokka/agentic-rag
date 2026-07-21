from __future__ import annotations

import hashlib
import json
import queue
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .base import VectorDB
from .bundle import BundlePaths, atomic_json, load_manifest
from .download import (
    cancel_transfers,
    download_dataset_shard,
    progress_heartbeat,
    status,
)
from .types import WikipediaRecord

REQUIRED_COLUMNS = ("id", "url", "title", "text", "embedding")
EMBEDDING_DIMENSION = 1024


class CheckpointError(RuntimeError):
    pass


def endpoint_fingerprint(endpoint: str) -> str:
    return hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:16]


def default_checkpoint_path(
    paths: BundlePaths, backend: str, collection: str, endpoint: str
) -> Path:
    safe_collection = "".join(character if character.isalnum() else "_" for character in collection)
    return paths.state_dir / f"{backend}-{safe_collection}-{endpoint_fingerprint(endpoint)}.json"


def _load_checkpoint(path: Path, expected: dict[str, Any]) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"Malformed ingestion checkpoint {path}: {exc}") from exc
    actual = {key: data.get(key) for key in expected}
    if actual != expected:
        raise CheckpointError(
            f"Checkpoint fingerprint mismatch at {path}; expected {expected}, found {actual}"
        )
    completed = data.get("completed_shards", [])
    if not isinstance(completed, list):
        raise CheckpointError(f"Malformed completed_shards in {path}")
    return {str(item) for item in completed}


def ingest_wikipedia(
    database: VectorDB,
    bundle_dir: str | Path | None = None,
    *,
    backend: str,
    endpoint: str,
    collection: str,
    metric: str,
    batch_size: int = 256,
    checkpoint_path: str | Path | None = None,
    max_shards: int | None = None,
    max_records: int | None = None,
    progress_interval: float = 30,
    download_timeout: int | None = None,
    token: str | None = None,
    downloader: Any = None,
    max_workers: int = 1,
    worker_database_factory: Callable[[], VectorDB] | None = None,
) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_shards is not None and max_shards < 1:
        raise ValueError("max_shards must be at least 1")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be at least 1")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if max_workers > 1 and worker_database_factory is None:
        raise ValueError("worker_database_factory is required when max_workers exceeds 1")

    paths = BundlePaths.resolve(bundle_dir)
    if progress_interval < 0:
        raise ValueError("progress_interval must not be negative")

    manifest = load_manifest(paths, require_complete=False)
    declared = manifest.get("dataset", {}).get("shards", [])
    if not isinstance(declared, list) or not declared:
        raise ValueError("Manifest contains no dataset shards")
    shards = sorted((paths.bundle_dir / str(item)).resolve() for item in declared)
    shards = shards[:max_shards]

    fingerprint = {
        "schema_version": 1,
        "dataset_revision": str(
            manifest.get("dataset", {}).get("resolved_revision", "")
        ),
        "backend": backend,
        "endpoint_fingerprint": endpoint_fingerprint(endpoint),
        "collection": collection,
        "dimension": database.dimension,
        "metric": metric.upper(),
    }
    checkpoint = checkpoint_path or default_checkpoint_path(
        paths, backend, collection, endpoint
    )
    checkpoint = Path(checkpoint).expanduser().resolve()
    completed = _load_checkpoint(checkpoint, fingerprint)
    pending: list[tuple[Path, str, str]] = []
    selected_relatives: set[str] = set()
    for shard in shards:
        try:
            relative = str(shard.relative_to(paths.bundle_dir))
            dataset_filename = str(shard.relative_to(paths.bundle_dir / "dataset"))
        except ValueError as exc:
            raise ValueError(
                f"Dataset shard is outside bundle directory: {shard}"
            ) from exc
        selected_relatives.add(relative)
        if relative in completed:
            shard.unlink(missing_ok=True)
        else:
            pending.append((shard, relative, dataset_filename))

    existing = [item for item in pending if item[0].is_file()]
    retained_items = existing[:max_workers]
    retained_paths = {item[0] for item in retained_items}
    removed = 0
    for local_shard in paths.dataset_dir.rglob("*.parquet"):
        if local_shard.resolve() not in retained_paths:
            local_shard.unlink()
            removed += 1
    if removed:
        status(
            f"deleted {removed} surplus local shards to enforce "
            f"{max_workers}-worker storage bound"
        )

    retained_relatives = {item[1] for item in retained_items}
    pending = retained_items + [
        item for item in pending if item[1] not in retained_relatives
    ]

    database.ensure_collection()
    if not pending:
        return 0
    ingested = 0
    reserved = 0
    state_lock = threading.Lock()
    checkpoint_lock = threading.Lock()
    active_lock = threading.Lock()
    stop = threading.Event()
    quota_reached = threading.Event()
    active_shards: set[str] = set()
    work: queue.Queue[tuple[int, Path, str, str]] = queue.Queue()
    for shard_index, (shard, relative, dataset_filename) in enumerate(pending, 1):
        work.put((shard_index, shard, relative, dataset_filename))

    def current_ingested() -> int:
        with state_lock:
            return ingested

    def reserve_records(size: int) -> int:
        nonlocal reserved
        with state_lock:
            if stop.is_set():
                return 0
            if max_records is None:
                return size
            remaining = max_records - ingested - reserved
            claimed = min(size, max(remaining, 0))
            reserved += claimed
            if ingested + reserved >= max_records:
                quota_reached.set()
                stop.set()
            return claimed

    def release_reservation(size: int) -> None:
        nonlocal reserved
        if max_records is None:
            return
        with state_lock:
            reserved -= size

    def acknowledge(size: int) -> None:
        nonlocal ingested, reserved
        with state_lock:
            if max_records is not None:
                reserved -= size
            ingested += size

    def process_shard(
        worker_database: VectorDB,
        shard_index: int,
        shard: Path,
        relative: str,
        dataset_filename: str,
    ) -> None:
        worker_progress_interval = 0 if max_workers > 1 else progress_interval
        if not shard.is_file():
            download_dataset_shard(
                paths,
                dataset_filename,
                revision=fingerprint["dataset_revision"],
                progress_interval=worker_progress_interval,
                download_timeout=download_timeout,
                token=token,
                downloader=downloader,
            )
        else:
            status(f"reusing downloaded shard {dataset_filename}")

        parquet = pq.ParquetFile(shard)
        names = set(parquet.schema_arrow.names)
        missing_columns = set(REQUIRED_COLUMNS) - names
        if missing_columns:
            parquet.close()
            raise ValueError(
                f"{shard.name} missing required columns: {sorted(missing_columns)}"
            )

        limited = False
        shard_progress = {"records": 0}
        shard_records = parquet.metadata.num_rows

        def detail() -> str:
            return (
                f"{shard_progress['records']:,}/{shard_records:,} shard records, "
                f"{current_ingested():,} records this run"
            )

        status(
            f"ingesting shard {shard_index}/{len(pending)}: {dataset_filename}, "
            f"{shard_records:,} records"
        )
        try:
            with progress_heartbeat(
                f"ingest shard {shard_index}/{len(pending)}",
                worker_progress_interval,
                detail,
            ):
                for batch in parquet.iter_batches(
                    batch_size=batch_size, columns=list(REQUIRED_COLUMNS)
                ):
                    if stop.is_set():
                        limited = True
                        break
                    values: dict[str, list[Any]] = {
                        name: batch.column(index).to_pylist()
                        for index, name in enumerate(REQUIRED_COLUMNS)
                    }
                    records: list[WikipediaRecord] = []
                    for row in range(batch.num_rows):
                        source_id = str(values["id"][row] or "")
                        if not source_id:
                            raise ValueError(f"{shard.name} contains an empty source id")
                        embedding = values["embedding"][row]
                        if embedding is None or len(embedding) != EMBEDDING_DIMENSION:
                            size = "null" if embedding is None else len(embedding)
                            raise ValueError(
                                f"{shard.name} source {source_id!r} has embedding "
                                f"dimension {size}; expected {EMBEDDING_DIMENSION}"
                            )
                        records.append(
                            WikipediaRecord(
                                source_id=source_id,
                                url=str(values["url"][row] or ""),
                                title=str(values["title"][row] or ""),
                                text=str(values["text"][row] or ""),
                                embedding=embedding,
                            )
                        )

                    claimed = reserve_records(len(records))
                    if claimed == 0:
                        limited = True
                        break
                    if claimed < len(records):
                        records = records[:claimed]
                        limited = True

                    try:
                        acknowledged = worker_database.upsert(records)
                    except BaseException:
                        release_reservation(claimed)
                        raise
                    if acknowledged != claimed:
                        release_reservation(claimed)
                        raise RuntimeError(
                            f"Backend acknowledged {acknowledged} of {claimed} "
                            f"records in {relative}"
                        )
                    acknowledge(acknowledged)
                    shard_progress["records"] += acknowledged
                    if max_records is not None and current_ingested() >= max_records:
                        limited = True
                        break
        finally:
            parquet.close()

        if limited and shard_progress["records"] == shard_records:
            limited = False
        if limited:
            reason = (
                "record limit reached"
                if quota_reached.is_set()
                else "ingestion stopped"
            )
            status(f"{reason}; retaining incomplete shard {dataset_filename}")
            return
        with checkpoint_lock:
            updated = completed | {relative}
            atomic_json(
                checkpoint,
                {**fingerprint, "completed_shards": sorted(updated)},
            )
            completed.add(relative)
        shard.unlink()
        status(
            f"shard {shard_index}/{len(pending)} checkpointed and deleted: "
            f"{dataset_filename}"
        )

    def consume(worker_database: VectorDB) -> None:
        while not stop.is_set():
            try:
                shard_index, shard, relative, dataset_filename = work.get_nowait()
            except queue.Empty:
                return
            with active_lock:
                active_shards.add(relative)
            try:
                process_shard(
                    worker_database,
                    shard_index,
                    shard,
                    relative,
                    dataset_filename,
                )
            finally:
                with active_lock:
                    active_shards.discard(relative)

    if max_workers == 1:
        consume(database)
    else:
        assert worker_database_factory is not None
        first_error: list[BaseException] = []
        error_lock = threading.Lock()
        database_lock = threading.Lock()
        active_databases: list[VectorDB] = []

        def parallel_worker() -> None:
            worker_database: VectorDB | None = None
            try:
                worker_database = worker_database_factory()
                with database_lock:
                    active_databases.append(worker_database)
                with worker_database:
                    consume(worker_database)
            except BaseException as exc:
                with error_lock:
                    if not first_error:
                        first_error.append(exc)
                stop.set()
            finally:
                with database_lock:
                    if worker_database is not None:
                        active_databases[:] = [
                            item for item in active_databases if item is not worker_database
                        ]

        def parallel_detail() -> str:
            with checkpoint_lock:
                checkpointed = len(completed & selected_relatives)
            with active_lock:
                active = len(active_shards)
            return (
                f"{checkpointed}/{len(shards)} shards checkpointed, "
                f"{active} active, {current_ingested():,} records this run"
            )

        def stop_parallel_workers() -> None:
            stop.set()
            with database_lock:
                databases = list(active_databases)
            for worker_database in databases:
                close = getattr(worker_database, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        pass
            cancel_transfers()

        worker_count = min(max_workers, len(pending))
        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="wikipedia-ingest",
        )
        futures = []
        interrupted = False
        try:
            futures = [executor.submit(parallel_worker) for _ in range(worker_count)]
            with progress_heartbeat(
                "parallel ingest",
                progress_interval,
                parallel_detail,
            ):
                for future in futures:
                    future.result()
        except KeyboardInterrupt:
            interrupted = True
            status("interrupt received; stopping all ingest workers")
            stop_parallel_workers()
            for future in futures:
                future.cancel()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        if interrupted:
            raise KeyboardInterrupt
        if first_error:
            raise first_error[0]
    return ingested
