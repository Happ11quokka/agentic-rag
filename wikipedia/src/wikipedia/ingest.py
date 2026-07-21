from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .base import VectorDB
from .bundle import BundlePaths, atomic_json, load_manifest
from .download import download_dataset_shard, progress_heartbeat, status
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
) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_shards is not None and max_shards < 1:
        raise ValueError("max_shards must be at least 1")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be at least 1")

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
    for shard in shards:
        try:
            relative = str(shard.relative_to(paths.bundle_dir))
            dataset_filename = str(shard.relative_to(paths.bundle_dir / "dataset"))
        except ValueError as exc:
            raise ValueError(
                f"Dataset shard is outside bundle directory: {shard}"
            ) from exc
        if relative in completed:
            shard.unlink(missing_ok=True)
        else:
            pending.append((shard, relative, dataset_filename))

    retained = next((shard for shard, _, _ in pending if shard.is_file()), None)
    removed = 0
    for local_shard in paths.dataset_dir.rglob("*.parquet"):
        if retained is None or local_shard.resolve() != retained:
            local_shard.unlink()
            removed += 1
    if removed:
        status(f"deleted {removed} surplus local shards to enforce one-shard storage")

    database.ensure_collection()
    ingested = 0

    for shard_index, (shard, relative, dataset_filename) in enumerate(pending, 1):
        if not shard.is_file():
            download_dataset_shard(
                paths,
                dataset_filename,
                revision=fingerprint["dataset_revision"],
                progress_interval=progress_interval,
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
                f"{ingested:,} records this run"
            )

        status(
            f"ingesting shard {shard_index}/{len(pending)}: {dataset_filename}, "
            f"{shard_records:,} records"
        )
        try:
            with progress_heartbeat(
                f"ingest shard {shard_index}/{len(pending)}",
                progress_interval,
                detail,
            ):
                for batch in parquet.iter_batches(
                    batch_size=batch_size, columns=list(REQUIRED_COLUMNS)
                ):
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

                    if max_records is not None:
                        remaining = max_records - ingested
                        if remaining <= 0:
                            limited = True
                            break
                        if len(records) > remaining:
                            records = records[:remaining]
                            limited = True

                    acknowledged = database.upsert(records)
                    if acknowledged != len(records):
                        raise RuntimeError(
                            f"Backend acknowledged {acknowledged} of {len(records)} "
                            f"records in {relative}"
                        )
                    ingested += acknowledged
                    shard_progress["records"] += acknowledged
                    if max_records is not None and ingested >= max_records:
                        limited = True
                        break
        finally:
            parquet.close()

        if limited and shard_progress["records"] == shard_records:
            limited = False
        if limited:
            status(
                f"record limit reached; retaining incomplete shard {dataset_filename}"
            )
            break
        completed.add(relative)
        atomic_json(checkpoint, {**fingerprint, "completed_shards": sorted(completed)})
        shard.unlink()
        status(
            f"shard {shard_index}/{len(pending)} checkpointed and deleted: "
            f"{dataset_filename}"
        )
    return ingested
