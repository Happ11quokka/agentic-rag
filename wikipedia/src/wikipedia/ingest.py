from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .base import VectorDB
from .bundle import BundlePaths, atomic_json
from .dataset import WikipediaDataset
from .types import IngestConfig


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


def ingest_dataset(
    database: VectorDB,
    dataset: WikipediaDataset,
    *,
    backend: str,
    endpoint: str,
    collection: str,
    metric: str,
    config: IngestConfig = IngestConfig(),
) -> int:
    fingerprint = {
        "schema_version": 1,
        "dataset_revision": dataset.revision,
        "backend": backend,
        "endpoint_fingerprint": endpoint_fingerprint(endpoint),
        "collection": collection,
        "dimension": database.dimension,
        "metric": metric.upper(),
    }
    checkpoint = config.checkpoint_path or default_checkpoint_path(
        dataset.paths, backend, collection, endpoint
    )
    checkpoint = Path(checkpoint).expanduser().resolve()
    completed = _load_checkpoint(checkpoint, fingerprint)
    database.ensure_collection()
    ingested = 0

    for shard in dataset.shards(config.max_shards):
        relative = str(shard.relative_to(dataset.paths.bundle_dir))
        if relative in completed:
            continue
        limited = False
        for records in dataset.iter_shard_batches(shard, batch_size=config.batch_size):
            if config.max_records is not None:
                remaining = config.max_records - ingested
                if remaining <= 0:
                    limited = True
                    break
                if len(records) > remaining:
                    records = records[:remaining]
                    limited = True
            acknowledged = database.upsert(records)
            if acknowledged != len(records):
                raise RuntimeError(
                    f"Backend acknowledged {acknowledged} of {len(records)} records in {relative}"
                )
            ingested += acknowledged
            if config.max_records is not None and ingested >= config.max_records:
                limited = True
                break
        if limited:
            break
        completed.add(relative)
        atomic_json(checkpoint, {**fingerprint, "completed_shards": sorted(completed)})
    return ingested
