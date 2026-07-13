from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .bundle import BundlePaths, load_manifest
from .types import WikipediaRecord

REQUIRED_COLUMNS = ("id", "url", "title", "text", "embedding")
EMBEDDING_DIMENSION = 1024


class DatasetError(ValueError):
    pass


class WikipediaDataset:
    def __init__(self, bundle_dir: str | Path | None = None) -> None:
        self.paths = BundlePaths.resolve(bundle_dir)
        self.manifest = load_manifest(self.paths, require_complete=True)

    @property
    def revision(self) -> str:
        return str(self.manifest.get("dataset", {}).get("resolved_revision", ""))

    def shards(self, max_shards: int | None = None) -> list[Path]:
        if max_shards is not None and max_shards < 1:
            raise ValueError("max_shards must be at least 1")
        declared = self.manifest.get("dataset", {}).get("shards", [])
        if not isinstance(declared, list) or not declared:
            raise DatasetError("Manifest contains no dataset shards")
        paths = sorted((self.paths.bundle_dir / str(item)).resolve() for item in declared)
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise DatasetError(f"Dataset shard is missing: {missing[0]}")
        return paths[:max_shards]

    def iter_shard_batches(
        self, shard: Path, *, batch_size: int = 256
    ) -> Iterator[list[WikipediaRecord]]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(shard)
        names = set(parquet.schema_arrow.names)
        missing = set(REQUIRED_COLUMNS) - names
        if missing:
            raise DatasetError(f"{shard.name} missing required columns: {sorted(missing)}")
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(REQUIRED_COLUMNS)):
            values: dict[str, list[Any]] = {
                name: batch.column(index).to_pylist()
                for index, name in enumerate(REQUIRED_COLUMNS)
            }
            records: list[WikipediaRecord] = []
            for row in range(batch.num_rows):
                source_id = str(values["id"][row] or "")
                if not source_id:
                    raise DatasetError(f"{shard.name} contains an empty source id")
                embedding = values["embedding"][row]
                if embedding is None or len(embedding) != EMBEDDING_DIMENSION:
                    size = "null" if embedding is None else len(embedding)
                    raise DatasetError(
                        f"{shard.name} source {source_id!r} has embedding dimension {size}; "
                        f"expected {EMBEDDING_DIMENSION}"
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
            yield records

    def iter_batches(
        self,
        *,
        batch_size: int = 256,
        max_shards: int | None = None,
        max_records: int | None = None,
    ) -> Iterator[list[WikipediaRecord]]:
        emitted = 0
        for shard in self.shards(max_shards):
            for records in self.iter_shard_batches(shard, batch_size=batch_size):
                if max_records is not None:
                    remaining = max_records - emitted
                    if remaining <= 0:
                        return
                    records = records[:remaining]
                if records:
                    emitted += len(records)
                    yield records
                if max_records is not None and emitted >= max_records:
                    return
