from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import VectorDB
from .types import SearchResult, WikipediaRecord

POINT_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL, "https://huggingface.co/datasets/Upstash/wikipedia-2024-06-bge-m3/en"
)


@dataclass(frozen=True, slots=True)
class QdrantConfig:
    url: str | None = None
    path: str | Path | None = None
    api_key: str | None = None
    collection_name: str = "wikipedia_2024_06_bge_m3_en_v1"
    prefer_grpc: bool = False
    grpc_port: int | None = None
    timeout: float = 60.0
    dimension: int = 1024
    distance: str = "DOT"
    on_disk: bool = True
    hnsw_on_disk: bool = True

    def __post_init__(self) -> None:
        if self.url and self.path:
            raise ValueError("Specify either Qdrant url or path, not both")
        if self.dimension < 1:
            raise ValueError("dimension must be positive")
        if self.distance.upper() not in {"DOT", "COSINE", "EUCLID", "MANHATTAN"}:
            raise ValueError(f"Unsupported Qdrant distance: {self.distance}")

    @property
    def endpoint(self) -> str:
        return self.url or str(Path(self.path).resolve() if self.path else ":memory:")


def point_id(source_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, source_id))


class QdrantVectorDB(VectorDB):
    def __init__(
        self,
        config: QdrantConfig,
        *,
        client: Any = None,
    ) -> None:
        self.config = config
        self.dimension = config.dimension
        if client is None:
            from qdrant_client import QdrantClient

            kwargs: dict[str, Any] = {
                "api_key": config.api_key,
                "prefer_grpc": config.prefer_grpc,
                "timeout": config.timeout,
            }
            if config.grpc_port is not None:
                kwargs["grpc_port"] = config.grpc_port
            if config.path is not None:
                kwargs["path"] = str(Path(config.path).expanduser().resolve())
            elif config.url is not None:
                kwargs["url"] = config.url
            else:
                kwargs["location"] = ":memory:"
            client = QdrantClient(**kwargs)
        self.client = client

    def ensure_collection(self) -> None:
        from qdrant_client import models

        name = self.config.collection_name
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=getattr(models.Distance, self.config.distance.upper()),
                    on_disk=self.config.on_disk,
                ),
                hnsw_config=models.HnswConfigDiff(on_disk=self.config.hnsw_on_disk),
            )
            return
        info = self.client.get_collection(name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            raise ValueError("Existing Qdrant collection uses named vectors; expected one vector")
        actual_distance = getattr(vectors.distance, "value", vectors.distance)
        if vectors.size != self.dimension or str(actual_distance).upper() != self.config.distance.upper():
            raise ValueError(
                f"Existing Qdrant collection mismatch: dimension={vectors.size}, "
                f"distance={actual_distance}; expected dimension={self.dimension}, "
                f"distance={self.config.distance.upper()}"
            )

    def upsert(self, records: Sequence[WikipediaRecord]) -> int:
        from qdrant_client import models

        if not records:
            return 0
        points = [
            models.PointStruct(
                id=point_id(record.source_id),
                vector=list(record.embedding),
                payload={
                    "source_id": record.source_id,
                    "url": record.url,
                    "title": record.title,
                    "text": record.text,
                },
            )
            for record in records
        ]
        response = self.client.upsert(
            collection_name=self.config.collection_name, points=points, wait=True
        )
        status = getattr(response, "status", None)
        if status is not None and str(getattr(status, "value", status)).lower() != "completed":
            raise RuntimeError(f"Qdrant upsert was not acknowledged as completed: {status}")
        return len(records)

    def search(self, vector: Any, *, limit: int = 10) -> list[SearchResult]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        response = self.client.query_points(
            collection_name=self.config.collection_name,
            query=self._validated_vector(vector),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(response, "points", response)
        results: list[SearchResult] = []
        for point in points:
            payload = point.payload or {}
            results.append(
                SearchResult(
                    source_id=str(payload.get("source_id", "")),
                    score=float(point.score),
                    url=str(payload.get("url", "")),
                    title=str(payload.get("title", "")),
                    text=str(payload.get("text", "")),
                )
            )
        return results

    def close(self) -> None:
        self.client.close()
