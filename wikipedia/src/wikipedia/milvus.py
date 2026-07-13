from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .base import VectorDB
from .types import EncoderConfig, SearchResult, WikipediaRecord


@dataclass(frozen=True, slots=True)
class MilvusConfig:
    uri: str = "http://localhost:19530"
    token: str | None = None
    user: str | None = None
    password: str | None = None
    database: str = "default"
    collection_name: str = "wikipedia_2024_06_bge_m3_en_v1"
    timeout: float = 60.0
    consistency_level: str = "Bounded"
    dimension: int = 1024
    metric_type: str = "IP"
    index_type: str = "AUTOINDEX"
    index_params: Mapping[str, Any] = field(default_factory=dict)
    search_params: Mapping[str, Any] = field(default_factory=dict)
    id_max_bytes: int = 512
    url_max_bytes: int = 4096
    title_max_bytes: int = 4096
    text_max_bytes: int = 65535

    def __post_init__(self) -> None:
        if self.token and (self.user or self.password):
            raise ValueError("Specify Milvus token or user/password, not both")
        if self.dimension < 1:
            raise ValueError("dimension must be positive")
        if self.metric_type.upper() != "IP":
            raise ValueError("Baseline supports Milvus IP metric only")

    @property
    def endpoint(self) -> str:
        return f"{self.uri.rstrip('/')}|db={self.database}"


class MilvusVectorDB(VectorDB):
    def __init__(
        self,
        config: MilvusConfig,
        encoder_config: EncoderConfig | None = None,
        *,
        client: Any = None,
    ) -> None:
        super().__init__(encoder_config)
        self.config = config
        self.dimension = config.dimension
        self._loaded = False
        if client is None:
            from pymilvus import MilvusClient

            kwargs: dict[str, Any] = {
                "uri": config.uri,
                "db_name": config.database,
                "timeout": config.timeout,
            }
            if config.token:
                kwargs["token"] = config.token
            elif config.user:
                kwargs.update(user=config.user, password=config.password or "")
            client = MilvusClient(**kwargs)
        self.client = client

    def ensure_collection(self) -> None:
        if not self.client.has_collection(
            self.config.collection_name, timeout=self.config.timeout
        ):
            self._create_collection()
            return
        description = self.client.describe_collection(
            self.config.collection_name, timeout=self.config.timeout
        )
        fields = {field["name"]: field for field in description.get("fields", [])}
        required = {"id", "url", "title", "text", "embedding"}
        if set(fields) != required:
            raise ValueError(f"Existing Milvus schema fields mismatch: {sorted(fields)}")
        primary = fields["id"]
        vector = fields["embedding"]
        params = vector.get("params", {})
        dimension = int(params.get("dim", vector.get("dim", 0)))
        if (
            not primary.get("is_primary")
            or description.get("auto_id", False)
            or dimension != self.dimension
        ):
            raise ValueError(
                f"Existing Milvus schema mismatch: primary={primary.get('is_primary')}, "
                f"auto_id={description.get('auto_id', False)}, dimension={dimension}; "
                f"expected primary id, auto_id=False, and dimension={self.dimension}"
            )
        self._validate_field_types_and_lengths(fields)
        indexes = self.client.list_indexes(
            self.config.collection_name, timeout=self.config.timeout
        )
        if "embedding" not in indexes:
            raise ValueError("Existing Milvus collection has no embedding index")
        detail = self.client.describe_index(
            self.config.collection_name, "embedding", timeout=self.config.timeout
        )
        metric = str(detail.get("metric_type", "")).upper()
        index_type = str(detail.get("index_type", "")).upper()
        if metric != self.config.metric_type.upper() or index_type != self.config.index_type.upper():
            raise ValueError(
                f"Existing Milvus index mismatch: metric={metric}, index_type={index_type}; "
                f"expected metric={self.config.metric_type}, index_type={self.config.index_type}"
            )

    def _validate_field_types_and_lengths(self, fields: Mapping[str, Any]) -> None:
        from pymilvus import DataType

        expected_types = {
            "id": DataType.VARCHAR,
            "url": DataType.VARCHAR,
            "title": DataType.VARCHAR,
            "text": DataType.VARCHAR,
            "embedding": DataType.FLOAT_VECTOR,
        }
        expected_lengths = {
            "id": self.config.id_max_bytes,
            "url": self.config.url_max_bytes,
            "title": self.config.title_max_bytes,
            "text": self.config.text_max_bytes,
        }
        for name, expected in expected_types.items():
            actual = fields[name].get("type", fields[name].get("datatype"))
            valid = actual == expected or str(actual).upper() in {
                expected.name,
                str(expected.value),
            }
            if not valid:
                raise ValueError(
                    f"Existing Milvus field {name!r} has type {actual}; expected {expected.name}"
                )
        for name, expected in expected_lengths.items():
            params = fields[name].get("params", {})
            actual = int(params.get("max_length", fields[name].get("max_length", 0)))
            if actual != expected:
                raise ValueError(
                    f"Existing Milvus field {name!r} max_length={actual}; expected {expected}"
                )

    def _create_collection(self) -> None:
        from pymilvus import DataType

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="id", datatype=DataType.VARCHAR, is_primary=True,
            max_length=self.config.id_max_bytes,
        )
        schema.add_field(
            field_name="url", datatype=DataType.VARCHAR, max_length=self.config.url_max_bytes
        )
        schema.add_field(
            field_name="title", datatype=DataType.VARCHAR, max_length=self.config.title_max_bytes
        )
        schema.add_field(
            field_name="text", datatype=DataType.VARCHAR, max_length=self.config.text_max_bytes
        )
        schema.add_field(
            field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dimension
        )
        indexes = self.client.prepare_index_params()
        indexes.add_index(
            field_name="embedding",
            index_type=self.config.index_type,
            metric_type=self.config.metric_type,
            params=dict(self.config.index_params),
        )
        self.client.create_collection(
            collection_name=self.config.collection_name,
            schema=schema,
            index_params=indexes,
            consistency_level=self.config.consistency_level,
            timeout=self.config.timeout,
        )

    @staticmethod
    def _check_bytes(value: str, maximum: int, field_name: str, source_id: str) -> None:
        size = len(value.encode("utf-8"))
        if size > maximum:
            raise ValueError(
                f"Milvus {field_name} exceeds {maximum} bytes for source_id={source_id!r} "
                f"({size} bytes); content was not truncated"
            )

    def upsert(self, records: Sequence[WikipediaRecord]) -> int:
        if not records:
            return 0
        rows = []
        for record in records:
            self._check_bytes(record.source_id, self.config.id_max_bytes, "id", record.source_id)
            self._check_bytes(record.url, self.config.url_max_bytes, "url", record.source_id)
            self._check_bytes(record.title, self.config.title_max_bytes, "title", record.source_id)
            self._check_bytes(record.text, self.config.text_max_bytes, "text", record.source_id)
            rows.append(
                {
                    "id": record.source_id,
                    "url": record.url,
                    "title": record.title,
                    "text": record.text,
                    "embedding": list(record.embedding),
                }
            )
        response = self.client.upsert(
            collection_name=self.config.collection_name,
            data=rows,
            timeout=self.config.timeout,
        )
        acknowledged = response.get("upsert_count", response.get("insert_count", len(rows)))
        return int(acknowledged)

    def _load(self) -> None:
        if not self._loaded:
            self.client.load_collection(
                self.config.collection_name, timeout=self.config.timeout
            )
            self._loaded = True

    def search_vector(self, vector: Any, *, limit: int = 10) -> list[SearchResult]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self._load()
        response = self.client.search(
            collection_name=self.config.collection_name,
            data=[self._validated_vector(vector)],
            limit=limit,
            output_fields=["id", "url", "title", "text"],
            search_params={
                "metric_type": self.config.metric_type,
                "params": dict(self.config.search_params),
            },
            consistency_level=self.config.consistency_level,
            timeout=self.config.timeout,
        )
        results: list[SearchResult] = []
        for hit in response[0] if response else []:
            entity = hit.get("entity", hit)
            results.append(
                SearchResult(
                    source_id=str(entity.get("id", hit.get("id", ""))),
                    score=float(hit.get("distance", hit.get("score", 0.0))),
                    url=str(entity.get("url", "")),
                    title=str(entity.get("title", "")),
                    text=str(entity.get("text", "")),
                )
            )
        return results

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close:
            close()
