from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .base import VectorDB
from .types import SearchResult, WikipediaRecord

DEFAULT_INDEX_TYPE = "DISKANN"
DEFAULT_SEARCH_LIST = 100


@dataclass(frozen=True, slots=True)
class MilvusConfig:
    uri: str = "http://localhost:19530"
    token: str | None = None
    user: str | None = None
    password: str | None = None
    database: str = "default"
    collection_name: str = "wikipedia_2024_06_bge_m3_en_v1"
    timeout: float = 60.0
    load_timeout: float = 1800.0
    index_timeout: float = 14400.0
    search_timeout: float = 3600.0
    # Milvus implements upsert as delete + insert, so a bulk load into a fresh
    # collection writes one delete tombstone per record and then spends the disk
    # compacting them away. Measured on the external HDD: ingesting 200k records
    # produced ~188k delete entries whose compaction saturated the disk and
    # starved the shard downloads. Insert is the correct write for a load that
    # has nothing to overwrite; enable upsert only when resuming into rows that
    # may already exist.
    upsert_existing: bool = False
    # Milvus caps VARCHAR at 65535 bytes and has no larger string type, so a
    # longer chunk cannot be stored intact. Truncation is opt-in and counted
    # rather than silent, because it alters the payload the retriever returns.
    truncate_text: bool = False
    consistency_level: str = "Bounded"
    dimension: int = 1024
    metric_type: str = "IP"
    index_type: str = DEFAULT_INDEX_TYPE
    index_params: Mapping[str, Any] = field(default_factory=dict)
    search_params: Mapping[str, Any] = field(
        default_factory=lambda: {"search_list": DEFAULT_SEARCH_LIST}
    )
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
        *,
        client: Any = None,
    ) -> None:
        self.config = config
        self.dimension = config.dimension
        self.truncated_records = 0
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

    def _check_bytes(
        self, value: str, maximum: int, field_name: str, source_id: str
    ) -> str:
        encoded = value.encode("utf-8")
        if len(encoded) <= maximum:
            return value
        if not (self.config.truncate_text and field_name == "text"):
            raise ValueError(
                f"Milvus {field_name} exceeds {maximum} bytes for source_id={source_id!r} "
                f"({len(encoded)} bytes); content was not truncated. Milvus caps VARCHAR "
                f"at 65535 bytes and has no larger string type — pass --milvus-truncate-text "
                f"to store these records with the payload cut to the limit"
            )
        self.truncated_records += 1
        # Cut on a UTF-8 boundary: slicing bytes can split a multi-byte character.
        return encoded[:maximum].decode("utf-8", errors="ignore")

    def upsert(self, records: Sequence[WikipediaRecord]) -> int:
        if not records:
            return 0
        rows = []
        for record in records:
            self._check_bytes(record.source_id, self.config.id_max_bytes, "id", record.source_id)
            self._check_bytes(record.url, self.config.url_max_bytes, "url", record.source_id)
            self._check_bytes(record.title, self.config.title_max_bytes, "title", record.source_id)
            text = self._check_bytes(
                record.text, self.config.text_max_bytes, "text", record.source_id
            )
            rows.append(
                {
                    "id": record.source_id,
                    "url": record.url,
                    "title": record.title,
                    "text": text,
                    "embedding": list(record.embedding),
                }
            )
        write = (
            self.client.upsert if self.config.upsert_existing else self.client.insert
        )
        response = write(
            collection_name=self.config.collection_name,
            data=rows,
            timeout=self.config.timeout,
        )
        acknowledged = response.get("upsert_count", response.get("insert_count", len(rows)))
        return int(acknowledged)

    def flush(self) -> None:
        """Seal growing segments.

        Milvus only builds indexes for sealed segments, so ingestion that ends
        without a flush leaves the newest data searchable by brute force alone —
        the on-disk index never covers it.
        """
        self.client.flush(self.config.collection_name, timeout=self.config.index_timeout)

    def index_state(self) -> dict[str, Any]:
        detail = self.client.describe_index(
            self.config.collection_name, "embedding", timeout=self.config.timeout
        )
        return {
            "index_type": str(detail.get("index_type", "")).upper(),
            "state": str(detail.get("state", "")),
            "total_rows": int(detail.get("total_rows") or 0),
            "indexed_rows": int(detail.get("indexed_rows") or 0),
            "pending_rows": int(detail.get("pending_index_rows") or 0),
            "reason": str(detail.get("index_state_fail_reason", "")),
        }

    def wait_for_index(
        self,
        *,
        timeout: float | None = None,
        poll: float = 15.0,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Block until every row is covered by the index.

        Neither ``state`` nor ``pending_rows`` is usable on its own. Measured on
        Milvus 2.5.27: before any segment is built it reports ``state=Finished``
        with ``indexed_rows=0``, so trusting the state benchmarks brute-force
        scans; once every row is built it reports ``pending_index_rows`` equal to
        the total, so waiting for pending to reach 0 never returns. Only
        ``indexed_rows`` moves monotonically with real progress, so completion is
        keyed on it reaching ``total_rows`` — confirmed on two consecutive polls,
        because compaction re-queues merged segments and can briefly perturb the
        counts.
        """
        limit = self.config.index_timeout if timeout is None else timeout
        deadline = time.monotonic() + limit
        state = self.index_state()
        confirmations = 0
        while True:
            if on_progress is not None:
                on_progress(state)
            if state["state"] == "Failed":
                raise RuntimeError(f"Milvus index build failed: {state['reason']}")
            if state["total_rows"] > 0 and state["indexed_rows"] >= state["total_rows"]:
                confirmations += 1
                if confirmations >= 2:
                    return state
            else:
                confirmations = 0
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Milvus index still building after {limit:.0f} s: "
                    f"indexed={state['indexed_rows']} of {state['total_rows']}"
                )
            time.sleep(poll)
            state = self.index_state()

    def load(self) -> None:
        """Pull the index into the query node.

        Worth calling explicitly before a benchmark: for an on-disk index this
        reads the index off the storage medium and takes far longer than a query,
        so leaving it to the first search reports load time as search latency.
        """
        if not self._loaded:
            self.client.load_collection(
                self.config.collection_name, timeout=self.config.load_timeout
            )
            self._loaded = True

    _load = load

    def search(self, vector: Any, *, limit: int = 10) -> list[SearchResult]:
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
            timeout=self.config.search_timeout,
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
