from __future__ import annotations

from types import SimpleNamespace

import pytest

from wikipedia.milvus import MilvusConfig, MilvusVectorDB
from wikipedia.qdrant import QdrantConfig, QdrantVectorDB, point_id
from wikipedia.types import WikipediaRecord


def record(source_id: str = "arbitrary/string-id") -> WikipediaRecord:
    vector = [0.0] * 1024
    vector[0] = 1.0
    return WikipediaRecord(source_id, "https://example.test", "Title", "Full text", vector)


def test_qdrant_in_memory_round_trip_and_idempotence() -> None:
    config = QdrantConfig(on_disk=False, hnsw_on_disk=False)
    with QdrantVectorDB(config) as database:
        database.ensure_collection()
        assert database.upsert([record()]) == 1
        assert database.upsert([record()]) == 1
        results = database.search_vector(record().embedding)
        assert len(results) == 1
        assert results[0].source_id == "arbitrary/string-id"
        assert results[0].text == "Full text"
    assert point_id("x") == point_id("x")
    assert point_id("x") != point_id("y")


def test_qdrant_existing_collection_mismatch() -> None:
    database = QdrantVectorDB(QdrantConfig(on_disk=False, hnsw_on_disk=False))
    try:
        database.ensure_collection()
        database.config = QdrantConfig(
            collection_name=database.config.collection_name,
            dimension=3,
            on_disk=False,
            hnsw_on_disk=False,
        )
        database.dimension = 3
        with pytest.raises(ValueError, match="mismatch"):
            database.ensure_collection()
    finally:
        database.close()


class FakeMilvus:
    def __init__(self):
        self.rows = []
        self.loaded = 0

    def has_collection(self, name, **kwargs): return True

    def describe_collection(self, name, **kwargs):
        return {
            "fields": [
                {"name": "id", "is_primary": True, "type": 21, "params": {"max_length": 512}},
                {"name": "url", "type": 21, "params": {"max_length": 4096}},
                {"name": "title", "type": 21, "params": {"max_length": 4096}},
                {"name": "text", "type": 21, "params": {"max_length": 9}},
                {"name": "embedding", "type": 101, "params": {"dim": "1024"}},
            ]
        }

    def list_indexes(self, name, **kwargs): return ["embedding"]

    def describe_index(self, name, field, **kwargs):
        return {"metric_type": "IP", "index_type": "AUTOINDEX"}

    def upsert(self, collection_name, data, **kwargs):
        self.rows.extend(data)
        return {"upsert_count": len(data)}

    def load_collection(self, name, **kwargs): self.loaded += 1

    def search(self, **kwargs):
        return [[{"id": "id-1", "distance": 0.75, "entity": {
            "id": "id-1", "url": "u", "title": "t", "text": "body"
        }}]]

    def close(self): pass


def test_milvus_mapping_loading_and_byte_rejection() -> None:
    fake = FakeMilvus()
    database = MilvusVectorDB(MilvusConfig(text_max_bytes=9), client=fake)
    database.ensure_collection()
    assert database.upsert([record("id-1")]) == 1
    results = database.search_vector(record().embedding)
    database.search_vector(record().embedding)
    assert results[0].source_id == "id-1"
    assert results[0].text == "body"
    assert fake.loaded == 1
    oversized = WikipediaRecord("id-2", "u", "t", "ten-bytes!", [0.0] * 1024)
    with pytest.raises(ValueError, match="not truncated"):
        database.upsert([oversized])
