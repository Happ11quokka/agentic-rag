from __future__ import annotations

import os
import uuid

import pytest

from wikipedia import MilvusConfig, MilvusVectorDB, WikipediaRecord


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("MILVUS_TEST_URI"), reason="MILVUS_TEST_URI is not set")
def test_live_milvus_round_trip() -> None:
    collection = f"wikipedia_vectordb_test_{uuid.uuid4().hex}"
    config = MilvusConfig(
        uri=os.environ["MILVUS_TEST_URI"],
        token=os.environ.get("MILVUS_TEST_TOKEN") or os.environ.get("MILVUS_TOKEN"),
        database=os.environ.get("MILVUS_TEST_DB_NAME", "default"),
        collection_name=collection,
    )
    vector = [0.0] * 1024
    vector[0] = 1.0
    database = MilvusVectorDB(config)
    try:
        database.ensure_collection()
        database.upsert(
            [WikipediaRecord("live-id", "https://example.test", "Live", "test", vector)]
        )
        results = database.search_vector(vector, limit=1)
        assert results[0].source_id == "live-id"
    finally:
        database.client.drop_collection(collection_name=collection, timeout=config.timeout)
        database.close()
