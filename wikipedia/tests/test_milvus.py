from typing import Any

import numpy as np
import pytest

from wikipedia.milvus import MilvusConfig, MilvusVectorDB
from wikipedia.types import WikipediaRecord


class FakeClient:
    def __init__(self, describe: list[dict[str, Any]] | None = None) -> None:
        self.inserted: list[list[dict[str, Any]]] = []
        self.upserted: list[list[dict[str, Any]]] = []
        self.flushed = 0
        self._describe = describe or []
        self.describe_calls = 0

    def insert(self, *, collection_name: str, data: list[Any], timeout: float) -> dict:
        self.inserted.append(data)
        return {"insert_count": len(data)}

    def upsert(self, *, collection_name: str, data: list[Any], timeout: float) -> dict:
        self.upserted.append(data)
        return {"upsert_count": len(data)}

    def flush(self, collection_name: str, timeout: float) -> None:
        self.flushed += 1

    def describe_index(
        self, collection_name: str, field: str, timeout: float
    ) -> dict[str, Any]:
        index = min(self.describe_calls, len(self._describe) - 1)
        self.describe_calls += 1
        return self._describe[index]


def _record(source_id: str = "a") -> WikipediaRecord:
    return WikipediaRecord(
        source_id=source_id,
        url="https://example.com",
        title="Aurora",
        text="Charged particles produce auroras.",
        embedding=np.zeros(1024, dtype=np.float32),
    )


def test_bulk_load_inserts_instead_of_upserting() -> None:
    client = FakeClient()
    database = MilvusVectorDB(MilvusConfig(), client=client)

    assert database.upsert([_record()]) == 1
    assert len(client.inserted) == 1
    assert client.upserted == []


def test_upsert_existing_opts_into_idempotent_writes() -> None:
    client = FakeClient()
    database = MilvusVectorDB(MilvusConfig(upsert_existing=True), client=client)

    assert database.upsert([_record()]) == 1
    assert len(client.upserted) == 1
    assert client.inserted == []


def test_oversized_text_fails_loudly_by_default() -> None:
    database = MilvusVectorDB(MilvusConfig(), client=FakeClient())
    record = _record()
    huge = WikipediaRecord(
        source_id=record.source_id,
        url=record.url,
        title=record.title,
        text="a" * 70_000,
        embedding=record.embedding,
    )

    with pytest.raises(ValueError, match="--milvus-truncate-text"):
        database.upsert([huge])


def test_truncate_text_cuts_on_a_utf8_boundary_and_counts() -> None:
    client = FakeClient()
    config = MilvusConfig(truncate_text=True, text_max_bytes=10)
    database = MilvusVectorDB(config, client=client)
    record = _record()
    # 4 three-byte characters = 12 bytes; a naive 10-byte slice splits the fourth.
    multibyte = WikipediaRecord(
        source_id=record.source_id,
        url=record.url,
        title=record.title,
        text="가나다라",
        embedding=record.embedding,
    )

    database.upsert([multibyte])

    stored = client.inserted[0][0]["text"]
    assert stored == "가나다"
    assert len(stored.encode("utf-8")) <= 10
    assert database.truncated_records == 1


def test_wait_for_index_ignores_finished_state_while_rows_are_unindexed() -> None:
    client = FakeClient(
        describe=[
            {
                "index_type": "DISKANN",
                "state": "Finished",
                "total_rows": 200_000,
                "indexed_rows": 0,
                "pending_index_rows": 200_000,
            },
            {
                "index_type": "DISKANN",
                "state": "Finished",
                "total_rows": 200_000,
                "indexed_rows": 200_000,
                "pending_index_rows": 0,
            },
        ]
    )
    database = MilvusVectorDB(MilvusConfig(), client=client)

    final = database.wait_for_index(poll=0.0)

    assert final["indexed_rows"] == 200_000
    # Two polls to see the build finish, one more to confirm it held.
    assert client.describe_calls == 3


def test_wait_for_index_returns_when_pending_never_drops_to_zero() -> None:
    """Milvus 2.5.27 reports pending_index_rows == total_rows after completion."""
    client = FakeClient(
        describe=[
            {
                "index_type": "DISKANN",
                "state": "Finished",
                "total_rows": 1_000_000,
                "indexed_rows": 1_000_000,
                "pending_index_rows": 1_000_000,
            }
        ]
    )
    database = MilvusVectorDB(MilvusConfig(), client=client)

    final = database.wait_for_index(poll=0.0)

    assert final["indexed_rows"] == 1_000_000


def test_wait_for_index_requires_two_consecutive_confirmations() -> None:
    """Compaction re-queues merged segments, so a single reading can mislead."""
    client = FakeClient(
        describe=[
            {
                "index_type": "DISKANN",
                "state": "Finished",
                "total_rows": 100,
                "indexed_rows": 100,
                "pending_index_rows": 0,
            },
            {
                "index_type": "DISKANN",
                "state": "Finished",
                "total_rows": 200,
                "indexed_rows": 100,
                "pending_index_rows": 100,
            },
            {
                "index_type": "DISKANN",
                "state": "Finished",
                "total_rows": 200,
                "indexed_rows": 200,
                "pending_index_rows": 0,
            },
        ]
    )
    database = MilvusVectorDB(MilvusConfig(), client=client)

    final = database.wait_for_index(poll=0.0)

    assert final["total_rows"] == 200
    assert final["indexed_rows"] == 200


def test_wait_for_index_raises_on_failed_build() -> None:
    client = FakeClient(
        describe=[
            {
                "index_type": "DISKANN",
                "state": "Failed",
                "total_rows": 10,
                "indexed_rows": 0,
                "pending_index_rows": 10,
                "index_state_fail_reason": "out of disk",
            }
        ]
    )
    database = MilvusVectorDB(MilvusConfig(), client=client)

    with pytest.raises(RuntimeError, match="out of disk"):
        database.wait_for_index(poll=0.0)


def test_wait_for_index_times_out_rather_than_reporting_success() -> None:
    client = FakeClient(
        describe=[
            {
                "index_type": "DISKANN",
                "state": "Finished",
                "total_rows": 10,
                "indexed_rows": 0,
                "pending_index_rows": 10,
            }
        ]
    )
    database = MilvusVectorDB(MilvusConfig(), client=client)

    with pytest.raises(TimeoutError, match="still building"):
        database.wait_for_index(timeout=0.0, poll=0.0)


def test_flush_uses_the_index_timeout_not_the_ingest_timeout() -> None:
    client = FakeClient()
    database = MilvusVectorDB(MilvusConfig(), client=client)

    database.flush()

    assert client.flushed == 1
