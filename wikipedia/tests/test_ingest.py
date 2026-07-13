from __future__ import annotations

import json
from pathlib import Path

import pytest

from wikipedia.base import VectorDB
from wikipedia.dataset import WikipediaDataset
from wikipedia.ingest import CheckpointError, ingest_dataset
from wikipedia.types import IngestConfig, SearchResult, WikipediaRecord


class FakeDB(VectorDB):
    dimension = 1024

    def __init__(self, fail=False):
        super().__init__()
        self.ids = []
        self.fail = fail

    def ensure_collection(self): pass

    def upsert(self, records):
        if self.fail:
            raise RuntimeError("backend failed")
        self.ids.extend(record.source_id for record in records)
        return len(records)

    def search_vector(self, vector, *, limit=10) -> list[SearchResult]: return []

    def close(self): pass


def test_resume_skips_completed_shards(bundle: Path) -> None:
    dataset = WikipediaDataset(bundle)
    checkpoint = bundle / "state" / "custom.json"
    first = FakeDB()
    assert ingest_dataset(
        first, dataset, backend="fake", endpoint="local", collection="c", metric="IP",
        config=IngestConfig(batch_size=1, checkpoint_path=checkpoint),
    ) == 3
    second = FakeDB()
    assert ingest_dataset(
        second, dataset, backend="fake", endpoint="local", collection="c", metric="IP",
        config=IngestConfig(checkpoint_path=checkpoint),
    ) == 0
    assert second.ids == []


def test_limited_and_failed_shards_not_completed(bundle: Path) -> None:
    dataset = WikipediaDataset(bundle)
    limited_path = bundle / "state" / "limited.json"
    ingest_dataset(
        FakeDB(), dataset, backend="fake", endpoint="local", collection="c", metric="IP",
        config=IngestConfig(checkpoint_path=limited_path, max_records=1),
    )
    assert not limited_path.exists()
    failed_path = bundle / "state" / "failed.json"
    with pytest.raises(RuntimeError, match="backend failed"):
        ingest_dataset(
            FakeDB(fail=True), dataset, backend="fake", endpoint="local", collection="c",
            metric="IP", config=IngestConfig(checkpoint_path=failed_path),
        )
    assert not failed_path.exists()


def test_checkpoint_mismatch_rejected(bundle: Path) -> None:
    dataset = WikipediaDataset(bundle)
    checkpoint = bundle / "state" / "checkpoint.json"
    ingest_dataset(
        FakeDB(), dataset, backend="fake", endpoint="one", collection="c", metric="IP",
        config=IngestConfig(checkpoint_path=checkpoint),
    )
    with pytest.raises(CheckpointError, match="mismatch"):
        ingest_dataset(
            FakeDB(), dataset, backend="fake", endpoint="two", collection="c", metric="IP",
            config=IngestConfig(checkpoint_path=checkpoint),
        )
