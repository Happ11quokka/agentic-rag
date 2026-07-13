from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wikipedia.dataset import DatasetError, WikipediaDataset


def test_sorted_shards_batching_and_limits(bundle: Path) -> None:
    dataset = WikipediaDataset(bundle)
    assert [path.name for path in dataset.shards()] == ["a.parquet", "b.parquet"]
    batches = list(dataset.iter_batches(batch_size=1, max_records=2))
    assert [[record.source_id for record in batch] for batch in batches] == [["a1"], ["a2"]]


def test_embedding_dimension_validation(bundle: Path) -> None:
    shard = bundle / "dataset" / "data" / "en" / "bad.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["bad"], "url": [""], "title": [""], "text": [""],
                "embedding": [[0.0] * 3],
            }
        ),
        shard,
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["shards"] = ["dataset/data/en/bad.parquet"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset = WikipediaDataset(bundle)
    with pytest.raises(DatasetError, match="dimension 3"):
        list(dataset.iter_batches())
