from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle with spaces"
    shard_dir = root / "dataset" / "data" / "en"
    shard_dir.mkdir(parents=True)
    (root / "models" / "bge-m3").mkdir(parents=True)
    (root / "state").mkdir()
    embedding = [0.0] * 1024
    for name, ids in (("b.parquet", ["b1"]), ("a.parquet", ["a1", "a2"])):
        table = pa.table(
            {
                "id": ids,
                "url": [f"https://example.test/{item}" for item in ids],
                "title": [item.upper() for item in ids],
                "text": [f"text {item}" for item in ids],
                "embedding": [embedding for _ in ids],
                "unused": [1 for _ in ids],
            }
        )
        pq.write_table(table, shard_dir / name)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "partial": True,
        "language": "en",
        "dataset": {
            "resolved_revision": "dataset-sha",
            "shards": ["dataset/data/en/b.parquet", "dataset/data/en/a.parquet"],
        },
        "model": {"resolved_revision": "model-sha", "path": "models/bge-m3"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root
