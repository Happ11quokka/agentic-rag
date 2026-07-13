from __future__ import annotations

import wikipedia.cli as module


class FakeDatabase:
    dimension = 1024

    def __init__(self, config):
        self.config = config

    def __enter__(self): return self

    def __exit__(self, *args): pass


def test_qdrant_cli_uses_environment_and_bounds_without_printing_secret(
    monkeypatch, capsys
) -> None:
    captured = {}
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "do-not-print")
    monkeypatch.setenv("QDRANT_COLLECTION", "wiki-test")
    monkeypatch.setattr(module, "WikipediaDataset", lambda bundle_dir: "marker-dataset")
    monkeypatch.setattr(module, "QdrantVectorDB", FakeDatabase)

    def ingest(database, dataset, **kwargs):
        captured.update(database=database, dataset=dataset, **kwargs)
        return 7

    monkeypatch.setattr(module, "ingest_dataset", ingest)
    module.main(["qdrant", "--max-shards", "2", "--max-records", "7"])
    assert captured["dataset"] == "marker-dataset"
    assert captured["database"].config.url == "http://qdrant:6333"
    assert captured["database"].config.collection_name == "wiki-test"
    assert captured["config"].max_shards == 2
    assert captured["config"].max_records == 7
    output = capsys.readouterr().out
    assert "do-not-print" not in output
    assert "endpoint_fingerprint=" in output


def test_milvus_cli_flags_override_environment(monkeypatch) -> None:
    captured = {}
    monkeypatch.setenv("MILVUS_URI", "http://environment:19530")
    monkeypatch.setenv("MILVUS_COLLECTION", "environment")
    monkeypatch.setattr(module, "WikipediaDataset", lambda bundle_dir: object())
    monkeypatch.setattr(module, "MilvusVectorDB", FakeDatabase)

    def ingest(database, dataset, **kwargs):
        captured.update(database=database, **kwargs)
        return 0

    monkeypatch.setattr(module, "ingest_dataset", ingest)
    module.main(
        ["milvus", "--uri", "http://flag:19530", "--collection", "flag", "--database", "db"]
    )
    config = captured["database"].config
    assert config.uri == "http://flag:19530"
    assert config.collection_name == "flag"
    assert config.database == "db"
