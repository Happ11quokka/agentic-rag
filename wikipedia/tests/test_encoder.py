from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from wikipedia.base import VectorDB
from wikipedia.encoder import Encoder
from wikipedia.types import EncoderConfig, SearchResult


def test_encoder_is_lazy_local_float32_and_normalized(monkeypatch) -> None:
    calls = []

    class FakeModel:
        def __init__(self, *args, **kwargs): calls.append((args, kwargs))

        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.ones((1, 1024), dtype=np.float64)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeModel)
    )
    encoder = Encoder(EncoderConfig("/local/model", revision="sha", device="cpu"))
    assert calls == []
    vector = encoder.encode_query("query")
    assert vector.dtype == np.float32
    assert calls[0][1] == {
        "revision": "sha", "device": "cpu", "local_files_only": True
    }
    assert calls[1][1]["normalize_embeddings"] is True


class DelegatingDB(VectorDB):
    dimension = 1024

    def ensure_collection(self): pass

    def upsert(self, records): return len(records)

    def search_vector(self, vector, *, limit=10):
        return [SearchResult("id", float(vector[0]), "", "", str(limit))]

    def close(self): pass


def test_text_search_delegates_to_vector(monkeypatch) -> None:
    database = DelegatingDB(EncoderConfig("/model"))
    monkeypatch.setattr(
        "wikipedia.base.Encoder.encode_query", lambda self, text: np.ones(1024, dtype=np.float32)
    )
    result = database.search_text("hello", limit=3)
    assert result[0].score == 1.0
    assert result[0].text == "3"
