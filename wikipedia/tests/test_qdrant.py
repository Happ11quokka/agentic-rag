from types import SimpleNamespace

import pytest
from qdrant_client import models

from wikipedia.qdrant import QdrantConfig, QdrantVectorDB


class FakeClient:
    def __init__(self, vectors: object | None = None) -> None:
        self.vectors = vectors
        self.created: dict[str, object] | None = None

    def collection_exists(self, name: str) -> bool:
        return self.vectors is not None

    def create_collection(self, **kwargs: object) -> None:
        self.created = kwargs

    def get_collection(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=self.vectors))
        )


def test_ensure_collection_creates_float16_vectors() -> None:
    client = FakeClient()
    database = QdrantVectorDB(QdrantConfig(float16=True), client=client)

    database.ensure_collection()

    assert client.created is not None
    vectors = client.created["vectors_config"]
    assert vectors.datatype is models.Datatype.FLOAT16


def test_ensure_collection_rejects_float32_for_float16_config() -> None:
    vectors = SimpleNamespace(
        size=1024,
        distance=models.Distance.DOT,
        datatype=None,
    )
    database = QdrantVectorDB(
        QdrantConfig(float16=True),
        client=FakeClient(vectors),
    )

    with pytest.raises(ValueError, match="datatype=float32"):
        database.ensure_collection()


def test_ensure_collection_accepts_matching_float16_collection() -> None:
    vectors = SimpleNamespace(
        size=1024,
        distance=models.Distance.DOT,
        datatype=models.Datatype.FLOAT16,
    )
    database = QdrantVectorDB(
        QdrantConfig(float16=True),
        client=FakeClient(vectors),
    )

    database.ensure_collection()
