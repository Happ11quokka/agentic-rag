from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from .encoder import Encoder
from .types import EncoderConfig, SearchResult, WikipediaRecord


class VectorDB(ABC):
    dimension: int

    def __init__(self, encoder_config: EncoderConfig | None = None) -> None:
        self._encoder_config = encoder_config
        self._encoder: Encoder | None = None

    @abstractmethod
    def ensure_collection(self) -> None: ...

    @abstractmethod
    def upsert(self, records: Sequence[WikipediaRecord]) -> int: ...

    @abstractmethod
    def search_vector(self, vector: Any, *, limit: int = 10) -> list[SearchResult]: ...

    def _validated_vector(self, vector: Any) -> list[float]:
        array = np.asarray(vector, dtype=np.float32)
        if array.shape != (self.dimension,):
            raise ValueError(
                f"search vector must have shape ({self.dimension},), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("search vector contains non-finite values")
        return array.tolist()

    def search_text(self, text: str, *, limit: int = 10) -> list[SearchResult]:
        if self._encoder is None:
            config = self._encoder_config or EncoderConfig.from_bundle()
            self._encoder = Encoder(config)
        return self.search_vector(self._encoder.encode_query(text), limit=limit)

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> VectorDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
