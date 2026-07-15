from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from .types import SearchResult, WikipediaRecord


class VectorDB(ABC):
    dimension: int

    @abstractmethod
    def ensure_collection(self) -> None: ...

    @abstractmethod
    def upsert(self, records: Sequence[WikipediaRecord]) -> int: ...

    @abstractmethod
    def search(self, vector: Any, *, limit: int = 10) -> list[SearchResult]: ...

    def _validated_vector(self, vector: Any) -> list[float]:
        array = np.asarray(vector, dtype=np.float32)
        if array.shape != (self.dimension,):
            raise ValueError(
                f"search vector must have shape ({self.dimension},), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("search vector contains non-finite values")
        return array.tolist()

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> VectorDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
