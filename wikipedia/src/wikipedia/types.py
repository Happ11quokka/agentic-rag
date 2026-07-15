from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class WikipediaRecord:
    source_id: str
    url: str
    title: str
    text: str
    embedding: Sequence[float]


@dataclass(frozen=True, slots=True)
class SearchResult:
    source_id: str
    score: float
    url: str
    title: str
    text: str
