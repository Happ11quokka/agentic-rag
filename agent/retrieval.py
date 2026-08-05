from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from wikipedia.types import SearchResult


class TimedRetriever:
    """Measure local encoding and vector search as separate operations."""

    def __init__(
        self,
        encoder: Any,
        database: Any,
        *,
        top_k: int = 3,
        max_chars_per_result: int = 1400,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if max_chars_per_result < 1:
            raise ValueError("max_chars_per_result must be at least 1")
        self.encoder = encoder
        self.database = database
        self.top_k = top_k
        self.max_chars_per_result = max_chars_per_result
        self.clock_ns = clock_ns

    def search(self, query: str) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search query must be a nonempty string")
        encode_start = self.clock_ns()
        vector = self.encoder.encode(query)
        encode_end = self.clock_ns()
        search_start = self.clock_ns()
        raw_results = self.database.search(vector, limit=self.top_k)
        search_end = self.clock_ns()

        results = [
            self._result_value(rank, result)
            for rank, result in enumerate(raw_results, start=1)
        ]
        return {
            "query": query,
            "encode_start_ns": encode_start,
            "encode_end_ns": encode_end,
            "encode_duration_ms": (encode_end - encode_start) / 1_000_000,
            "search_start_ns": search_start,
            "search_end_ns": search_end,
            "search_duration_ms": (search_end - search_start) / 1_000_000,
            "results": results,
        }

    def _result_value(self, rank: int, result: SearchResult) -> dict[str, Any]:
        original = result.text
        shown = original[: self.max_chars_per_result]
        return {
            "rank": rank,
            "source_id": result.source_id,
            "title": result.title,
            "url": result.url,
            "score": result.score,
            "snippet": shown,
            "original_length": len(original),
            "truncated": len(shown) < len(original),
        }


def render_search_results(call: dict[str, Any]) -> str:
    """Render exactly the snippet values retained in the trace."""
    visible = [
        {
            "rank": item["rank"],
            "source_id": item["source_id"],
            "title": item["title"],
            "url": item["url"],
            "score": item["score"],
            "snippet": item["snippet"],
        }
        for item in call["results"]
    ]
    return "<search_results>\n" + json.dumps(
        visible, ensure_ascii=False, separators=(",", ":")
    ) + "\n</search_results>"
